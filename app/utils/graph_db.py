

import os
import json
import logging
import tempfile
import asyncio
from typing import Dict, List, Any, Set
from config import (OPENAI_API_KEY, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
import pandas as pd
import requests
from neo4j import AsyncGraphDatabase
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(funcName)s - %(message)s")
logger = logging.getLogger(__name__)

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
# NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
# NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
OPENAI_MODEL = "gpt-4o"

_openai_client = None
_neo4j_driver = None


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if not _openai_client:
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


def get_neo4j_driver():
    global _neo4j_driver
    if not _neo4j_driver:
        _neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    return _neo4j_driver



UNIVERSAL_SCHEMA_PROMPT = """You are a graph database architect. Design an optimal Neo4j schema from tabular data.

DATA STATISTICS:
{data_stats}

GRAPH DESIGN PRINCIPLES:

1. PRIMARY ENTITIES = High Cardinality Unique Identifiers
   Rule: If unique_count / total_rows > 0.8 → PRIMARY ENTITY
   Example: order_id (8000 unique / 8000 rows) → ORDER node

2. CATEGORICAL ENTITIES = Low Cardinality Repeated Values  
   Rule: If unique_count / total_rows < 0.1 → CATEGORICAL ENTITY (separate node)
   Example: country (25 unique / 8000 rows) → COUNTRY node
   Why: Enables aggregation queries "Group by X", "Filter by Y"

3. MULTI-VALUED = Detect Delimiters
   Rule: If values contain "," or "|" or ";" → SPLIT into separate nodes
   Example: "Drama, Comedy" → GENRE nodes with relationships

4. PROPERTIES = Medium Cardinality or Numeric
   Rule: Everything else stays as properties on nodes
   Example: price, description, date_added

5. RELATIONSHIPS = Connect Entities
   Primary Entity connects TO categorical entities
   Format: (PRIMARY)-[ACTION]->(CATEGORY)

ALGORITHM:

FOR EACH COLUMN:
  cardinality_ratio = unique_count / total_rows
  
  IF cardinality_ratio > 0.8:
    → PRIMARY ENTITY (core node)
  
  ELIF cardinality_ratio < 0.1:
    IF contains_delimiters:
      → CATEGORICAL ENTITY (multi-valued, split)
    ELSE:
      → CATEGORICAL ENTITY (single-valued)
  
  ELSE:
    → PROPERTY (attribute on primary entity)

RETURN STRICT JSON:

{{
  "primary_entity": {{
    "label": "MAIN_ENTITY_LABEL",
    "id_column": "unique_identifier",
    "properties": ["prop1", "prop2"]
  }},
  "categorical_entities": [
    {{
      "label": "CATEGORY_LABEL",
      "source_column": "column_name",
      "is_multi_valued": true/false,
      "delimiter": "," or null
    }}
  ],
  "relationships": [
    {{
      "type": "RELATIONSHIP_TYPE",
      "from": "PRIMARY_LABEL",
      "to": "CATEGORY_LABEL",
      "via_column": "source_column"
    }}
  ]
}}

APPLY THIS ALGORITHM TO THE DATA ABOVE.
"""



async def download_excel(url: str) -> pd.DataFrame:
    """Download Excel/CSV"""
    logger.info(f"Downloading: {url}")
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, lambda: requests.get(url, timeout=60))
    response.raise_for_status()
    
    ext = os.path.splitext(url.split("?")[0].lower())[1]
    if ext not in [".csv", ".xlsx", ".xls"]:
        ext = ".csv"
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    
    try:
        df = await loop.run_in_executor(
            None, 
            pd.read_csv if ext == ".csv" else pd.read_excel, 
            tmp_path
        )
        logger.info(f" Loaded: {len(df)} rows × {len(df.columns)} columns")
        return df
    finally:
        os.unlink(tmp_path)


async def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and fill nulls"""
    df = df.copy()
    df.columns = [
        "".join(c if c.isalnum() or c == "_" else "" 
                for c in col.strip().lower().replace(" ", "_").replace("-", "_"))
        for col in df.columns
    ]
    
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("Unknown")
    
    return df




def analyze_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """Statistical analysis for schema inference"""
    total_rows = len(df)
    stats = {"total_rows": total_rows, "columns": []}
    
    for col in df.columns:
        unique_count = df[col].nunique()
        cardinality_ratio = unique_count / total_rows if total_rows > 0 else 0
        
        # Detect multi-valued fields
        contains_delimiter = False
        delimiter = None
        if df[col].dtype == "object":
            sample = df[col].dropna().astype(str).head(50)
            for delim in [",", "|", ";"]:
                if sample.str.contains(delim, regex=False).sum() > len(sample) * 0.3:
                    contains_delimiter = True
                    delimiter = delim
                    break
        
        col_stats = {
            "name": col,
            "dtype": str(df[col].dtype),
            "unique_count": unique_count,
            "cardinality_ratio": round(cardinality_ratio, 3),
            "null_count": int(df[col].isnull().sum()),
            "contains_delimiter": contains_delimiter,
            "delimiter": delimiter,
            "sample_values": df[col].dropna().astype(str).head(3).tolist()
        }
        
        stats["columns"].append(col_stats)
    
    return stats


async def infer_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """LLM-based universal schema inference"""
    logger.info("🧠 Analyzing schema...")
    
    client = get_openai_client()
    stats = analyze_columns(df)
    
    prompt = UNIVERSAL_SCHEMA_PROMPT.format(data_stats=json.dumps(stats, indent=2))
    
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    schema = json.loads(response.choices[0].message.content)
    
    logger.info(f"📊 Primary: {schema['primary_entity']['label']}")
    logger.info(f"🏷️  Categories: {len(schema['categorical_entities'])}")
    logger.info(f"🔗 Relationships: {len(schema['relationships'])}")
    
    return schema



async def execute_cypher(query: str, params: Dict = None) -> List[Dict]:
    """Execute Cypher query"""
    driver = get_neo4j_driver()
    async with driver.session() as session:
        result = await session.run(query, params or {})
        return [dict(record) async for record in result]


async def clear_database():
    """Clear entire graph"""
    logger.info("🗑️  Clearing database...")
    await execute_cypher("MATCH (n) DETACH DELETE n")


async def create_constraints(schema: Dict[str, Any]):
    """Create uniqueness constraints"""
    logger.info("🔐 Creating constraints...")
    
    # Primary entity
    primary = schema["primary_entity"]
    label = primary["label"].upper()  # ✅ FORCE UPPERCASE
    id_col = primary["id_column"]
    query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{id_col} IS UNIQUE"
    await execute_cypher(query)
    logger.info(f"   ✓ {label}.{id_col}")
    
    # Categorical entities
    for cat in schema["categorical_entities"]:
        label = cat["label"].upper()  # ✅ FORCE UPPERCASE
        query = f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE"
        await execute_cypher(query)
        logger.info(f"   ✓ {label}.name")


async def ingest_primary_entities(df: pd.DataFrame, schema: Dict[str, Any]):
    """Ingest main entity nodes"""
    logger.info("📦 Ingesting primary entities...")
    
    primary = schema["primary_entity"]
    label = primary["label"].upper()  # ✅ FORCE UPPERCASE
    id_col = primary["id_column"]
    props = [p for p in primary["properties"] if p in df.columns]
    
    df_entity = df[[id_col] + props].dropna(subset=[id_col]).drop_duplicates(subset=[id_col])
    
    logger.info(f"   → {label}: {len(df_entity)} nodes")
    
    # Batch insert
    batch_size = 500
    for i in range(0, len(df_entity), batch_size):
        batch = df_entity.iloc[i:i+batch_size]
        
        for _, row in batch.iterrows():
            props_set = ", ".join([f"n.{p} = ${p}" for p in props])
            set_clause = f"SET {props_set}" if props_set else ""
            query = f"MERGE (n:{label} {{{id_col}: ${id_col}}}) {set_clause}"
            
            params = {col: str(row[col]) if not pd.isna(row[col]) else None 
                     for col in [id_col] + props}
            await execute_cypher(query, params)
        
        logger.info(f"      {min(i+batch_size, len(df_entity))}/{len(df_entity)}")


async def ingest_categorical_entities(df: pd.DataFrame, schema: Dict[str, Any]):
    """Ingest categorical dimension nodes"""
    logger.info("🏷️  Ingesting categorical entities...")
    
    for cat in schema["categorical_entities"]:
        label = cat["label"].upper()  # ✅ FORCE UPPERCASE LABELS
        source_col = cat["source_column"]
        is_multi = cat.get("is_multi_valued", False)
        delimiter = cat.get("delimiter", ",")
        
        if source_col not in df.columns:
            continue
        
        # Collect unique values (preserve original case for display)
        unique_values: Set[str] = set()
        
        for val in df[source_col].dropna():
            val_str = str(val).strip()
            if is_multi and delimiter:
                parts = [p.strip() for p in val_str.split(delimiter) if p.strip()]
                unique_values.update(parts)
            else:
                if val_str and val_str != "Unknown":
                    unique_values.add(val_str)
        
        logger.info(f"   → {label}: {len(unique_values)} nodes")
        
        # Create nodes with UPPERCASE label
        for value in unique_values:
            query = f"MERGE (n:{label} {{name: $name}})"
            await execute_cypher(query, {"name": value})


async def ingest_relationships(df: pd.DataFrame, schema: Dict[str, Any]):
    """Create relationships between entities"""
    logger.info("🔗 Creating relationships...")
    
    primary = schema["primary_entity"]
    primary_label = primary["label"].upper()  # ✅ FORCE UPPERCASE
    primary_id_col = primary["id_column"]
    
    for rel in schema["relationships"]:
        rel_type = rel["type"].upper()  # ✅ FORCE UPPERCASE
        to_label = rel["to"].upper()  # ✅ FORCE UPPERCASE
        via_col = rel["via_column"]
        
        if via_col not in df.columns:
            continue
        
        # Find if this is multi-valued
        cat_entity = next((c for c in schema["categorical_entities"] 
                          if c["label"].upper() == to_label), None)
        
        if not cat_entity:
            continue
        
        is_multi = cat_entity.get("is_multi_valued", False)
        delimiter = cat_entity.get("delimiter", ",")
        
        logger.info(f"   → {rel_type}: {primary_label} → {to_label}")
        
        rel_count = 0
        for _, row in df[[primary_id_col, via_col]].dropna().iterrows():
            primary_id = str(row[primary_id_col])
            value_str = str(row[via_col]).strip()
            
            if not value_str or value_str == "Unknown":
                continue
            
            # Handle multi-valued
            values = []
            if is_multi and delimiter:
                values = [v.strip() for v in value_str.split(delimiter) if v.strip()]
            else:
                values = [value_str]
            
            # Create relationships
            for value in values:
                query = (
                    f"MATCH (a:{primary_label} {{{primary_id_col}: $primary_id}}) "
                    f"MATCH (b:{to_label} {{name: $value}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b)"
                )
                await execute_cypher(query, {"primary_id": primary_id, "value": value})
                rel_count += 1
        
        logger.info(f"      Created {rel_count} edges")




async def get_graph_schema() -> Dict[str, List]:
    """Get live schema from Neo4j"""
    labels = await execute_cypher("CALL db.labels() YIELD label RETURN collect(label) AS labels")
    rels = await execute_cypher("CALL db.relationshipTypes() YIELD relationshipType RETURN collect(relationshipType) AS rels")
    props = await execute_cypher("CALL db.propertyKeys() YIELD propertyKey RETURN collect(propertyKey) AS props")
    
    return {
        "labels": labels[0]["labels"] if labels else [],
        "relationships": rels[0]["rels"] if rels else [],
        "properties": props[0]["props"] if props else []
    }


async def get_schema_context() -> str:
    """Build detailed schema context with relationships and properties per node"""
    logger.info("🔍 Building schema context...")
    
    # Get relationship patterns
    rel_query = """
    MATCH (a)-[r]->(b)
    WITH labels(a)[0] AS from_label, type(r) AS rel_type, labels(b)[0] AS to_label
    RETURN DISTINCT from_label, rel_type, to_label
    LIMIT 100
    """
    relationships = await execute_cypher(rel_query)
    
    # Get sample properties for each label
    schema = await get_graph_schema()
    
    context_parts = []
    
    # Document relationship patterns
    if relationships:
        context_parts.append("RELATIONSHIP PATTERNS:")
        for rel in relationships:
            pattern = f"({rel['from_label']})-[{rel['rel_type']}]->({rel['to_label']})"
            context_parts.append(f"  - {pattern}")
    
    # Document node properties
    context_parts.append("\nNODE PROPERTIES:")
    for label in schema["labels"]:
        sample_query = f"MATCH (n:{label}) RETURN properties(n) AS props LIMIT 1"
        try:
            sample = await execute_cypher(sample_query)
            if sample and sample[0].get("props"):
                props = list(sample[0]["props"].keys())
                context_parts.append(f"  - {label}: {', '.join(props)}")
        except:
            context_parts.append(f"  - {label}: (properties unknown)")
    
    return "\n".join(context_parts)


TEXT_TO_CYPHER = """You are a Neo4j Cypher expert. Convert natural language to accurate Cypher queries.

GRAPH SCHEMA:
- Node Labels: {labels}
- Relationships: {relationships}  
- Properties: {properties}

SCHEMA STRUCTURE UNDERSTANDING:
{schema_context}

CRITICAL RULES:

1. ALWAYS start with MATCH to find nodes
2. For counting: Use COUNT() with aggregation
3. For listing: RETURN the node or properties
4. For text search: WHERE toLower(n.property) CONTAINS toLower('search_term')
5. For filtering by category: MATCH through relationships
6. ALWAYS add LIMIT 100 to prevent timeouts
7. Use property names exactly as listed (case-sensitive!)

COMMON QUERY PATTERNS:

Pattern 1 - Count categorical entities:
Q: "How many genres are there?"
A: MATCH (n:GENRE) RETURN COUNT(n) AS count

Pattern 2 - List categorical entities:
Q: "What are the top 5 countries?"
A: MATCH (c:COUNTRY)<-[:AVAILABLE_IN]-(s) RETURN c.name, COUNT(s) AS count ORDER BY count DESC LIMIT 5

Pattern 3 - Find by relationship traversal:
Q: "Which directors created the most content?"
A: MATCH (d:DIRECTOR)-[:DIRECTED]->(s) RETURN d.name, COUNT(s) AS count ORDER BY count DESC LIMIT 10

Pattern 4 - Search primary entities:
Q: "Find movies with 'love' in the title"
A: MATCH (s:SHOW) WHERE toLower(s.title) CONTAINS 'love' RETURN s.title LIMIT 100

Pattern 5 - Filter by category:
Q: "Show movies in the Drama genre"
A: MATCH (s:SHOW)-[:HAS_GENRE]->(g:GENRE) WHERE g.name = 'Drama' RETURN s.title LIMIT 100

Pattern 6 - Multi-hop traversal:
Q: "Which actors worked in comedies?"
A: MATCH (a:ACTOR)-[:ACTED_IN]->(s)-[:HAS_GENRE]->(g:GENRE) WHERE g.name = 'Comedy' RETURN DISTINCT a.name LIMIT 100

QUESTION: {question}

Return ONLY valid Cypher (no explanation, no markdown):
"""


async def generate_cypher(question: str) -> str:
    """Generate Cypher from question with rich schema context"""
    client = get_openai_client()
    schema = await get_graph_schema()
    schema_context = await get_schema_context()
    
    prompt = TEXT_TO_CYPHER.format(
        labels=schema["labels"],
        relationships=schema["relationships"],
        properties=schema["properties"],
        schema_context=schema_context,
        question=question
    )
    
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    cypher = response.choices[0].message.content.strip()
    cypher = cypher.replace("```cypher", "").replace("```", "").replace("```", "").strip()
    
    return cypher


async def answer_question(question: str) -> Dict[str, Any]:
    """Answer question using Graph RAG"""
    logger.info(f"{question}")
    
    cypher = await generate_cypher(question)
    logger.info(f"🔍 Cypher: {cypher}")
    
    try:
        results = await execute_cypher(cypher)
    except Exception as e:
        logger.error(f"Query failed: {e}")
        return {"question": question, "error": str(e)}
    
    # Generate answer
    client = get_openai_client()
    prompt = f"""Answer this question using the query results.

QUESTION: {question}

CYPHER: {cypher}

RESULTS: {json.dumps(results, indent=2, default=str)}

Provide a clear answer:
"""
    
    response = await client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    
    answer = response.choices[0].message.content.strip()
    
    return {
        "question": question,
        "cypher": cypher,
        "results": results,
        "answer": answer
    }




async def flow_build_graph_rag(excel_url: str) -> Dict[str, Any]:
    """
    UNIVERSAL GRAPH RAG BUILD FLOW
    Works with ANY tabular dataset
    """
    logger.info("=" * 80)
    logger.info("🚀 UNIVERSAL GRAPH RAG BUILD")
    logger.info("=" * 80)
    
    try:
        # 1. Download
        df = await download_excel(excel_url)
        
        # 2. Normalize
        df = await normalize_dataframe(df)
        
        # 3. Clear existing graph
        await clear_database()
        
        # 4. Infer schema
        schema = await infer_schema(df)
        
        # 5. Create constraints
        await create_constraints(schema)
        
        # 6. Ingest primary entities
        await ingest_primary_entities(df, schema)
        
        # 7. Ingest categorical entities
        await ingest_categorical_entities(df, schema)
        
        # 8. Create relationships
        await ingest_relationships(df, schema)
        
        logger.info("=" * 80)
        logger.info("✅ BUILD COMPLETE")
        logger.info("=" * 80)
        
        return {
            "status": "success",
            "rows": len(df),
            "schema": schema
        }
        
    except Exception as e:
        logger.error(f"❌ Build failed: {e}", exc_info=True)
        return {"status": "error", "error": str(e)}




# async def main():
#     """Example usage"""
    
#     # Netflix dataset
#     url = "https://kapture-p.s3.amazonaws.com/team_message_app/031125/hkbql/netflix_titles_CLEANED.csv"
    
#     # Build graph
#     report = await flow_build_graph_rag(url)
#     print(json.dumps(report, indent=2))
    
#     # Ask questions
#     if report["status"] == "success":
#         questions = [
#             "How many unique genres are there?",
#             "Which directors have created the most content?",
#             "What are the top 5 countries by content count?"
#         ]
        
#         for q in questions:
#             print(f"\n{'='*80}")
#             response = await answer_question(q)
#             print(f"Q: {response['question']}")
#             print(f"A: {response['answer']}")


# if __name__ == "__main__":
#     asyncio.run(main())