from config import milvus_client
import logging
from pymilvus import FieldSchema, DataType, CollectionSchema, Function, FunctionType,Collection
from pymilvus import MilvusException
from pydantic import BaseModel,Field
from typing import List,Optional,Dict,Any
from app.utils.chunking import( get_embeddings,
                               bm25_tonized
)
from typing import List,Dict,Any
import json

logger = logging.getLogger(__name__)

async def create_milvus_collection(collection_name: str,description: str = "The Knowledgebase for") -> bool:
    """
    Creates collection with proper schema, BM25 function, and indexes.
    This is the RECOMMENDED approach - create indexes BEFORE data insertion.
    """
    try:
        # checking if collection already exists
        if milvus_client.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' already exists.")
            
        # defining fields
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=25_000, enable_analyzer=True),
            FieldSchema(name="keyword_text", dtype=DataType.ARRAY, element_type=DataType.VARCHAR, max_capacity=750, max_length=712),
            FieldSchema(name="sourceURL", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="sparse", dtype=DataType.SPARSE_FLOAT_VECTOR),  # BM25 output field
            FieldSchema(name="start_time", dtype=DataType.FLOAT),  # Add start_time field
            FieldSchema(name="end_time", dtype=DataType.FLOAT), 
        ]
        
        # create schema for collection
        schema = CollectionSchema(
            fields,
            description=f"{description} '{collection_name}'",
            enable_dynamic_field=True
        )
        
        # Add BM25 function to schema
        bm25_function = Function(
            name="text_bm25_emb",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
        schema.add_function(bm25_function)
        
        index_params = milvus_client.prepare_index_params()
        
        # Index for dense vector (semantic search)
        index_params.add_index(
            field_name="embedding",
            index_type="HNSW",
            metric_type="COSINE",
            params={"M": 16, "efConstruction": 200}
        )
        
        # Index for sparse vector (BM25 search)
        index_params.add_index(
            field_name="sparse",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
            params={
                "inverted_index_algo": "DAAT_MAXSCORE",
                "bm25_k1": 1.2,
                "bm25_b": 0.75
            }
        )
        
        # Create collection with schema and indexes
        milvus_client.create_collection(
            collection_name, 
            schema=schema, 
            index_params=index_params
        )
        
        # Load collection to make it available for search
        milvus_client.load_collection(collection_name=collection_name)
        
        logger.info(f"Collection '{collection_name}' created successfully with indexes")
        return collection_name
        
    except MilvusException as me:
        logger.error(f"Milvus error creating collection '{collection_name}': {me}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected error creating collection '{collection_name}': {e}", exc_info=True)
        return None
    

async def insert_data_to_collection(
    collection_name: str,
    chunked_data: List[Dict[str, Any]],
    batch_size: int = 1000   # <-- tune this depending on your embedding dimension
) -> bool:
    """
    The BM25 sparse vector will be automatically generated from the text field.
    """
    if not chunked_data:
        logger.warning("chunked_data is empty; skipping insert.")
        return True
    
    try:
        # Get tokenized corpus and embeddings
        tokenized_corpus = await bm25_tonized(chunked_data)
        embeddings = await get_embeddings(chunked_data)
        
        expected_length = len(chunked_data)
        logger.info(f"Preparing to insert {expected_length} documents")
        
        # Validate lengths
        if len(tokenized_corpus) != expected_length:
            raise ValueError(f"Length mismatch: tokenized_corpus has {len(tokenized_corpus)} items, expected {expected_length}")
        if len(embeddings) != expected_length:
            raise ValueError(f"Length mismatch: embeddings has {len(embeddings)} items, expected {expected_length}")
        
        # Build all data entries
        all_data = []
        for i in range(expected_length):
            all_data.append({
                "embedding": embeddings[i],
                "text": chunked_data[i]['text'],
                "keyword_text": tokenized_corpus[i],
                "sourceURL": chunked_data[i]['sourceURL'],
                "start_time": chunked_data[i].get('start_time', 0.0),
                "end_time": chunked_data[i].get('end_time', 0.0)
            })
        
        # Insert in batches
        total_inserted = 0
        for i in range(0, expected_length, batch_size):
            batch = all_data[i:i+batch_size]
            result = milvus_client.insert(
                collection_name=collection_name,
                data=batch
            )
            
            inserted_count = result["insert_count"]
            total_inserted += inserted_count
            logger.info(f"Batch inserted: {inserted_count} rows (running total: {total_inserted})")
        
        if total_inserted == expected_length:
            logger.info(f"Insert successful: {total_inserted} rows inserted and automatically indexed")
            return True
        else:
            logger.error(f"Insert count mismatch: inserted {total_inserted}, expected {expected_length}")
            return False

    except Exception as e:
        logger.error(f"Data insertion failed for collection '{collection_name}': {e}", exc_info=True)
        return False