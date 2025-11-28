from typing import List,Tuple,Dict,Any
import json
from config import(
    embedding_model,cross_encoder
)
from pymilvus import AnnSearchRequest, RRFRanker
from config import milvus_client
import logging

logger = logging.getLogger(__name__)


async def vector_search(collection: str, query: str, top_k: int = 5, ef: int = 200) -> List[Tuple[str, str, float]]:
    """
    Performs vector search and returns a list of (text, sourceURL, distance) tuples.
    """
    q_emb = embedding_model.encode([query]).tolist()[0]
    
    results = milvus_client.search(
        collection_name=collection,
        anns_field="embedding",
        data=[q_emb],
        limit=top_k,
        search_params={"metric_type": "COSINE", "params": {"ef": ef}},
        output_fields=["text", "sourceURL", "start_time", "end_time"],
    )
    
    candidates = []
    for hits in results:
        for hit in hits:
            text = hit["entity"].get("text", "")
            source_url = hit["entity"].get("sourceURL", "")
            start_time = hit["entity"].get("start_time", 0.0)
            end_time = hit["entity"].get("end_time", 0.0)
            distance = hit.get("distance", 0.0)
            candidates.append({"text":text, "sourceURL":source_url, "start_time":start_time, "end_time":end_time, "distance":distance})
    
    return candidates

async def keyword_search_milvus(
    collection: str,
    query: str,
    top_k: int = 5
) -> List[Dict[str, object]]:
    """
    Performs BM25 full-text search on a Milvus collection.
    Returns a list of dicts with:
        - text
        - sourceURL
        - distance (BM25 score)
    """

    # Perform search directly on the sparse BM25 field

    search_params={"metric_type": "BM25", "params": {"k1": 1.2, "b": 0.75}}
    
    results = milvus_client.search(
        collection_name=collection,
        anns_field="sparse",        # BM25 sparse field
        data=[query],               # query text
        limit=top_k,
        search_params=search_params,
        output_fields=["text", "sourceURL"]
    )

    candidates = []
    for hits in results:  # results is List[List[dict]]
        for hit in hits:
            text = hit["entity"].get("text", "")
            source_url = hit["entity"].get("sourceURL", "")
            distance = hit.get("distance", 0.0)
            candidates.append({
                "text": text,
                "sourceURL": source_url,
                "distance": float(distance)
            })

    return candidates


async def hybrid_retrieve(collection_name: str, query: str, top_k: int = 20,top_rerank:int=4) -> Tuple[List[str], List[str]]:
    # texts, tokens, source_urls = await fetch_all_docs(collection=collection_name)

    # Get keyword and vector search results (both return list of dicts)
    # kw_results = await keyword_search(query=query, texts=texts, token_corpus=tokens, source_urls=source_urls, top_k=top_k)
    kw_results = await keyword_search_milvus(collection=collection_name,query=query, top_k=top_k)
    vec_results = await vector_search(collection=collection_name, query=query, top_k=top_k)

    # Combine results and deduplicate based on `text`
    seen = set()
    candidates = []
    for result in kw_results + vec_results:
        text = result["text"]
        source = result["sourceURL"]
        if text not in seen:
            seen.add(text)
            candidates.append((text, source))

    # Prepare for cross-encoder scoring
    pairs = [(query, text) for text, _ in candidates]
    ce_scores = cross_encoder.predict(pairs) #.ce_scores will provide scores 

    # Rerank based on score
    reranked = sorted(zip(candidates, ce_scores), key=lambda x: x[1], reverse=True)

    # Split into separate lists
    final_texts = [text for (text, _), _ in reranked[:top_rerank]]
    final_sources = [source for (_, source), _ in reranked[:top_rerank]]

    return final_texts, final_sources