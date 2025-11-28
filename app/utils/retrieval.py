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