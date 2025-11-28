from app import app
from app.utils.graph_db import clear_database, create_constraints, download_excel, infer_schema, ingest_categorical_entities, ingest_primary_entities, ingest_relationships, normalize_dataframe
from config import API_PREFIX, MONGO_DB_URI, MONGO_DB, milvus_client, MILVUS_URI, MILVUS_TOKEN
from app.database import (
            MONGO_DB_PDF_COLLECTION,
            MONGO_DB_COLLECTION_LIST,
            MONGO_DB_VIDEO_COLLECTION,
            MONGO_DB_IMAGE_COLLECTION,
            MONGO_DB_EXCEL_COLLECTION,
            insert_one_data)
from app.milvus import create_milvus_collection, insert_data_to_collection
from typing import List
from app.utils import (chunking_for_pdf,
                       convert_video_to_audio,
                       transcribe_audio,
                       process_and_store_transcript,
                       download_file,
                       caption_image,
                       extract_text_from_images,
                       process_and_store_content,
                       download_image_from_s3
                       )
from motor.motor_asyncio import AsyncIOMotorClient
from app.database.mongo_client import set_db
import logging
from datetime import datetime, UTC, time
from app.utils import generate_safe_collection_name, vector_search, keyword_search_milvus, hybrid_retrieve,answer_question
from pymilvus import Collection , connections
from typing import Dict, Any
from fastapi import APIRouter,HTTPException, Response

_client = AsyncIOMotorClient(MONGO_DB_URI)
_db = _client[MONGO_DB]

logging.info("MongoDB connected successfully.")
set_db(_db) 


try:
    connections.connect(
        alias="default",
        uri=MILVUS_URI,
        token=MILVUS_TOKEN
    )
    logging.info("PyMilvus connection established successfully.")
except Exception as e:
    logging.error(f"Failed to establish PyMilvus connection: {e}")


@app.get(f"{API_PREFIX}/health", tags = ["Health Check"])
async def health_check():
    return {"status": "ok", "message": "API is healthy and running."}

@app.get(f"{API_PREFIX}/list-collection", tags = ["Milvus Collection Info"])
async def list_collections():
    try:
        collections = milvus_client.list_collections()
        # collections is already a list of strings, no need to extract .name
        collection_names = collections if isinstance(collections, list) else []
        logging.info(f"Retrieved {len(collection_names)} collections from Milvus.")
        return {"status": "success", "collections": collection_names}
    
    except Exception as e:
        logging.error(f"Error listing collections: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}

@app.post(f"{API_PREFIX}/create-collection", tags = ["Milvus Collection"])
async def create_collection(collection_name: str):
    
    try:
        collection_name = generate_safe_collection_name(collection_name)
        await create_milvus_collection(collection_name)

        await insert_one_data(
            MONGO_DB_COLLECTION_LIST,
            {
                "collection_name": collection_name,
                "status":"created",
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC)
            }
        )

        logging.info(f"Collection '{collection_name}' created and logged in MongoDB.")
        return {"status": "success", "collection_name": collection_name}

    except Exception as e:

        await insert_one_data(
            MONGO_DB_COLLECTION_LIST,
            {
                "collection_name": collection_name,
                "status":"failed",
                "error": str(e),
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC)
            }
        )
        logging.error(f"Error creating collection: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post(f"{API_PREFIX}/pdf-insert-data", tags = ["Milvus Collection"])
async def pdf_insert_data(knowledge_base_id: str, pdf_links: List[str]):
    try:
        all_chunk_data = []
        chunked_data = await chunking_for_pdf(pdf_links)
        all_chunk_data.extend(chunked_data)

        result = await insert_data_to_collection(
                    collection_name=knowledge_base_id,
                    chunked_data=all_chunk_data
                )
        
        if result:
            logging.info(f"Inserted {len(all_chunk_data)} chunks into collection '{knowledge_base_id}' from PDF links.")

            await insert_one_data(
                MONGO_DB_PDF_COLLECTION,
                {
                    "knowledge_base_id": knowledge_base_id,
                    "pdf_links": pdf_links,
                    "status":"success",
                    "length": len(all_chunk_data),
                    "created_at": datetime.now(UTC),
                    "updated_at": datetime.now(UTC)
                }
            )
            return {"status": "success", "collection_name": knowledge_base_id}
        else:
            logging.error(f"Failed to insert data into collection '{knowledge_base_id}' from PDF links.")

            await insert_one_data(
            MONGO_DB_PDF_COLLECTION,
            {
                "knowledge_base_id": knowledge_base_id,
                "pdf_links": pdf_links,
                "status":"failed",
                "error": str(e),
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC)
            }
        )
            
            return {"status": "error", "message": "Failed to insert data into Milvus collection."}
        
    except Exception as e:

        logging.error(f"Error inserting PDF data: {e}", exc_info=True)

        await insert_one_data(
            MONGO_DB_PDF_COLLECTION,
            {
                "knowledge_base_id": knowledge_base_id,
                "pdf_links": pdf_links,
                "status":"failed",
                "error": str(e),
                "created_at": datetime.now(UTC),
                "updated_at": datetime.now(UTC)
            }
        )
        return {"status": "error", "message": str(e)}


@app.post(f"{API_PREFIX}/video-insert-data", tags=["Milvus Collection"])
async def video_insert_data(collection_name: str, video_urls: List[str]):
    try:

        for video_url in video_urls:
            video_path = download_file(video_url)

            audio_path = await convert_video_to_audio(video_path)

            transcript_text = await transcribe_audio(audio_path)

            chunked_data = await process_and_store_transcript(transcript_text, video_url)

            logging.info(f"Processed transcript into {len(chunked_data)} chunks.")

            result = await insert_data_to_collection(
                collection_name=collection_name,
                chunked_data=chunked_data
            )

            if result:
                logging.info(f"Inserted {len(chunked_data)} chunks into collection '{collection_name}' from video URL.")
                await insert_one_data(
                    MONGO_DB_VIDEO_COLLECTION,
                    {
                        "knowledge_base_id": collection_name,
                        "video_links": video_url,
                        "status": "success",
                        "length": len(chunked_data),
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC)
                    }
                )
                return {"status": "success", "collection_name": collection_name}
            else:
                logging.error(f"Failed to insert data into collection '{collection_name}' from video URL.")
                await insert_one_data(
                    MONGO_DB_VIDEO_COLLECTION,
                    {
                        "knowledge_base_id": collection_name,
                        "video_links": video_url,
                        "status": "failed",
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC)
                    }
                )
                return {"status": "error", "message": "Failed to insert data into Milvus collection."}

    except Exception as e:
        logging.error(f"Error inserting video data: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post(f"{API_PREFIX}/image-insert-data", tags = ["Milvus Collection"])
async def image_insert_data(collection_name: str, image_paths: List[str]):
    try:
        for image_path in image_paths:

            image = download_image_from_s3(image_path)

            caption = await caption_image(image)
            logging.info(f"Generated caption: {caption}")

            extracted_text = await extract_text_from_images(image)
            logging.info(f"Extracted text: {extracted_text}")

            chunked_data = process_and_store_content(caption, extracted_text, image_path)

            logging.info(f"Chunked data: {chunked_data}")

            result = await insert_data_to_collection(
                collection_name=collection_name,
                chunked_data=chunked_data
            )


            if result:
                logging.info(f"Inserted {len(chunked_data)} chunks into collection '{collection_name}' from image.")

                await insert_one_data(
                    MONGO_DB_IMAGE_COLLECTION,
                    {
                        "knowledge_base_id": collection_name,
                        "image_links": image_path,
                        "status": "success",
                        "length": len(chunked_data),
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC)
                    }
                )
                return {"status": "success", "collection_name": collection_name}
            else:
                logging.error(f"Failed to insert data into collection '{collection_name}' from image.")
                await insert_one_data(
                    MONGO_DB_IMAGE_COLLECTION,
                    {
                        "knowledge_base_id": collection_name,
                        "image_links": image_path,
                        "status": "failed",
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC)
                    }
                )
                return {"status": "error", "message": "Failed to insert data into Milvus collection."}

    except Exception as e:
        logging.error(f"Error inserting image data: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}


@app.post(f"{API_PREFIX}/image-insert-data", tags = ["NEO4J Collection"])
async def image_insert_data(collection_name: str, image_paths: List[str]):
    try:
       pass

    except Exception as e:
        logging.error(f"Error inserting image data: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}





@app.post(f"{API_PREFIX}/graph-rag", tags = ["Neo4j Graph RAG"])
async def flow_build_graph_rag(excel_url: str) -> Dict[str, Any]:
    """
    UNIVERSAL GRAPH RAG BUILD FLOW
    Works with ANY tabular dataset
    """

    
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
        
        await insert_one_data(
                    MONGO_DB_EXCEL_COLLECTION,
                    {
                        "excel_url": excel_url,
                        "status": "success",
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC)
                    })
        return {
            "status": "success",
            "rows": len(df),
            "schema": schema
        }
        
    except Exception as e:
        await insert_one_data(
                    MONGO_DB_EXCEL_COLLECTION,
                    {
                        "excel_url": excel_url,
                        "status": "failed",
                        "created_at": datetime.now(UTC),
                        "updated_at": datetime.now(UTC)
                    }
        )
        return {"status": "error", "error": str(e)}
    



#Search API
@app.post(f"{API_PREFIX}/vector-search", tags = ["Milvus Collection Retrieval"])
async def retrieve_data(collection_name, query):
    try:

        retrieved_data = await vector_search(
            collection=collection_name,
            query=query,
        )

        logging.info(f"Retrieved {len(retrieved_data)} results from collection '{collection_name}' for query.")

        return {"status":"success", "data": retrieved_data}

    except Exception as e:
        logging.error(f"Error retrieving data: {e}", exc_info=True)

        return {"status": "failed", "message": str(e)}
    

@app.post(f"{API_PREFIX}/keyword_search", summary="To retrive the documents from given collection via keyword search",
             description="It retrive most relavent docs")
async def keyword_retrival(collection_name, query):
    try:
        # collection_name = f"{payload.coll}_{payload.version}" if payload.version is not None else payload.coll
        # if not milvus_client.has_collection(collection_name):
        #     raise HTTPException(
        #         status_code=404,
        #         detail=f"Knowledge base version not found: '{collection_name}'"
        #     )
        
        kw = await keyword_search_milvus(collection=collection_name, query=query)

        return {"status_code": 200, "retrievedData": kw}
    except Exception as e:
        
        raise HTTPException(status_code=500, detail=str(e))

@app.post(f"{API_PREFIX}/hybrid_search",summary="To retrive the documents from given collection via keyword + vector search",
             description="It retrive most relavent docs")
async def hybrid_retrival(collection_name,query)-> Dict:
    try:
        # collection_name = f"{payload.coll}_{payload.version}" if payload.version is not None else payload.coll
        # if not milvus_client.has_collection(collection_name):
        #     raise HTTPException(
        #         status_code=404,
        #         detail=f"Knowledge base version not found: '{collection_name}'"
        #     )

        final_texts,sources = await hybrid_retrieve(collection_name,query)

        
        return {"status_code":200,"response":final_texts,"sources":sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(f"{API_PREFIX}/graph_search",summary="To retrive the answer from Neo4j graph DB via LLM question answering",
             description="It retrive most relavent docs")
async def hybrid_retrieval(query)-> Dict:
    try:
        # collection_name = f"{payload.coll}_{payload.version}" if payload.version is not None else payload.coll
        # if not milvus_client.has_collection(collection_name):
        #     raise HTTPException(
        #         status_code=404,
        #         detail=f"Knowledge base version not found: '{collection_name}'"
        #     )

        output = await answer_question(query)


        return {"status_code":200,"response":output.get("answer",""),"sources":output.get("results","")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    