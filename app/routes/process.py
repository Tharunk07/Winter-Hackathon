from app import app
from config import API_PREFIX, MONGO_DB_URI, MONGO_DB, milvus_client, MILVUS_URI, MILVUS_TOKEN
from app.database import (
            MONGO_DB_PDF_COLLECTION,
            MONGO_DB_COLLECTION_LIST,
            MONGO_DB_VIDEO_COLLECTION,
            MONGO_DB_IMAGE_COLLECTION,
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
from datetime import datetime, UTC
from app.utils import generate_safe_collection_name, vector_search
from pymilvus import Collection , connections

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