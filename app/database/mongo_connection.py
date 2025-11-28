from app import Config
from .mongo_client import get_db

MONGO_CONNECTION = Config.MONGO_DB_URI
KAPAI_DB = Config.MONGO_DB

MONGO_DB_PDF_COLLECTION = "Multimodel_pdf_collection"
MONGO_DB_COLLECTION_LIST = "Multimodel_collection_list"
MONGO_DB_VIDEO_COLLECTION = "Multimodel_video_collection"
def get_collection(collection_name):

    return get_db()[collection_name]


async def insert_one_data(collection_name, data):

    return await get_collection(collection_name).insert_one(data)

async def insert_many_data(collection_name, data):

    return await get_collection(collection_name).insert_many(data)

async def find_data(collection_name, find_query):

    cursor = get_collection(collection_name).find(find_query)
    return await cursor.to_list(length=None)

async def update_data(collection_name, find_query, update_query):

    return await get_collection(collection_name).update_many(find_query, update_query)

async def find_one_data(collection_name, find_query, sort_by="created_at", skip_by=0):

    cursor = (
        get_collection(collection_name)
        .find(find_query)
        .sort(sort_by, -1)
        .skip(skip_by)
        .limit(1)
    )
    docs = await cursor.to_list(length=1)
    return docs[0] if docs else None


async def update_one_data(collection_name, find_query, update_query, sort=[("created_at", -1)]):

    return await get_collection(collection_name).find_one_and_update(
        filter=find_query,
        update=update_query,
        sort=sort,
    )