import logging

_db = None

def set_db(db):
    global _db
    _db = db
    logging.info(f"Mongo DB Initialized Successfully!")

def get_db():
    if _db is None:
        raise RuntimeError("MongoDB has not been initialized.")
    return _db
