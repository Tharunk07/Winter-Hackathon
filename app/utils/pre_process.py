import logging
import re
from datetime import datetime

def generate_safe_collection_name(base: str) -> str:
    '''
        Used to generate an unique id for knowledge base
    '''
    safe_base = re.sub(r'[^a-zA-Z0-9]', '', base)
    timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    return f"hackathon_{safe_base}{timestamp}"

def process_and_store_content(caption, extracted_text, image_path):

    try:
        combined_text = f"Caption: {caption}\nExtracted Text: {extracted_text}"

        chunk_data = []

        chunk_data.append({
            "text": combined_text,
            "sourceURL": image_path,
            "start_time": 0.0,
            "end_time": 0.0
        })
        return chunk_data
    
    except Exception as e:
        logging.error(f"Error in processing and storing content: {e}")
        return None