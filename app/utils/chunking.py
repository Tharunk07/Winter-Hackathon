from langchain_text_splitters import RecursiveCharacterTextSplitter
import fitz
import io
import re
import requests
from typing import List
from config import embedding_model
from typing import Dict,List,Any
import pandas as pd
import logging
logger = logging.getLogger(__name__)


async def clean_unicode_junk(docs: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Remove unwanted Unicode characters like bullets (• = \u2022) 
    and ellipsis (…) = \u2026 from extracted PDF text.

    Args:
        docs: List of dicts with {"text": ..., "sourceURL": ...}

    Returns:
        Cleaned list of dicts in the same format.
    """
    cleaned_docs = []
    
    # Characters/patterns to strip
    junk_pattern = re.compile(r"[\u2022\u2026]+")

    for doc in docs:
        text = doc.get("text", "")
        
        # Remove junk characters
        cleaned_text = junk_pattern.sub("", text)
        
        # Also collapse extra spaces caused by removal
        cleaned_text = re.sub(r"\s{2,}", " ", cleaned_text).strip()

        cleaned_docs.append({
            "text": cleaned_text,
            "sourceURL": doc.get("sourceURL", "")
        })

    return cleaned_docs


async def download_extract_text_from_pdf(links: List[str]) -> List[Dict[str, str]]:
    """
    Given a list of PDF links, extract clean text from each using PyMuPDF.
    Skips:
      - Non-PDF links
      - Failed downloads
      - Image-only pages (no real text)
      - Garbled or byte-like text
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/pdf",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "identity",
        "Referer": "https://google.com"
    }

    extracted = []

    for url in links:
        if not url.lower().endswith(".pdf"):
            logger.warning(f"Skipping non-PDF link: {url}")
            continue

        try:
            resp = requests.get(url, headers=headers, timeout=(10, 60))
            resp.raise_for_status()
        except Exception as e:
            logger.warning(f"{url} download failed: {e}")
            continue

        try:
            pdf_data = io.BytesIO(resp.content)
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            pages = []

            for page_num, page in enumerate(doc, start=1):

                text = page.get_text("text") or ""
                if not text.strip():
                    image_list = page.get_images(full=True)
                    if image_list:
                        logger.info(f"Page {page_num} in {url} contains only images — skipped.")
                        continue

                cleaned = re.sub(r"\(cid:\d+\)", "", text).strip()

                pages.append(cleaned)

            doc.close()

            if pages:
                combined = "\n\n".join(pages).strip()
                extracted.append({"text": combined, "sourceURL": url})
            else:
                logger.warning(f"No readable text found in {url}")

        except Exception as e:
            logger.warning(f"Failed to parse PDF {url}: {e}")
            continue

    logger.info(f"PDFs extracted successfully: {len(extracted)}")

    extracted =  await clean_unicode_junk(docs=extracted)
    return extracted


async def recursive_chunking(pages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Split multiple text documents (each with its source URL) into overlapping chunks 
    using RecursiveCharacterTextSplitter.
    
    Args:
        pages (List[Dict[str, str]]): A list of dictionaries, where each dictionary contains:
            - "text" (str): The text to be chunked.
            - "sourceURL" (str): The associated source link.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, where each contains:
            - "text": The chunked portion of the original text.
            - "sourceURL": The corresponding source link.
    """
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=72,
        length_function=len,
    )
    
    chunked_documents = []
    
    for page in pages:
        text = page.get("text", "")
        source_url = page.get("sourceURL", "default")
        
        # Skip empty texts
        if not text.strip():
            continue
        
        chunks = text_splitter.split_text(text)
        
        chunked_documents.extend([
            {"text": chunk, "sourceURL": source_url}
            for chunk in chunks
        ])
    
    return chunked_documents



async def get_embeddings(chunked_text:List) -> List:
    chunked_text=[chunk['text'] for chunk in chunked_text]
    embeddings = embedding_model.encode(chunked_text).tolist()
    return embeddings

async def bm25_tonized(chunked_text:List)-> List:
    tokenized_corpus = [chunk['text'].split() for chunk in chunked_text]
    return tokenized_corpus



async def Recursive_chunking_with_url(extracted_data: List[Dict[str, Any]],is_pdf:bool=True) -> List[Dict[str, str]]:
    """
    For each document, chunks its main text and any linked PDFs.
    Returns [{"text": ..., "sourceURL": ...}, ...]
    """
    try:
        chunk_with_url: List[Dict[str, str]] = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=72,
            length_function=len
        )

        seen_pdfs = set()  # to avoid re-downloading duplicate PDFs
        all_pdf_links = []
        documents_links = []
        for page in extracted_data:
            # Chunk HTML/markdown content
            chunks = text_splitter.split_text(page["text"])
            chunk_with_url.extend([
                {"text": chunk, "sourceURL": page["sourceURL"]}
                for chunk in chunks
            ])

            # Collect new PDFs (avoiding duplicates)
            for link in page.get("file_links", []):
                if link.lower().endswith(".pdf") and link not in seen_pdfs:
                    seen_pdfs.add(link)
                    all_pdf_links.append(link)
                    documents_links.append(link)

        # Extract text from PDF links
        if is_pdf:
            pdf_entries = await download_extract_text_from_pdf(links=all_pdf_links)

            for pdf_doc in pdf_entries:
                pdf_chunks = text_splitter.split_text(pdf_doc["text"])
                chunk_with_url.extend([
                    {"text": chunk, "sourceURL": pdf_doc["sourceURL"]}
                    for chunk in pdf_chunks
                ])
        logger.info(f"Web data chunks len = {len(chunk_with_url)}")
        return chunk_with_url,documents_links
    except Exception as e:
        logger.error(f"The occured in Recursive_chunking_with_url = {e}")
        raise



async def chunking_for_pdf(pdf_links:List[str]) -> List[Dict[str, str]]:
    '''
        Chunking method design specifically for pdf alone
    '''
    try:
        logger.info("Pdf chunking started")
        chunk_with_url: List[Dict[str, str]] = []

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=72,
            length_function=len
        )


        pdf_links = list(dict.fromkeys(pdf_links))
        pdf_entries = await download_extract_text_from_pdf(links=pdf_links)

        for pdf_doc in pdf_entries:
            pdf_chunks = text_splitter.split_text(pdf_doc["text"])
            chunk_with_url.extend([
                {"text": chunk, "sourceURL": pdf_doc["sourceURL"]}  
                for chunk in pdf_chunks
            ])
        logger.info(f"Pdf chunker length = {len(chunk_with_url)}")
        return chunk_with_url
    except Exception as e:
        logger.error(f"error occured  in the  chunking_for_pdf = {e}")
        raise 


async def chunking_for_qa_json(qa_data: List[Dict[str, str]], sourceURL: str = "default.json") -> List[Dict[str, str]]:
    """
        
    """
    chunked: List[Dict[str, str]] = []
    try:
        for entry in qa_data:
            question = entry.get("Question", "").strip()
            answer = entry.get("Answer", "").strip()
            sourceURL = entry.get("Source_URL", sourceURL).strip()

            combined = f"Q:{question},A:{answer}"

            # If combined length is within 2000 chars → keep as is
            if len(combined) <= 2000:
                chunked.append({"text": combined, "sourceURL": sourceURL})
            else:
                # Split only the Answer into chunks
                max_len = 2000 - len(f"Q:{question},A:") - 10  # buffer
                start = 0
                while start < len(answer):
                    part = answer[start:start+max_len]
                    part_combined = f"Q:{question},A:{part}"
                    chunked.append({"text": part_combined, "sourceURL": sourceURL})
                    start += max_len

        logger.info(f"QA json length = {len(chunked)}")
        return chunked
    except Exception as e:
        logger.error(f"error occurred in the chunking_for_qa_json = {e}")


async def process_and_chunk_data_excel(df: pd.DataFrame, source_url: str) -> List[Dict[str, str]]:
        """
        Process DataFrame and create chunks based on column structure.
        
        Args:
            df (pd.DataFrame): DataFrame to process
            source_url (str): Original source URL for metadata
            
        Returns:
            List[Dict[str, str]]: List of chunked data
        """
        chunked: List[Dict[str, str]] = []
        columns = list(df.columns)
        
        # Check if it's a Q&A format (exactly 2 columns: Question and Answer)
        if len(columns) == 2 and set(columns) == {"Question", "Answer"}:
            logger.info("Detected Q&A format - using Q&A chunking method")
            
            # Convert DataFrame to list of dictionaries for Q&A processing
            qa_data = df.to_dict('records')
            
            for entry in qa_data:
                question = str(entry.get("Question", "")).strip()
                answer = str(entry.get("Answer", "")).strip()
                
                # Skip empty entries
                if not question and not answer:
                    continue
                
                combined = f"Q:{question},A:{answer}"
                
                # If combined length is within 2000 chars → keep as is
                if len(combined) <= 2000:
                    chunked.append({"text": combined, "sourceURL": source_url})
                else:
                    # Split only the Answer into chunks
                    max_len = 2000 - len(f"Q:{question},A:") - 10  # buffer
                    start = 0
                    while start < len(answer):
                        part = answer[start:start+max_len]
                        part_combined = f"Q:{question},A:{part}"
                        chunked.append({"text": part_combined, "sourceURL": source_url})
                        start += max_len
        
        else:
            logger.info(f"Detected general format with {len(columns)} columns - using row-based chunking")
            
            # Process each row as a separate chunk with all column data
            for index, row in df.iterrows():
                row_data = []
                
                # Create formatted string for each column-value pair
                for col in columns:
                    value = str(row[col]) if pd.notna(row[col]) else ""
                    if value.strip():  # Only include non-empty values
                        row_data.append(f"column:{col}, row:{value}")
                
                if row_data:  # Only add if there's actual data
                    chunk_text = "[" + "...".join(row_data) + "]"
                    
                    # If chunk is too long, we might need to split it further
                    if len(chunk_text) <= 2000:
                        chunked.append({"text": chunk_text, "sourceURL": source_url})
                    else:
                        # For very long rows, split into multiple chunks
                        # This is a simple approach - you might want to customize this
                        max_chunk_size = 1990  # Leave some buffer
                        for i in range(0, len(chunk_text), max_chunk_size):
                            partial_chunk = chunk_text[i:i+max_chunk_size]
                            if i > 0:
                                partial_chunk = "[..." + partial_chunk
                            if i + max_chunk_size < len(chunk_text):
                                partial_chunk = partial_chunk + "...]"
                            chunked.append({"text": partial_chunk, "sourceURL": source_url})
        
        logger.info(f"Generated {len(chunked)} chunks from the Excel data")
        return chunked