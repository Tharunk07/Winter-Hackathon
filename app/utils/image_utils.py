from openai import OpenAI
import base64
from config import OPENAI_API_KEY
import requests
import os
from urllib.parse import urlparse

client = OpenAI(api_key=OPENAI_API_KEY)



def download_image_from_s3(s3_url: str, output_dir: str = "./downloads") -> str:
    """
    Downloads an image from an S3 URL and saves it to the specified directory.

    Args:
        s3_url (str): The S3 URL of the image to download.
        output_dir (str): The directory to save the downloaded image.

    Returns:
        str: The path to the saved image.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Extract the file name from the S3 URL
        parsed_url = urlparse(s3_url)
        file_name = os.path.basename(parsed_url.path)
        output_path = os.path.join(output_dir, file_name)

        # Download the image
        response = requests.get(s3_url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"Image downloaded successfully: {output_path}")
        return output_path

    except Exception as e:
        raise RuntimeError(f"Failed to download image from {s3_url}: {e}")
    
async def caption_image(path_to_image: str) -> str:
    with open(path_to_image, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Provide a concise caption for this image to store in the RAG properly describe everything in the image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    }
                ]
            }
        ],
        max_tokens=100
    )
    return resp.choices[0].message.content



async def extract_text_from_images(path: str) -> str:
    texts = []
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    data_uri = f"data:image/png;base64,{b64}"

    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all text from this image for RAG ingestion. Return only the text without any additional commentary."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri}
                    }
                ]
            }
        ],
        max_tokens=1000
    )
    texts.append(resp.choices[0].message.content)
    return "\n".join(texts)

