from openai import OpenAI
import base64
from config import OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)


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
                        "text": "Provide a concise caption for this image to store in the RAG"
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



async def extract_text_from_images(paths_to_images: list[str]) -> str:
    texts = []
    for path in paths_to_images:
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

