from openai import OpenAI
from typing import Dict, Any
import logging
import subprocess, os
import json
from config import OPENAI_API_KEY
import requests
import os


logger = logging.getLogger(__name__)

client = OpenAI(api_key=OPENAI_API_KEY)



def download_file(url: str, output_dir: str = "./downloads") -> str:
    """
    Downloads a file from the given URL and saves it to the specified directory.

    Args:
        url (str): The URL of the file to download.
        output_dir (str): The directory to save the downloaded file.

    Returns:
        str: The path to the downloaded file.
    """
    try:
        # Ensure the output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Extract the file name from the URL
        file_name = os.path.basename(url)
        output_path = os.path.join(output_dir, file_name)

        # Download the file
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(output_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        return output_path

    except Exception as e:
        raise RuntimeError(f"Failed to download file from {url}: {e}")
    
async def transcribe_audio(audio_file):

    audio_file = open(audio_file, "rb")

    transcript = client.audio.transcriptions.create(
        model="gpt-4o-transcribe",
        file=audio_file,
        prompt=(
            "Transcribe the audio in English. "
            "Return the result strictly as JSON. "
            "Split the transcript into segments of 30 seconds. "
            "For each segment, include:\n"
            "  - start_time (in seconds)\n"
            "  - end_time (in seconds)\n"
            "  - text\n"
            "Example JSON format:\n"
            "{\n"
            '  "segments": [\n'
            "    {\n"
            '      "start_time": 0,\n'
            '      "end_time": 5,\n'
            '      "text": "Hello world..."\n'
            "    }\n"
            "  ]\n"
            "}"
        )
    )

    print(transcript.text)

    transcript_json = json.loads(transcript.text)

    return transcript_json



async def process_and_store_transcript(transcript: Dict[str, Any], audio_path):
    """
    Process the transcript and store it in Milvus.
    """
    try:
        segments = transcript.get("segments", [])
        chunked_data = [
            {
                "text": segment["text"],
                "start_time": segment["start_time"],
                "end_time": segment["end_time"],
                "sourceURL": audio_path
            }
            for segment in segments
        ]

        return chunked_data
    
    except Exception as e:
        logger.error(f"Error processing and storing transcript: {e}", exc_info=True)
        return []



async def convert_video_to_audio(input_file: str) -> str:
    """
    Convert a video file to audio (WAV format) using ffmpeg.
    Returns the path to the output audio file.
    """
    output_file = os.path.splitext(input_file)[0] + ".wav"
    subprocess.run([
        "ffmpeg",
        "-i", input_file,
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-f", "wav",
        output_file
    ], check=True)

    print("Created WAV:", output_file)
    return output_file