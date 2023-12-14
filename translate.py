import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import concurrent
from logger import logger

import os
import openai
import tiktoken
import re

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
assert openai.api_key is not None, "OpenAI API key not found. Please set it in the .env file."

MODEL = "gpt-4-1106-preview"
TOKENS_PER_CHUNK = 2_000 # Safe value, might be able to increase depending on the language
MAX_OUTPUT_TOKENS = 4095 # Fixed by OpenAI
MAX_RETRIES = 1 # Despite instructions, model sometimes skips/merges subtitles. Retrying helps.

with open("./prompt.txt", encoding="utf-8") as f:
    prompt_template = f.read().replace("{target_language}", os.getenv("TARGET_LANGUAGE"))

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def translate_chunk(chunk_idx, chunk, stop_flag, attempt=0):
    if stop_flag.is_set():
        return chunk_idx, "", ""

    chunk_number = chunk_idx + 1
    response = get_translation(chunk_number, chunk)
    error_message = ""

    try:
        check_response(response, chunk, chunk_number)
    except Exception as e:
        if attempt < MAX_RETRIES and isinstance(e, MissingSubtitlesError):
            logger.info(f"Retrying chunk {chunk_number}, after error: {e}")
            return translate_chunk(chunk_idx, chunk, stop_flag, attempt + 1)
        else:
            error_message = str(e)

    return chunk_idx, response, error_message

def get_translation(chunk_number:int, chunk:str) -> str:
    prompt = prompt_template.replace("{subtitles}", chunk.strip())
    messages = [{"role": "system", "content": prompt}]
    logger.info(f"Processing chunk {chunk_number}, with {num_tokens_from_string(chunk)} tokens.")
    return openai.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=0.5,
        n=1,
        stop=None,
    ).choices[0].message.content

def check_response(response: str, chunk: str, chunk_number):
    token_count = num_tokens_from_string(response)
    if token_count >= MAX_OUTPUT_TOKENS:
        raise ResponseTooLongError(
            f"Response too long. Might be missing tokens. {token_count} tokens, max is {MAX_OUTPUT_TOKENS} tokens. Try a smaller chunk size."
        )
    missing_subtitles = get_missing_subtitles(response, chunk)
    if missing_subtitles:
        raise MissingSubtitlesError(
            f"Chunk {chunk_number} is missing {len(missing_subtitles)} subtitles. Try a smaller chunk size."
        )

def translate_subtitles(srt_data: str, num_threads: int = 1):
    text = preprocess(srt_data)
    chunks = make_chunks(text, max_tokens_per_chunk=TOKENS_PER_CHUNK)
    logger.info(f"Split into {len(chunks)} chunks.")
    assert "<1>" in chunks[0], "First chunk does not contain subtitle id. Please check your input."

    translations = [""] * len(chunks)
    futures = []
    stop_flag = threading.Event()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i, chunk in enumerate(chunks):
            future = executor.submit(translate_chunk, i, chunk, stop_flag)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                index, response, err = future.result()
                translations[index] = response
                if err != "":
                    raise Exception(err)
            except Exception as e:
                logger.error(e)
                logger.info("Stopping translation early.")
                stop_flag.set()
                for fut in futures:
                    fut.cancel()
                break

    joined_text = "\n\n".join(translations)
    return post_process_text(joined_text, srt_data)

def make_chunks(text, max_tokens_per_chunk) -> list[str]:
    lines = text.split("\n")
    pieces = []
    current_piece = ""
    for ln in lines:
        candidate_length = num_tokens_from_string(current_piece) + num_tokens_from_string(ln) + 1
        if candidate_length <= max_tokens_per_chunk:
            current_piece += ("\n" + ln)
        else:
            pieces.append(current_piece)
            current_piece = ln

    pieces.append(current_piece)

    return pieces

def parse_srt(srt_content):
    lines = srt_content.strip().split('\n')
    subtitles = {}
    i = 0
    while i < len(lines):
        subtitle_id = int(lines[i])
        timestamp = lines[i + 1]
        text_lines = []
        i += 2
        while i < len(lines) and lines[i].strip() != "":
            text_lines.append(lines[i])
            i += 1
        subtitles[subtitle_id] = {"timestamp": timestamp, "text": " ".join(text_lines)}
        i += 1

    return subtitles


def insert_timestamps(parsed_subtitles, content):
    def replacement(match):
        subtitle_id = int(match.group(1))
        return f'\n{subtitle_id}\n{parsed_subtitles[subtitle_id]["timestamp"]}\n{match.group(2)}'

    return re.sub(r'^<(\d+)>([^\n]+?)</\d+>$', replacement, content, flags=re.MULTILINE)

def post_process_text(text: str, original_text: str) -> str:
    if original_text == "":
        return text
    original_subtitles = parse_srt(original_text)
    output_string = insert_timestamps(original_subtitles, text)
    output_string = re.sub(r'\n{3,}', "\n\n", output_string)
    output_string = output_string.strip()
    return clean_text(output_string)

def clean_text(text: str) -> str:
    output_string = re.sub(r'[<>]\s*$', "", text, flags=re.MULTILINE) # breaks subtitle parsing on Plex
    output_string = re.sub(r'^START\n\n', "", output_string, flags=re.MULTILINE)
    output_string = re.sub(r'\n\nEND', "", output_string, flags=re.MULTILINE)
    return output_string

def get_missing_subtitles(translated: str, original_text: str):
    translated_ids = re.findall(r'^<(\d+)>', translated.strip(), flags=re.MULTILINE)
    lines = re.findall(r"<(\d+)>(.*?)</\1>", original_text.strip(), flags=re.MULTILINE)
    entries = {id_: text for id_, text in lines}

    return {key: value for key, value in entries.items() if key not in translated_ids}

def preprocess(text: str) -> str:
    subtitle_pattern = r"(\d+)\s+\d+:\d{2}:\d{2},\d{3} --> \d+:\d{2}:\d{2},\d{3}\s*([^\n]+)\s+"
    output_string = re.sub(r'\r', "", text)
    output_string = re.sub(subtitle_pattern, r"<\1>\2</\1>\n", output_string)
    return output_string

class ResponseTooLongError(Exception):
    """Exception raised when the response exceeds the maximum token limit."""

class MissingSubtitlesError(Exception):
    """Exception raised when subtitles are missing in the response."""
