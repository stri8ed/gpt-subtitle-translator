import random
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent
from logger import logger
from gpt import GPT
import re
from constants import MAX_RETRIES, MODEL

model = GPT(model_name=MODEL)

with open("./prompt.txt", encoding="utf-8") as f:
    prompt_template = f.read()

def translate_chunk(chunk_idx, chunk, stop_flag, lang, attempt=0):
    if stop_flag.is_set():
        return chunk_idx, "", ""

    chunk_number = chunk_idx + 1
    subtitles, mapping = randomize_ids(chunk)
    response = get_translation(chunk_number, subtitles, lang)
    response = revert_id_randomization(response, mapping)
    error_message = ""

    try:
        validate_response(response, chunk, chunk_number)
    except Exception as e:
        if attempt < MAX_RETRIES and isinstance(e, MissingSubtitlesError):
            logger.info(f"Retrying chunk {chunk_number}, after error: {e}")
            return translate_chunk(chunk_idx, chunk, stop_flag, lang, attempt + 1)
        else:
            error_message = str(e)

    return chunk_idx, response, error_message, attempt + 1

def get_translation(chunk_number:int, chunk:str, lang:str) -> str:
    prompt = prompt_template.replace("{subtitles}", chunk.strip()) \
        .replace("{target_language}", lang)
    logger.info(f"Processing chunk {chunk_number}, with {model.num_tokens_from_string(chunk)} tokens.")
    return model.generate_completion(prompt)

def validate_response(response: str, chunk: str, chunk_number):
    token_count = model.num_tokens_from_string(response)
    if token_count >= model.max_output_tokens():
        raise ResponseTooLongError(
            f"Response too long. Might be missing tokens. {token_count} tokens, max is {model.max_output_tokens()} tokens. Try a smaller chunk size."
        )
    missing_subtitles = get_missing_subtitles(response, chunk)
    if missing_subtitles:
        raise MissingSubtitlesError(
            f"Chunk {chunk_number} is missing {len(missing_subtitles)} subtitles. Try a smaller chunk size."
        )

    logger.info(f"got chunk {chunk_number}, length is {token_count} tokens.")

def translate_subtitles(
    srt_data: str,
    lang: str,
    tokens_per_chunk: int,
    num_threads: int = 1
):
    parsed_srt = parse_srt(srt_data)
    text = preprocess(parsed_srt)
    chunks = make_chunks(text, max_tokens_per_chunk=tokens_per_chunk)
    logger.info(f"Split into {len(chunks)} chunks.")

    translations = [""] * len(chunks)
    futures = []
    stop_flag = threading.Event()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        for i, chunk in enumerate(chunks):
            future = executor.submit(translate_chunk, i, chunk, stop_flag, lang, 0)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            try:
                index, response, err, attempts = future.result()
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

    print(f"Total API cost: ${model.get_total_cost():.5f}")
    joined_text = "\n\n".join(translations)
    return post_process_text(joined_text, parsed_srt)

def make_chunks(text, max_tokens_per_chunk) -> list[str]:
    lines = text.split("\n")
    pieces = []
    current_piece = ""
    for ln in lines:
        candidate_length = model.num_tokens_from_string(current_piece) + model.num_tokens_from_string(ln) + 1
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
    try:
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
    except Exception as e:
        raise Exception("Invalid SRT file. Please check your input. " + str(e))

    return subtitles

def randomize_ids(subtitles: str) -> (str, dict):
    """
    Randomize subtitle IDs to avoid GPT skipping/merging subtitles.

    Valid subtitles have consecutive numeric IDs, which seems to make GPT more likely to skip/merge neighboring subtitles.
    By assigning new IDs randomly, while preserving the order, we can help GPT avoid this behavior.
    """
    items = subtitles.strip().split("\n")
    pattern = re.compile(r"<(\d+)>(.*?)</\d+>")

    tag_count = len(items)
    new_ids = random.sample(range(1, tag_count * 10), tag_count)

    mapping = {}
    updated = []

    for index, string in enumerate(items):
        original_id = pattern.search(string).group(1)
        new_id = new_ids[index]
        mapping[new_id] = original_id
        updated_string = re.sub(pattern, f"<{new_id}>\\2</{new_id}>", string)
        updated.append(updated_string)

    return "\n".join(updated), mapping

def revert_id_randomization(subtitles: str, mapping: dict) -> str:
    pattern = re.compile(r"<(\d+)>(.*?)</\d+>")

    def replace_with_original(match):
        original_id = mapping[int(match.group(1))]
        return f"<{original_id}>{match.group(2)}</{original_id}>"

    reverted_chunk = re.sub(pattern, replace_with_original, subtitles)
    return reverted_chunk

def insert_timestamps(parsed_subtitles, content):
    def replacement(match):
        subtitle_id = int(match.group(1))
        return f'\n{subtitle_id}\n{parsed_subtitles[subtitle_id]["timestamp"]}\n{match.group(2)}'

    return re.sub(r'^<(\d+)>([^\n]*?)</\d+>$', replacement, content, flags=re.MULTILINE)

def post_process_text(text: str, original_subtitles: dict) -> str:
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

def preprocess(srt_data: dict) -> str:
    result = ""
    for key, value in srt_data.items():
        result += f"<{key}>{value['text']}</{key}>\n"
    return result

class ResponseTooLongError(Exception):
    """Exception raised when the response exceeds the maximum token limit."""

class MissingSubtitlesError(Exception):
    """Exception raised when subtitles are missing in the response."""
