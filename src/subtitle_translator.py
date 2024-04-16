
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent
from src.models.base_model import BaseModel
from src.logger import logger
from src.constants import MAX_RETRIES
from src.subtitle_processor import SubtitleProcessor

class SubtitleTranslator:
    def __init__(self, model: BaseModel, lang: str, num_threads: int = 1, tokens_per_chunk: int = 500):
        self.model = model
        self.lang = lang
        self.num_threads = num_threads
        self.tokens_per_chunk = tokens_per_chunk
        self.processor = SubtitleProcessor(model)
        with open("./prompt.txt", encoding="utf-8") as f:
            self.prompt_template = f.read()

    def translate_subtitles(self, srt_data: str):
        parsed_srt = self.processor.parse_srt(srt_data)
        preprocessed_text = self.processor.preprocess(parsed_srt)
        chunks = self.processor.make_chunks(preprocessed_text, self.tokens_per_chunk)
        logger.info(f"Split into {len(chunks)} chunks.")
        translations = [""] * len(chunks)
        futures = []
        stop_flag = threading.Event()

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for i, chunk in enumerate(chunks):
                future = executor.submit(self.translate_chunk, i, chunk, stop_flag, 0)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    index, response, err, attempts = future.result()
                    translations[index] = response
                    if err:
                        raise Exception(err)
                except Exception as e:
                    logger.error(e)
                    logger.info("Stopping translation early.")
                    stop_flag.set()
                    for fut in futures:
                        fut.cancel()
                    break

        joined_text = "\n\n".join(translations)
        return self.processor.post_process_text(joined_text, parsed_srt)

    def translate_chunk(self, chunk_idx, chunk, stop_flag, attempt):
        if stop_flag.is_set():
            return chunk_idx, "", ""

        chunk_number = chunk_idx + 1
        subtitles, mapping = self.processor.randomize_ids(chunk)
        response = self.get_translation(chunk_number, subtitles)
        response = response.strip()
        response = self.processor.revert_id_randomization(response, mapping)
        error_message = ""

        try:
            self.validate_response(response, chunk, chunk_number)
        except Exception as e:
            if attempt < MAX_RETRIES and isinstance(e, MissingSubtitlesError):
                logger.info(f"Retrying chunk {chunk_number}, after error: {e}")
                # todo split chunk in half
                return self.translate_chunk(chunk_idx, chunk, stop_flag, attempt + 1)
            else:
                error_message = str(e)

        return chunk_idx, response, error_message, attempt + 1

    def get_translation(self, chunk_number, chunk):
        prompt = self.prompt_template.replace("{subtitles}", chunk.strip()) \
            .replace("{target_language}", self.lang)
        logger.info(f"Processing chunk {chunk_number}, with {self.model.num_tokens_from_string(chunk)} tokens.")
        return self.model.generate_completion(prompt)

    def validate_response(self, response, chunk, chunk_number):
        token_count = self.model.num_tokens_from_string(response)
        if token_count >= self.model.max_output_tokens():
            raise ResponseTooLongError(
                f"Response too long. Might be missing tokens. {token_count} tokens, max is {self.model.max_output_tokens()} tokens. Try a smaller chunk size."
            )
        missing_subtitles = self.processor.get_missing_subtitles(response, chunk)
        if missing_subtitles:
            raise MissingSubtitlesError(
                f"Chunk {chunk_number} is missing {len(missing_subtitles)} subtitles. Try a smaller chunk size."
            )

        logger.info(f"Got chunk {chunk_number}, length is {token_count} tokens.")

class ResponseTooLongError(Exception):
    """Exception raised when the response exceeds the maximum token limit."""

class MissingSubtitlesError(Exception):
    """Exception raised when subtitles are missing in the response."""
