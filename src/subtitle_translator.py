
import threading
from concurrent.futures import ThreadPoolExecutor
import concurrent
from typing import Callable, Optional

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

    def translate_subtitles(self, srt_data: str, progress_callback: Optional[Callable[[float], None]] = None) -> str:
        parsed_srt = self.processor.parse_srt(srt_data)
        preprocessed_text = self.processor.preprocess(parsed_srt)
        chunks = self.processor.make_chunks(preprocessed_text, self.tokens_per_chunk)
        logger.info(f"Split into {len(chunks)} chunks.")
        translations = [""] * len(chunks)
        futures = []
        err = None
        stop_flag = threading.Event()

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            for i, chunk in enumerate(chunks):
                future = executor.submit(self.translate_chunk, i, chunk, stop_flag, 0)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                try:
                    index, response, _ = future.result()
                    translations[index] = response
                    if progress_callback:
                        progress_callback(len([t for t in translations if t]) / len(chunks))
                except Exception as e:
                    err = e
                    stop_flag.set()
                    for fut in futures:
                        fut.cancel()
                    break

        joined_text = "\n\n".join(translations)
        result_text = self.processor.post_process_text(joined_text, parsed_srt)

        if err:
            raise TranslationError(err, result_text)

        return result_text

    def translate_chunk(self, chunk_idx, chunk, stop_flag, attempt):
        if stop_flag.is_set():
            return chunk_idx, "", ""

        chunk_number = chunk_idx + 1
        subtitles, mapping = self.processor.randomize_ids(chunk)
        response = self.get_translation(chunk_number, subtitles)
        response = response.strip()
        response = self.processor.revert_id_randomization(response, mapping)

        try:
            self.validate_response(response, chunk, chunk_number)
        except Exception as e:
            if attempt < MAX_RETRIES and isinstance(e, MissingSubtitlesError):
                logger.info(f"Retrying chunk {chunk_number}, after error: {e}")
                return self.translate_chunk(chunk_idx, chunk, stop_flag, attempt + 1)
            else:
                raise e

        return chunk_idx, response, attempt + 1

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
            if len(missing_subtitles) == len(self.processor.split_on_tags(chunk)):
                raise RefuseToTranslateError(f"Response: {response}")
            else:
                raise MissingSubtitlesError(
                    f"Chunk {chunk_number} is missing {len(missing_subtitles)} subtitles. Try a smaller chunk size."
                )

        logger.info(f"Got chunk {chunk_number}, length is {token_count} tokens.")

class ResponseTooLongError(Exception):
    """Exception raised when the response exceeds the maximum token limit."""

class MissingSubtitlesError(Exception):
    """Exception raised when subtitles are missing in the response."""

class RefuseToTranslateError(Exception):
    """Exception raised when the model refuses to translate the text."""

class TranslationError(Exception):
    """
    Exception raised when an error occurs during the translation process.
    Stores the error type, partial translation.
    """
    def __init__(self, original_exception, partial_translation = None):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.partial_translation = partial_translation

    def __str__(self):
        return "".join([
            f"An error occurred ({type(self.original_exception).__name__}): {str(self.original_exception)}\n",
            f"Partial translation: {self.partial_translation}" if self.partial_translation else ""
        ])
