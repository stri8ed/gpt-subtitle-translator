import os
import re
import threading
import traceback
import zlib
from concurrent.futures import ThreadPoolExecutor
import concurrent
from typing import Callable, Optional, Literal

from gpt_subtitle_translator.constants import COMPRESSION_RATIO_THRESHOLD
from gpt_subtitle_translator.models.base_model import BaseModel
from gpt_subtitle_translator.logger import logger
from gpt_subtitle_translator.subtitle_processor import SubtitleProcessor, Chunk


class SubtitleTranslator:
    def __init__(
        self,
        model: BaseModel,
        lang: str,
        num_threads: int = 1,
        tokens_per_chunk: int = 500,
        max_retries: int = 1,
        retry_on_refusal: bool = False,
        temperature: float = 0.5,
        source_type: Literal["ocr", "transcription"] = "ocr"
    ):
        self.model = model
        self.lang = lang
        self.temperature = temperature
        self.num_threads = num_threads
        self.tokens_per_chunk = tokens_per_chunk
        self.max_retries = max_retries
        self.retry_on_refusal = retry_on_refusal
        self.processor = SubtitleProcessor(model)
        self.prompt_template = self.load_prompt(source_type)

    @staticmethod
    def load_prompt(source_type: str) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_file = os.path.join(script_dir, '.', f"prompt/prompt_{source_type}.txt")
        with open(prompt_file, encoding="utf-8") as f:
            prompt = f.read()
        return prompt

    def translate_subtitles(
        self,
        srt_data: str,
        progress_callback: Optional[Callable[[float], None]] = None,
        on_complete_chunk: Optional[Callable[[int, str], None]] = None,
        completed_chunks: Optional[dict[int, str]] = None,
        timeout: Optional[float] = None
    ) -> str:
        parsed_srt = self.processor.parse_srt(srt_data)
        preprocessed_text = self.processor.preprocess(parsed_srt)
        chunks = self.processor.make_chunks(preprocessed_text, self.tokens_per_chunk)
        logger.info(f"Split into {len(chunks)} chunks.")

        completed_chunks = completed_chunks or {}
        translations = [""] * len(chunks)
        futures = []
        err = None
        stop_flag = threading.Event()

        for idx, translation in completed_chunks.items():
            translations[idx] = translation
            logger.info(f"Skipping chunk {idx + 1} (already completed)")

        executor = ThreadPoolExecutor(max_workers=self.num_threads)
        try:
            for i, chunk in enumerate(chunks):
                if i not in completed_chunks:
                    future = executor.submit(self.translate_chunk, chunk, stop_flag, 0)
                    futures.append(future)

            try:
                for future in concurrent.futures.as_completed(futures, timeout=timeout):
                    index, response, raw_response = future.result()
                    translations[index] = response
                    if progress_callback:
                        progress_callback(len([t for t in translations if t]) / len(chunks))
                    if on_complete_chunk:
                        on_complete_chunk(index, raw_response)
            except Exception as e:
                if not err:
                    err = e
                    stack_trace = traceback.format_exc()
                    stop_flag.set()
                    for fut in futures:
                        fut.cancel()
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        joined_text = "\n\n".join(translations)
        result_text = self.processor.post_process_text(joined_text, parsed_srt)

        if err:
            completed_chunks = {i: t for i, t in enumerate(translations) if t}
            total_chunks = len(chunks)
            raise TranslationError(err, stack_trace, result_text, total_chunks, completed_chunks)

        return result_text

    def translate_chunk(
        self,
        chunk: Chunk,
        stop_flag,
        attempt: int,
        temperature=None,
        shuffle_on_retry=True,
        randomize_ids=False
    ):
        if stop_flag.is_set():
            return chunk.idx, "", ""

        chunk_number = chunk.idx + 1
        if randomize_ids:
            subtitles, mapping = self.processor.randomize_ids(chunk.text)
        else:
            subtitles, mapping = self.processor.reindex_ids(chunk.text)

        if attempt >= 2 and shuffle_on_retry:
            logger.info(f"Shuffling order of chunk {chunk_number} after error.")
            subtitles = self.processor.shuffle_order(subtitles)

        try:
            raw_response, num_tokens = self.get_translation(
                chunk_number, subtitles, chunk.num_tokens, temperature
            )
            response = raw_response.strip()
            response = self.processor.extract_subtitles(response, mapping)
            self.validate_response(
                response, chunk.text, chunk_number, raw_response, num_tokens
            )
        except Exception as e:
            if (
                attempt < self.max_retries and
                not isinstance(e, QuotaExhaustedError) and
                (
                    isinstance(e, (MissingSubtitlesError, ResponseRepetitiveError, UntranslatedResponseError))
                    or "retry" in str(e)
                    or "Please try again" in str(e)
                    or (isinstance(e, RefuseToTranslateError) and self.retry_on_refusal)
                )
            ):
                logger.info(
                    f"Retrying chunk {chunk_number}, after error: {e} [attempt {attempt + 1}]"
                )
                shuffle_on_retry = isinstance(e, (MissingSubtitlesError, RefuseToTranslateError))
                temperature = 0.5 if isinstance(e, ResponseRepetitiveError) else None
                return self.translate_chunk(chunk, stop_flag, attempt + 1, temperature, shuffle_on_retry, randomize_ids)
            else:
                raise e

        return chunk.idx, response, raw_response

    def get_translation(self, chunk_number, text: str, num_tokens: int, temperature=None) -> (str, int):
        prompt = self.prompt_template.replace("{subtitles}", text.strip()) \
            .replace("{target_language}", self.lang)
        logger.info(f"Processing chunk {chunk_number}, with {num_tokens} tokens.")
        return self.model.generate_completion(prompt, temperature or self.temperature, target_language=self.lang)

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r'\W', '', text).strip()

    @staticmethod
    def get_compression_ratio(text: str) -> float:
        text_bytes = text.encode("utf-8")
        return len(text_bytes) / len(zlib.compress(text_bytes))

    def validate_response(self, response: str, original_text: str, chunk_number: int, raw_response: str, num_tokens: int):
        if num_tokens >= self.model.max_output_tokens():
            if self.get_compression_ratio(raw_response) >= COMPRESSION_RATIO_THRESHOLD:
                raise ResponseRepetitiveError(
                    f"Chunk {chunk_number} response is stuck in a repeating pattern. {num_tokens} tokens, "
                    f"Preview: {raw_response[:1000]}"
                )
            else:
                raise ResponseTooLongError(
                    f"Chunk {chunk_number} response too long. Might be missing tokens. {num_tokens} tokens, "
                    f"max is {self.model.max_output_tokens()} tokens. Try a smaller chunk size. "
                    f"Preview: {raw_response[:1000]}"
                )

        original_count = len(self.processor.split_on_tags(original_text))
        missing_subtitles = self.processor.get_missing_subtitles(response, original_text)

        if missing_subtitles:
            if len(missing_subtitles) == original_count and len(raw_response) > 0:
                raise MissingSubtitlesError(f"Chunk {chunk_number} is missing all subtitles.")
            else:
                raise MissingSubtitlesError(
                    f"Chunk {chunk_number} is missing {len(missing_subtitles)} subtitles. Try a smaller chunk size."
                )

        translated_count = len(re.findall(r'^<(\d+)>', response.strip(), flags=re.MULTILINE))
        if translated_count > original_count:
            raise MissingSubtitlesError(
                f"Chunk {chunk_number} generated {translated_count - original_count} extra subtitles: "
                f"expected {original_count}, got {translated_count}."
            )

        if self.normalize_text(response) == self.normalize_text(original_text):
            raise UntranslatedResponseError(
                f"Chunk {chunk_number} returned the original text verbatim without translating."
            )

        logger.info(f"Got chunk {chunk_number}, length is {num_tokens} tokens.")


class ResponseTooLongError(Exception):
    """Exception raised when the response exceeds the maximum token limit."""

class ResponseRepetitiveError(Exception):
    """Exception raised when the response is stuck in a repeating pattern."""

class MissingSubtitlesError(Exception):
    """Exception raised when subtitles are missing in the response."""

class UntranslatedResponseError(Exception):
    """Exception raised when the response is identical to the original text (not translated)."""

class RefuseToTranslateError(Exception):
    """Exception raised when the model refuses to translate the text."""

class QuotaExhaustedError(Exception):
    """Exception raised when API quota is exhausted. Should fail fast to try next model."""


class TranslationError(Exception):
    """
    Exception raised when an error occurs during the translation process.
    Stores the error type, partial translation.
    """

    def __init__(
        self,
        original_exception,
        stack_trace,
        partial_translation=None,
        total_chunks=0,
        completed_chunks: Optional[dict[int, str]] = None
    ):
        super().__init__(str(original_exception))
        self.original_exception = original_exception
        self.partial_translation = partial_translation
        self.stack_trace = stack_trace
        self.completed_chunks = completed_chunks or {}
        self.total_chunks = total_chunks

    def __str__(self):
        return "".join([
            f"An error occurred ({type(self.original_exception).__name__}): {str(self.original_exception)}\n"
            f"Completed chunks: {len(self.completed_chunks)}\n",
            f"Total chunks: {self.total_chunks}\n",
            f"Partial translation\n: {self.partial_translation[:350]}" if self.partial_translation else ""
        ])

    def get_stack_trace(self):
        return self.stack_trace
