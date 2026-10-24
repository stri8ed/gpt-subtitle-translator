import os
import random
import time
from typing import Union

import httpx
from dotenv import load_dotenv

from gpt_subtitle_translator.logger import logger
from gpt_subtitle_translator.models.base_model import BaseModel
from gpt_subtitle_translator.models.schema import build_json_schema
from gpt_subtitle_translator.subtitle_translator import RefuseToTranslateError, ResponseTooLongError, QuotaExhaustedError

load_dotenv()

API_URL = "https://api.openai.com/v1"

# No price table: these run on the free data-sharing credits, so get_total_cost reports 0.
model_params = {
    "gpt-5.6-terra": {
        "max_output_tokens": 128_000,
        "reasoning_effort": "medium",
    },
}


def get_model_params(model_name: str):
    for key, value in model_params.items():
        if key in model_name:
            return value
    return None


class OpenAIError(Exception):
    def __init__(self, status_code: int, message: str, code: str = None):
        super().__init__(f"OpenAI error {status_code}: {message}")
        self.status_code = status_code
        self.code = code


class OpenAI(BaseModel):
    def __init__(self, model_name: str = "gpt-5.6-terra", api_key: Union[str, None] = None):
        super().__init__(model_name)
        _model_params = get_model_params(model_name)
        assert _model_params is not None, f"Model {model_name} info not found."
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key or os.environ['OPENAI_API_KEY']}"},
            timeout=httpx.Timeout(60 * 5, connect=10),
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.params = _model_params
        self.average_tokens_per_char = None
        self.retry_delays = [3, 6, 12]
        self.max_attempts = len(self.retry_delays) + 1

    def _post(self, path: str, body: dict) -> dict:
        response = self.client.post(f"{API_URL}/{path}", json=body)
        try:
            data = response.json()
        except ValueError:
            raise OpenAIError(response.status_code, response.text[:1000]) from None
        if response.status_code >= 400:
            error = data.get("error") or {}
            raise OpenAIError(response.status_code, error.get("message", response.text[:1000]), error.get("code"))
        return data

    def _post_with_retries(self, path: str, body: dict) -> dict:
        for attempt in range(self.max_attempts):
            retry_sleep_time = round(self.retry_delays[min(attempt, len(self.retry_delays) - 1)] * random.uniform(0.75, 1.25), 1)
            try:
                return self._post(path, body)
            except OpenAIError as e:
                if e.code == "insufficient_quota":
                    raise QuotaExhaustedError(str(e)) from e
                transient = e.status_code in (408, 429) or e.status_code >= 500
                if attempt < self.max_attempts - 1 and transient:
                    logger.warning(f"OpenAI server error: {e}. Retrying in {retry_sleep_time} seconds...")
                    time.sleep(retry_sleep_time)
                    continue
                if e.status_code == 429:
                    raise QuotaExhaustedError(str(e)) from e
                raise e
            except httpx.TransportError as e:
                if attempt < self.max_attempts - 1:
                    logger.warning(f"OpenAI connection error: {e!r}. Retrying in {retry_sleep_time} seconds...")
                    time.sleep(retry_sleep_time)
                    continue
                raise e

    def generate_completion(self, prompt: str, temperature: float, target_language: str = None) -> (str, int):
        # Temperature is omitted (only the default 1 is accepted), and the schema is non-strict because
        # strict mode can't express the [id, thoughts, translation] tuple (no prefixItems)
        data = self._post_with_retries("chat/completions", {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": self.params["max_output_tokens"],
            "reasoning_effort": self.params["reasoning_effort"],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "translation", "strict": False, "schema": build_json_schema(target_language)},
            },
        })

        usage = data.get("usage") or {}
        output_token_count = usage.get("completion_tokens") or 0
        self.total_input_tokens += usage.get("prompt_tokens") or 0
        self.total_output_tokens += output_token_count
        self.total_cached_tokens += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0

        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        message = choice.get("message") or {}
        message_text = message.get("content")
        if not message_text:
            if message.get("refusal") or finish_reason == "content_filter":
                raise RefuseToTranslateError(f"Model refused: {message.get('refusal') or finish_reason}")
            if finish_reason == "length":
                raise ResponseTooLongError("Response too long and was cut off")
            message_text = f"finish_reason: {finish_reason}"

        if finish_reason == "length":
            output_token_count = self.params["max_output_tokens"]

        return message_text, output_token_count

    def init_vocab(self, text: str):
        data = self._post_with_retries("responses/input_tokens", {"model": self.model_name, "input": text})
        self.average_tokens_per_char = max(data.get("input_tokens") or 0, 1) / len(text)

    def get_total_cost(self) -> float:
        return 0.0

    def num_tokens_from_string(self, string: str) -> int:
        return int(len(string) * self.average_tokens_per_char)

    def max_output_tokens(self) -> int:
        return self.params["max_output_tokens"]
