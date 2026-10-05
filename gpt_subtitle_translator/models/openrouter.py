import os
import time
from typing import Union

import httpx
from dotenv import load_dotenv

from gpt_subtitle_translator.logger import logger
from gpt_subtitle_translator.models.base_model import BaseModel
from gpt_subtitle_translator.models.schema import build_json_schema
from gpt_subtitle_translator.subtitle_translator import RefuseToTranslateError, ResponseTooLongError, QuotaExhaustedError

load_dotenv()

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Cost is taken from OpenRouter's reported usage.cost, so no price table is kept here.
model_params = {
    # Reasoning is mandatory for Muse Spark; "minimal" is the lowest effort supported.
    # The contributor tier lets Meta train on prompts, so the OpenRouter account must allow
    # "paid model training" endpoints (https://openrouter.ai/settings/privacy).
    "meta/muse-spark-1.3": {
        "max_output_tokens": 65_536,
        "reasoning_effort": "minimal",
    },
}


def get_model_params(model_name: str):
    for key, value in model_params.items():
        if key in model_name:
            return value
    return None


class OpenRouterError(Exception):
    def __init__(self, status_code: int, message: str, from_upstream: bool = False):
        super().__init__(f"OpenRouter error {status_code}: {message}")
        self.status_code = status_code
        # True when the error was relayed from the upstream provider rather than raised by OpenRouter itself
        self.from_upstream = from_upstream


class OpenRouter(BaseModel):
    def __init__(self, model_name: str = "meta/muse-spark-1.3-contributor", api_key: Union[str, None] = None):
        super().__init__(model_name)
        _model_params = get_model_params(model_name)
        assert _model_params is not None, f"Model {model_name} info not found."
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {api_key or os.environ['OPENROUTER_API_KEY']}"},
            timeout=httpx.Timeout(60 * 5, connect=10),
        )
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.total_cost = 0.0
        self.params = _model_params
        self.average_tokens_per_char = None
        self.max_attempts = 3

    def _post(self, body: dict) -> dict:
        response = self.client.post(API_URL, json=body)
        try:
            data = response.json()
        except ValueError:
            raise OpenRouterError(response.status_code, response.text[:1000]) from None

        # Errors can arrive as a non-2xx status, or as a 200 whose body (or choice) carries an error
        error = data.get("error") or (data.get("choices") or [{}])[0].get("error")
        if response.status_code >= 400 or error:
            error = error or {}
            message = error.get("message", response.text[:1000])
            raw = (error.get("metadata") or {}).get("raw")
            if raw:
                message += f" ({str(raw)[:1000]})"
            raise OpenRouterError(error.get("code") or response.status_code, message, from_upstream=bool(raw))
        return data

    def _post_with_retries(self, body: dict) -> dict:
        for attempt in range(self.max_attempts):
            retry_sleep_time = 2 * (attempt + 1)
            try:
                return self._post(body)
            except OpenRouterError as e:
                if e.status_code in (402, 429):
                    raise QuotaExhaustedError(str(e)) from e
                if e.status_code == 403:
                    raise RefuseToTranslateError(f"Input flagged by moderation: {e}") from e
                # Meta intermittently answers model_not_found (404) under concurrent load
                transient = e.status_code == 408 or e.status_code >= 500 or (e.status_code == 404 and e.from_upstream)
                if attempt < self.max_attempts - 1 and transient:
                    logger.warning(f"OpenRouter server error: {e}. Retrying in {retry_sleep_time} seconds...")
                    time.sleep(retry_sleep_time)
                    continue
                raise e
            except httpx.TransportError as e:
                if attempt < self.max_attempts - 1:
                    logger.warning(f"OpenRouter connection error: {e!r}. Retrying in {retry_sleep_time} seconds...")
                    time.sleep(retry_sleep_time)
                    continue
                raise e

    def _record_usage(self, usage: dict) -> int:
        """Accumulates usage and returns the output token count, which includes reasoning tokens."""
        output_token_count = usage.get("completion_tokens") or 0
        self.total_input_tokens += usage.get("prompt_tokens") or 0
        self.total_output_tokens += output_token_count
        self.total_cached_tokens += (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        self.total_cost += usage.get("cost") or 0
        return output_token_count

    def generate_completion(self, prompt: str, temperature: float, target_language: str = None) -> (str, int):
        body = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.params["max_output_tokens"],
            "reasoning": {"effort": self.params["reasoning_effort"], "exclude": True},
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "translation", "strict": True, "schema": build_json_schema(target_language)},
            },
            "provider": {"require_parameters": True},
        }

        data = self._post_with_retries(body)
        output_token_count = self._record_usage(data.get("usage") or {})

        choice = data["choices"][0]
        finish_reason = choice.get("finish_reason")
        message_text = (choice.get("message") or {}).get("content")
        if not message_text:
            match finish_reason:
                case "content_filter":
                    raise RefuseToTranslateError("Output blocked by content filtering policy")
                case "length":
                    raise ResponseTooLongError("Response too long and was cut off")
                case _:
                    message_text = f"finish_reason: {finish_reason}"

        if finish_reason == "length":
            output_token_count = self.params["max_output_tokens"]

        return message_text, output_token_count

    def init_vocab(self, text: str):
        token_count = self._get_token_count(text)  # no tokenize endpoint, so this costs a request; only do it once
        self.average_tokens_per_char = token_count / len(text)

    def _get_token_count(self, string: str) -> int:
        """OpenRouter has no token counting endpoint, so send the text with a minimal output budget
        and read the billed prompt token count."""
        data = self._post_with_retries({
            "model": self.model_name,
            "messages": [{"role": "user", "content": string}],
            "max_tokens": 16,  # Muse Spark rejects anything lower
            "reasoning": {"effort": self.params["reasoning_effort"], "exclude": True},
        })
        usage = data.get("usage") or {}
        self._record_usage(usage)
        return max(usage.get("prompt_tokens") or 0, 1)

    def get_total_cost(self) -> float:
        return self.total_cost

    def num_tokens_from_string(self, string: str) -> int:
        num_chars = len(string)
        return int(num_chars * self.average_tokens_per_char)

    def max_output_tokens(self) -> int:
        return self.params["max_output_tokens"]
