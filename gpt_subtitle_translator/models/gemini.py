import os
import time
from typing import Union

from dotenv import load_dotenv
from google import genai
from google.genai.errors import ClientError
from google.genai.types import FinishReason, GenerateContentConfig, \
    HttpOptions, SafetySetting, ThinkingConfig, ThinkingLevel

from gpt_subtitle_translator.logger import logger
from gpt_subtitle_translator.models.base_model import BaseModel
from gpt_subtitle_translator.subtitle_translator import RefuseToTranslateError, ResponseTooLongError, QuotaExhaustedError

load_dotenv()

model_params = {
    "gemini-2.0-flash": {
        "price_input": 0.0001,
        "price_output": 0.0004,
        "price_cached": 0.000025,
        "max_output_tokens": 8192,
        "thinking_enabled": False,
    },
    "gemini-2.5-flash": {
        "price_input": 0.0003,
        "price_output": 0.0025,
        "price_cached": 0.000075,
        "max_output_tokens": 65_536,
        "thinking_enabled": True,
    },
    "gemini-3-flash": {
        "price_input": 0.0005,
        "price_output": 0.003,
        "max_output_tokens": 65_536,
        "thinking_enabled": True,
    },
    "gemini-2.5-flash-lite-preview": {
        "price_input": 0.0001,
        "price_output": 0.0004,
        "max_output_tokens": 65_536,
        "thinking_enabled": True,
    },
}

JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "The subtitle ID number matching the original subtitle"
            },
            "original": {
                "type": "string",
                "description": "The original subtitle text in the source language",
                "maxLength": 750
            },
            "thoughts": {
                "type": "string",
                "description": "Brief reasoning about ambiguities, errors, or challenging translations. Empty string for straightforward cases.",
                "maxLength": 650,
            },
            "translation": {
                "type": "string",
                "description": "The translated subtitle text in the target language",
                "maxLength": 1000,
            }
        },
        "required": [
            "id",
            "original",
            "thoughts",
            "translation"
        ]
    }
}


def get_model_params(model_name: str):
    for key, value in model_params.items():
        if key in model_name:
            return value
    return None


class Gemini(BaseModel):
    def __init__(self, model_name: str = "gemini-2.0-flash-001", api_key: Union[str, None] = None):
        super().__init__(model_name)
        _model_params = get_model_params(model_name)
        assert _model_params is not None, f"Model {model_name} info not found."
        self.client = genai.Client(api_key=api_key or os.environ["GEMINI_API_KEY"])
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cached_tokens = 0
        self.params = _model_params
        self.average_tokens_per_char = None
        self.max_attempts = 3

    def generate_completion(self, prompt: str, temperature: float) -> (str, int):
        message = None
        thinking_config = None

        if self.params['thinking_enabled']:
            kwargs = {}
            if "3" not in self.model_name:
                kwargs['thinking_budget'] = 0
            else:
                kwargs['thinking_level'] = ThinkingLevel.MINIMAL
            thinking_config = ThinkingConfig(**kwargs)

        for attempt in range(self.max_attempts):
            try:
                message = self.client.models.generate_content(
                    contents=[prompt],
                    model=self.model_name,
                    config=GenerateContentConfig(
                        temperature=temperature,
                        response_schema=JSON_SCHEMA,
                        response_mime_type="application/json",
                        max_output_tokens=self.params["max_output_tokens"],
                        http_options=HttpOptions(
                            timeout=1000 * 60 * 5
                        ),
                        thinking_config=thinking_config,
                        safety_settings=[
                            SafetySetting(
                                category="HARM_CATEGORY_HARASSMENT",
                                threshold="OFF"
                            ),
                            SafetySetting(
                                category="HARM_CATEGORY_CIVIC_INTEGRITY",
                                threshold="OFF"
                            ),
                            SafetySetting(
                                category="HARM_CATEGORY_HATE_SPEECH",
                                threshold="OFF"
                            ),
                            SafetySetting(
                                category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
                                threshold="OFF"
                            ),
                            SafetySetting(
                                category="HARM_CATEGORY_DANGEROUS_CONTENT",
                                threshold="OFF"
                            )
                        ])
                )
                break
            except ClientError as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    raise QuotaExhaustedError(error_str) from e

                if attempt < self.max_attempts - 1:
                    logger.warning(f"Gemini client error: {e}. Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                raise e
            except Exception as e:
                if attempt < self.max_attempts - 1 and ("Server disconnected" in str(e) or "RemoteProtocolError" in type(e).__name__):
                    logger.warning(f"Gemini connection error: {e}. Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                raise e

        if message.text:
            message_text = message.text
        else:
            if message.candidates is None:
                block_reason = message.prompt_feedback.block_reason if message.prompt_feedback else None
                raise RefuseToTranslateError(f"Prompt blocked for reason: {block_reason or 'Unknown'}. Prompt: {prompt[-1000:]}")
            if message.candidates[0].finish_reason == FinishReason.SAFETY:
                raise RefuseToTranslateError("Output blocked by content filtering policy")
            if message.candidates[0].finish_reason == FinishReason.RECITATION:
                raise RefuseToTranslateError("Output blocked due to RECITATION policy")
            elif message.candidates[0].finish_reason == FinishReason.MAX_TOKENS:
                raise ResponseTooLongError("Response too long. Might be missing tokens.")
            else:
                message_text = f"finish_reason: {message.candidates[0].finish_reason}"

        usage = message.usage_metadata
        cached_token_count = (usage.cached_content_token_count or 0 if hasattr(usage, 'cached_content_token_count') else 0)
        thought_tokens = (usage.thoughts_token_count or 0 if hasattr(usage, 'thoughts_token_count') else 0)
        output_token_count = (usage.candidates_token_count or 0) + thought_tokens
        input_token_count = usage.prompt_token_count or 0
        self.total_input_tokens += input_token_count
        self.total_output_tokens += output_token_count
        self.total_cached_tokens += cached_token_count
        return message_text, output_token_count

    def init_vocab(self, text: str):
        token_count = self._get_token_count(text)  # get token count requires an http request, so we only do it once
        self.average_tokens_per_char = token_count / len(text)

    def get_total_cost(self) -> float:
        input_tokens = self.total_input_tokens - self.total_cached_tokens
        input_cost = (input_tokens / 1000) * self.params["price_input"]
        input_cost += self.total_cached_tokens / 1000 * self.params.get("price_cached", 0)
        output_cost = (self.total_output_tokens / 1000) * self.params["price_output"]
        return input_cost + output_cost

    def num_tokens_from_string(self, string: str) -> int:
        num_chars = len(string)
        return int(num_chars * self.average_tokens_per_char)

    def _get_token_count(self, string: str) -> int:
        res = self.client.models.count_tokens(
            model=self.model_name,
            contents=string,
        )
        return max(res.total_tokens, 1)

    def max_output_tokens(self) -> int:
        return self.params["max_output_tokens"]
