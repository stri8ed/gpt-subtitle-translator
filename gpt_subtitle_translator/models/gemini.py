import json
import os
from typing import Union

from google import genai

from dotenv import load_dotenv
from google.genai.types import HarmBlockThreshold, FinishReason, GenerateContentConfigDict, GenerateContentConfig, \
    HttpOptions, SafetySetting, ThinkingConfig

from gpt_subtitle_translator.models.base_model import BaseModel
from gpt_subtitle_translator.subtitle_translator import RefuseToTranslateError, ResponseTooLongError

load_dotenv()

model_params = {
    "gemini-1.5-flash-latest": {
        "price_input": 0.000075,
        "price_output": 0.0003,
        "max_output_tokens": 8192,
        "thinking_enabled": False,
    },
    "gemini-exp-1206": {
        "price_input": 0.00035,
        "price_output": 0.00053,
        "max_output_tokens": 8192,
    "thinking_enabled": False,
    },
    "gemini-1.5-pro-latest": {
        "price_input": 0.00125,
        "price_output": 0.005,
        "max_output_tokens": 8192,
    "thinking_enabled": False,
    },
    "gemini-2.0-flash" : {
        "price_input": 0.0001,
        "price_output": 0.0004,
        "max_output_tokens": 8192,
        "thinking_enabled": False,
    },
    "gemini-2.5-flash-preview" : {
        "price_input": 0.00016,
        "price_output": 0.0006,
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
        "description": "The original subtitle text in the source language"
      },
      "translation": {
        "type": "string",
        "description": "The translated subtitle text in the target language"
      },
      "thoughts": {
        "type": "string",
        "description": "Brief reasoning about ambiguities, errors, or challenging translations. Empty string for straightforward cases.",
        "maxLength": "750",
      }
    },
    "required": [
      "id",
      "original",
      "translation",
      "thoughts"
    ]
  }
}

def get_model_params(model_name: str):
    for key, value in model_params.items():
        if key in model_name:
            return value
    return None


class Gemini(BaseModel):
    def __init__(self, model_name: str = "gemini-2.0-flash-001", params: Union[None, dict] = None):
        super().__init__(model_name)
        _model_params = get_model_params(model_name)
        assert _model_params is not None, f"Model {model_name} info not found."
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.params = _model_params
        self.average_tokens_per_char = None


    def generate_completion(self, prompt: str, temperature: float) -> (str, int):
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
                thinking_config=self.params['thinking_enabled'] and ThinkingConfig(
                    thinking_budget=0
                ) or None,
                safety_settings=[
                    SafetySetting(
                        category="HARM_CATEGORY_HARASSMENT",
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

        if message.text:
            message_text = message.text
        else:
            if message.candidates[0].finish_reason == FinishReason.SAFETY:
                raise RefuseToTranslateError("Output blocked by content filtering policy")
            elif message.candidates[0].finish_reason == FinishReason.MAX_TOKENS:
                raise ResponseTooLongError("Response too long. Might be missing tokens.")
            else:
                message_text = f"finish_reason: {message.candidates[0].finish_reason}"

        usage = message.usage_metadata
        thought_tokens = (usage.thoughts_token_count or 0 if hasattr(usage, 'thoughts_token_count') else 0)
        output_token_count = usage.candidates_token_count + thought_tokens
        input_token_count = usage.prompt_token_count
        self.total_input_tokens += input_token_count
        self.total_output_tokens += output_token_count
        return message_text, output_token_count

    def init_vocab(self, text: str):
        token_count = self._get_token_count(text) # get token count requires an http request, so we only do it once
        self.average_tokens_per_char = token_count / len(text)

    def get_total_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1000) * self.params["price_input"]
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
