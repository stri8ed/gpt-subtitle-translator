import json
import os
from typing import Union

import google.generativeai as genai
from dotenv import load_dotenv
from google.generativeai import GenerationConfig
from google.ai.generativelanguage_v1 import HarmCategory
from google.generativeai.types import HarmBlockThreshold
from google.generativeai.types.answer_types import FinishReason

from gpt_subtitle_translator.models.base_model import BaseModel
from gpt_subtitle_translator.subtitle_translator import RefuseToTranslateError

load_dotenv()

model_params = {
    "gemini-1.5-flash-latest": {
        "price_input": 0.000075,
        "price_output": 0.0003,
        "max_output_tokens": 8192,
    },
    "gemini-exp-1206": {
        "price_input": 0.00035,
        "price_output": 0.00053,
        "max_output_tokens": 8192,
    },
    "gemini-1.5-pro-latest": {
        "price_input": 0.00125,
        "price_output": 0.005,
        "max_output_tokens": 8192,
    }
}

model_params['gemini-2.0-flash-exp'] = model_params['gemini-1.5-flash-latest']

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class Gemini(BaseModel):
    def __init__(self, model_name: str = "gemini-2.0-flash-exp", params: Union[None, dict] = None):
        super().__init__(model_name)
        assert model_name in model_params, f"Model {model_name} info not found."
        self.client = genai.GenerativeModel(model_name)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.params = model_params[model_name]
        self.average_tokens_per_char = None


    def generate_completion(self, prompt: str, temperature: float) -> (str, int):
        message = self.client.generate_content(
            contents=[prompt],
            request_options={"timeout": 1000},
            safety_settings=[
                {
                    "category": HarmCategory.HARM_CATEGORY_HARASSMENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                },
                {
                    "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    "threshold": HarmBlockThreshold.BLOCK_NONE,
                }
            ],
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=self.params["max_output_tokens"],
            ),
        )

        if message.parts:
            message_text = message.text
        else:
            if message.candidates[0].finish_reason == FinishReason.SAFETY:
                raise RefuseToTranslateError("Output blocked by content filtering policy")
            else:
                message_text = f"finish_reason: {message.candidates[0].finish_reason}"

        usage = message.usage_metadata
        output_token_count = usage.candidates_token_count
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
        res = self.client.count_tokens([string])
        return max(res.total_tokens, 1)

    def max_output_tokens(self) -> int:
        return self.params["max_output_tokens"]
