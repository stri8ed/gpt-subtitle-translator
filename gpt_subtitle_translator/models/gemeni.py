import os
from typing import Union

import google.generativeai as genai
import tiktoken
from dotenv import load_dotenv
from google.generativeai import GenerationConfig

from gpt_subtitle_translator.models.base_model import BaseModel

load_dotenv()

model_params = {
    "gemini-1.5-flash-latest": {
        "price_input": 0.00035,
        "price_output": 0.00053,
        "max_output_tokens": 8192,
    },
}

genai.configure(api_key=os.environ["GEMENI_API_KEY"])

class Gemeni(BaseModel):
    def __init__(self, model_name: str = "gemini-1.5-flash-latest", params: Union[None, dict] = None):
        super().__init__(model_name)
        self.client = genai.GenerativeModel(model_name)
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.params = model_params[model_name]


    def generate_completion(self, prompt: str, temperature: float) -> (str, int):
        message = self.client.generate_content(
            contents=[prompt],
            generation_config=GenerationConfig(
                temperature=temperature,
                max_output_tokens=self.params["max_output_tokens"],
            ),

        )
        self.total_input_tokens +=message.usage_metadata.prompt_token_count
        self.total_output_tokens += message.usage_metadata.candidates_token_count
        return message.text, message.usage_metadata.candidates_token_count

    def get_total_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1000) * self.params["price_input"]
        output_cost = (self.total_output_tokens / 1000) * self.params["price_output"]
        return input_cost + output_cost

    def num_tokens_from_string(self, string: str) -> int:
        """
        This is not correct. No tokenizer is available for Claude models.
        """
        if len(string) == 0:
            return 0
        res = self.client.count_tokens([string])
        return res.total_tokens

    def max_output_tokens(self) -> int:
        return self.params["max_output_tokens"]
