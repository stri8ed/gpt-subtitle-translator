import os

import anthropic
import tiktoken
from anthropic import AnthropicBedrock
from dotenv import load_dotenv

from constants import TEMPERATURE
from models.base_model import BaseModel

load_dotenv()

model_params = {
    "claude-3-haiku-20240307": {
        "price_input": 0.00025,
        "price_output": 0.00125,
        "max_output_tokens": 4096,
    },
    "claude-3-sonnet-20240229": {
        "price_input": 0.003,
        "price_output": 0.015,
        "max_output_tokens": 4096,
    }
}

class Claude(BaseModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        if model_name.startswith("anthropic."):
            self.client = AnthropicBedrock()
            model_name = next(key for key in model_params if key in model_name)
        else:
            assert (os.getenv("ANTHROPIC_API_KEY") is not None),\
                "Anthropic API key not found. Please set it in the .env file."
            self.client = anthropic.Anthropic()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.params = model_params[model_name]


    def generate_completion(self, prompt: str) -> str:
        message = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.params["max_output_tokens"],
            temperature=TEMPERATURE,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        self.total_input_tokens += message.usage.input_tokens
        self.total_output_tokens += message.usage.output_tokens
        return message.content[0].text

    def get_total_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1000) * self.params["price_input"]
        output_cost = (self.total_output_tokens / 1000) * self.params["price_output"]
        return input_cost + output_cost

    def num_tokens_from_string(self, string: str) -> int:
        """
        This is not correct. No tokenizer is available for Claude models.
        """
        encoding = tiktoken.encoding_for_model("gpt-4")
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def max_output_tokens(self) -> int:
        return self.params["max_output_tokens"]
