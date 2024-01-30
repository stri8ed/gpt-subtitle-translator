import os
from typing import Literal
import openai
import tiktoken
from constants import TEMPERATURE
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
assert (
    openai.api_key is not None
), "OpenAI API key not found. Please set it in the .env file."

# as of 01/30/2024
model_params = {
    "gpt-4-turbo-preview": {
        "price_input": 0.01,
        "price_output": 0.03,
        "max_output_tokens": 4096,
    },
    "gpt-4": {
        "price_input": 0.03,
        "price_output": 0.06,
        "max_output_tokens": 8192,
    },
    "gpt-3.5-turbo-1106": {
        "price_input": 0.0010,
        "price_output": 0.0020,
        "max_output_tokens": 4096,
    }
}

def calculate_cost(
    model: str,
    num_tokens: int, token_type: Literal["input", "output"] = "input"
) -> float:
    cost_per_1k = model_params[model][f"price_{token_type}"]
    return (num_tokens / 1000) * cost_per_1k

class CompletionGenerator:
    def __init__(self, model: str):
        assert model in model_params, f"Model {model} info not found."
        self.model = model
        self.max_output_tokens = model_params[model]["max_output_tokens"]
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate_completion(self, prompt: str) -> str:
        messages = [{"role": "system", "content": prompt}]
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_output_tokens,
            temperature=TEMPERATURE,
            n=1,
            stop=None,
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    def get_total_cost(self) -> float:
        input_cost = calculate_cost(self.model, self.total_input_tokens, "input")
        output_cost = calculate_cost(self.model, self.total_output_tokens, "output")
        return input_cost + output_cost

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model)
        num_tokens = len(encoding.encode(string))
        return num_tokens
