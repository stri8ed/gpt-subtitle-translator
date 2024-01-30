import os
from typing import Literal

import openai
import tiktoken
from constants import MODEL, MAX_OUTPUT_TOKENS, TEMPERATURE
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
assert (
    openai.api_key is not None
), "OpenAI API key not found. Please set it in the .env file."

# as of 01/30/2024
openai_pricing_per_1k = {
    "gpt-4-turbo-preview": {
        "input": 0.01,
        "output": 0.03
    },
    "gpt-4": {
        "input": 0.03,
        "output": 0.06
    },
    "gpt-3.5-turbo-1106": {
        "input": 0.0010,
        "output": 0.0020
    }
}

def num_tokens_from_string(string: str) -> int:
    encoding = tiktoken.encoding_for_model(MODEL)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def calculate_cost(
    num_tokens: int, token_type: Literal["input", "output"] = "input"
) -> float:
    cost_per_1k = openai_pricing_per_1k[MODEL][token_type]
    return (num_tokens / 1000) * cost_per_1k

class CompletionGenerator:
    def __init__(self):
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate_completion(self, prompt: str) -> str:
        messages = [{"role": "system", "content": prompt}]
        response = openai.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=MAX_OUTPUT_TOKENS,
            temperature=TEMPERATURE,
            n=1,
            stop=None,
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content

    def get_total_cost(self) -> float:
        input_cost = calculate_cost(self.total_input_tokens, "input")
        output_cost = calculate_cost(self.total_output_tokens, "output")
        return input_cost + output_cost
