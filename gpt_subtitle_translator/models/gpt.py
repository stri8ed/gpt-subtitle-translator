import os
import openai
import tiktoken
from dotenv import load_dotenv

from gpt_subtitle_translator.models.base_model import BaseModel

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
    "gpt-3.5-turbo-0125": {
        "price_input": 0.0010,
        "price_output": 0.0020,
        "max_output_tokens": 4096,
    }
}

class GPT(BaseModel):
    def __init__(self, model_name: str):
        assert model_name in model_params, f"Model {model_name} info not found."
        super().__init__(model_name)
        self.params = model_params[model_name]
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def generate_completion(self, prompt: str, temperature: float) -> (str, int):
        messages = [{"role": "system", "content": prompt}]
        response = openai.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.params["max_output_tokens"],
            temperature=temperature,
            n=1,
            stop=None,
        )
        self.total_input_tokens += response.usage.prompt_tokens
        self.total_output_tokens += response.usage.completion_tokens
        return response.choices[0].message.content, response.usage.completion_tokens

    def get_total_cost(self) -> float:
        input_cost =  (self.total_input_tokens / 1000) * self.params["price_input"]
        output_cost = (self.total_output_tokens / 1000) * self.params["price_output"]
        return input_cost + output_cost

    def num_tokens_from_string(self, string: str) -> int:
        encoding = tiktoken.encoding_for_model(self.model_name)
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def max_output_tokens(self) -> int:
        return self.params["max_output_tokens"]
