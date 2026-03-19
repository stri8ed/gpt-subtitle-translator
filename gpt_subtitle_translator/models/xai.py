import os
from typing import Union
from pydantic import BaseModel as PydanticBaseModel, RootModel

from xai_sdk import Client
from xai_sdk.chat import user
from dotenv import load_dotenv
import json

from gpt_subtitle_translator.models.base_model import BaseModel
from gpt_subtitle_translator.subtitle_translator import RefuseToTranslateError, ResponseTooLongError

load_dotenv()

model_params = {
    "grok-4-fast-non-reasoning": {
        "price_input": 0.0002,
        "price_output": 0.0015,
        "max_output_tokens": 65_536,
    },
}

class SubtitleTranslation(PydanticBaseModel):
    id: int
    original: str
    translation: str
    thoughts: str

class TranslationList(RootModel[list[SubtitleTranslation]]):
    pass

def get_model_params(model_name: str):
    for key, value in model_params.items():
        if key in model_name:
            return value
    return None


class XAI(BaseModel):
    def __init__(self, model_name: str = "grok-4-fast-non-reasoning", api_key: Union[str, None] = None):
        super().__init__(model_name)
        _model_params = get_model_params(model_name)
        assert _model_params is not None, f"Model {model_name} info not found."
        self.client = Client(api_key=api_key or os.environ["XAI_API_KEY"])
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.params = _model_params
        self.average_tokens_per_char = None

    def generate_completion(self, prompt: str, temperature: float, target_language: str = None) -> (str, int):
        try:
            chat = self.client.chat.create(
                model=self.model_name,
                temperature=temperature,
                max_tokens=self.params["max_output_tokens"]
            )
            chat.append(user(prompt))

            # Use parse() method for structured output
            response, translation_list = chat.parse(TranslationList)

            # Convert response to JSON format matching the expected schema
            translations_list = []
            for translation in translation_list.root:
                translations_list.append({
                    "id": translation.id,
                    "original": translation.original,
                    "translation": translation.translation,
                    "thoughts": translation.thoughts
                })

            message_text = json.dumps(translations_list, ensure_ascii=False, indent=2)

            reasoning_tokens = response.usage.reasoning_tokens or 0
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens + reasoning_tokens

            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens

            return message_text, output_tokens

        except Exception as e:
            error_msg = str(e).lower()
            if "content filtering" in error_msg or "safety" in error_msg:
                raise RefuseToTranslateError("Output blocked by content filtering policy")
            elif "max_tokens" in error_msg or "too long" in error_msg:
                raise ResponseTooLongError("Response too long. Might be missing tokens.")
            else:
                raise RefuseToTranslateError(f"XAI API error: {str(e)}")

    def init_vocab(self, text: str):
        tokens = self.client.tokenize.tokenize_text(text=text, model=self.model_name)
        self.average_tokens_per_char = len(tokens) / len(text)

    def get_total_cost(self) -> float:
        input_cost = (self.total_input_tokens / 1000) * self.params["price_input"]
        output_cost = (self.total_output_tokens / 1000) * self.params["price_output"]
        return input_cost + output_cost

    def num_tokens_from_string(self, string: str) -> int:
        num_chars = len(string)
        return int(num_chars * self.average_tokens_per_char)

    def max_output_tokens(self) -> int:
        return self.params["max_output_tokens"]