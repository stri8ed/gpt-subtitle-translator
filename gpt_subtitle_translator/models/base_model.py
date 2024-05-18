from abc import ABC, abstractmethod


class BaseModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    def generate_completion(self, prompt: str, temperature: float) -> (str, int):
        pass

    def num_tokens_from_string(self, string: str) -> int:
        pass

    def max_output_tokens(self) -> int:
        pass

    def get_total_cost(self) -> float:
        pass