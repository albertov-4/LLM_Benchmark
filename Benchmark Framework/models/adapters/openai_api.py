"""Adapter scaffold for API-based models."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class OpenAIAdapterConfig:
    model_id: str
    api_model_name: str
    temperature: float = 0.0
    max_tokens: int = 4096


class OpenAIAPIAdapter:
    """Common interface placeholder for remote chat-completions style APIs."""

    def __init__(self, config: OpenAIAdapterConfig):
        self.config = config

    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "raw_text": "",
            "usage": {},
            "latency_s": None,
        }
