"""Adapter scaffold for local Hugging Face models."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HFLocalConfig:
    model_id: str
    weights_path: str
    temperature: float = 0.0
    top_k: int = 10
    max_tokens: int = 4096


class HFLocalAdapter:
    """Minimal interface expected by the benchmark runner."""

    def __init__(self, config: HFLocalConfig):
        self.config = config

    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, object]:
        """Placeholder implementation for future local inference."""
        return {
            "model_id": self.config.model_id,
            "raw_text": "",
            "usage": {},
            "latency_s": None,
        }
