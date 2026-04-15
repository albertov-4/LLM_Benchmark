"""Adapter scaffold for vLLM-backed models."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VLLMConfig:
    model_id: str
    endpoint: str
    temperature: float = 0.0
    max_tokens: int = 4096


class VLLMAdapter:
    """Placeholder adapter for a shared vLLM inference endpoint."""

    def __init__(self, config: VLLMConfig):
        self.config = config

    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, object]:
        return {
            "model_id": self.config.model_id,
            "raw_text": "",
            "usage": {},
            "latency_s": None,
        }
