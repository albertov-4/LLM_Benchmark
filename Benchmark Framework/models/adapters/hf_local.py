"""Adapter scaffold for local Hugging Face models.

The goal here is not to lock the implementation too early.
This file documents the interface and the main steps for a local adapter.
"""

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
        self.model = None
        self.tokenizer = None

    def load_model_pseudocode(self) -> None:
        """Pseudocode for local model loading.

        Suggested implementation:
        1. load tokenizer from `weights_path`
        2. load model from `weights_path`
        3. move model to GPU or CPU
        4. store handles on `self.model` and `self.tokenizer`
        """
        # TODO: replace this placeholder with real HF loading logic.
        self.model = "TODO_MODEL_HANDLE"
        self.tokenizer = "TODO_TOKENIZER_HANDLE"

    def generate(self, messages: List[Dict[str, str]]) -> Dict[str, object]:
        """Pseudocode generation entry point.

        Suggested implementation:
        1. make sure the model is loaded
        2. convert `messages` to the tokenizer chat template
        3. call `model.generate(...)`
        4. decode only the newly generated tokens
        5. return a normalized payload
        """
        return {
            "model_id": self.config.model_id,
            "raw_text": "",
            "usage": {},
            "latency_s": None,
            "notes": [
                "TODO: format messages with tokenizer chat template",
                "TODO: run local generation",
                "TODO: decode and normalize response payload",
            ],
        }
