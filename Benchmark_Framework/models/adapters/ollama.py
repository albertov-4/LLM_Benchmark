"""Ollama adapter for benchmark runs.

This adapter talks to a local Ollama server through its HTTP API while keeping
the same normalized `generate(messages)` interface used by the runner.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import json


@dataclass(slots=True)
class OllamaConfig:
    model_id: str
    ollama_model: str = ""
    base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    top_k: int | None = 10
    max_tokens: int = 4096
    timeout_seconds: int = 300


class OllamaAdapter:
    """Adapter used by the benchmark runner for Ollama generation."""

    def __init__(self, config: OllamaConfig):
        self.config = config

    def _resolve_ollama_model(self) -> str:
        """Return the concrete model name passed to Ollama."""
        ollama_model = self.config.ollama_model.strip()
        if ollama_model:
            return ollama_model
        return self.config.model_id

    def _chat_url(self) -> str:
        """Return the Ollama chat endpoint URL."""
        return f"{self.config.base_url.rstrip('/')}/api/chat"

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the JSON body expected by Ollama."""
        options: dict[str, Any] = {
            "temperature": self.config.temperature,
            "num_predict": self.config.max_tokens,
        }
        if self.config.top_k is not None:
            options["top_k"] = self.config.top_k

        return {
            "model": self._resolve_ollama_model(),
            "messages": [
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
                for message in messages
            ],
            "stream": False,
            "options": options,
        }

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one non-streaming chat request to Ollama."""
        request = Request(
            self._chat_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                response_text = response.read().decode("utf-8")
        except HTTPError as exc:  # pragma: no cover - depends on local server
            error_text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama request failed with HTTP {exc.code}: {error_text}") from exc
        except URLError as exc:  # pragma: no cover - depends on local server
            raise RuntimeError(
                "Unable to reach Ollama. Verify that Ollama is running and that "
                f"`base_url` is correct: {self.config.base_url}"
            ) from exc

        try:
            decoded = json.loads(response_text)
        except json.JSONDecodeError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Ollama returned invalid JSON: {response_text[:500]}") from exc

        if not isinstance(decoded, dict):
            raise RuntimeError("Ollama returned an unexpected non-object response.")
        return decoded

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Run one Ollama generation and return a normalized payload."""
        payload = self._build_payload(messages)

        start_time = perf_counter()
        response = self._post_chat(payload)
        latency_s = perf_counter() - start_time

        message = response.get("message", {})
        raw_text = ""
        if isinstance(message, dict):
            raw_text = str(message.get("content", "")).strip()

        prompt_tokens = response.get("prompt_eval_count")
        completion_tokens = response.get("eval_count")
        usage = {
            "prompt_tokens": prompt_tokens if isinstance(prompt_tokens, int) else None,
            "completion_tokens": completion_tokens if isinstance(completion_tokens, int) else None,
            "total_tokens": (
                prompt_tokens + completion_tokens
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int)
                else None
            ),
        }

        notes: list[str] = []
        if response.get("done") is False:
            notes.append("Ollama response reported done=false.")

        return {
            "model_id": self.config.model_id,
            "raw_text": raw_text,
            "usage": usage,
            "latency_s": latency_s,
            "notes": notes,
            "provider_response": {
                "model": response.get("model"),
                "done": response.get("done"),
                "total_duration": response.get("total_duration"),
                "load_duration": response.get("load_duration"),
                "prompt_eval_duration": response.get("prompt_eval_duration"),
                "eval_duration": response.get("eval_duration"),
            },
        }
