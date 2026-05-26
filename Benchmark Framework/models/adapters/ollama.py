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
import re


THINKING_TAG_PATTERN = re.compile(
    r"<(?P<tag>think|thinking|reasoning|analysis)\b[^>]*>(?P<body>.*?)</(?P=tag)>",
    re.IGNORECASE | re.DOTALL,
)
OPEN_THINKING_TAG_PATTERN = re.compile(
    r"^\s*<(?P<tag>think|thinking|reasoning|analysis)\b[^>]*>(?P<body>.*)$",
    re.IGNORECASE | re.DOTALL,
)
FINAL_MARKER_PATTERN = re.compile(
    r"(?im)^\s*(?:final\s+answer|final\s+plan|answer|plan|actions?)\s*:?\s*$"
)


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _join_nonempty(parts: list[str]) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip()).strip()


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in text.strip().splitlines()]
    normalized: list[str] = []
    previous_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank and previous_blank:
            continue
        normalized.append(line)
        previous_blank = is_blank
    return "\n".join(normalized).strip()


def _extract_reasoning_from_text(raw_text: Any, explicit_reasoning: Any = "") -> dict[str, Any]:
    answer_text = _clean_text(raw_text)
    reasoning_parts: list[str] = []
    notes: list[str] = []

    provider_reasoning = _clean_text(explicit_reasoning)
    if provider_reasoning:
        reasoning_parts.append(provider_reasoning)
        notes.append("provider reasoning captured separately from raw_text.")

    inline_reasoning_parts: list[str] = []

    def replace_thinking_block(match: re.Match[str]) -> str:
        inline_reasoning = _clean_text(match.group("body"))
        if inline_reasoning:
            inline_reasoning_parts.append(inline_reasoning)
        return "\n"

    answer_text = THINKING_TAG_PATTERN.sub(replace_thinking_block, answer_text)

    if inline_reasoning_parts:
        reasoning_parts.extend(inline_reasoning_parts)
        notes.append("inline reasoning extracted from raw_text.")
    else:
        open_match = OPEN_THINKING_TAG_PATTERN.match(answer_text)
        if open_match:
            body = _clean_text(open_match.group("body"))
            marker = FINAL_MARKER_PATTERN.search(body)
            if marker:
                inline_reasoning = _clean_text(body[: marker.start()])
                final_answer = _clean_text(body[marker.end() :])
                if inline_reasoning and final_answer:
                    reasoning_parts.append(inline_reasoning)
                    answer_text = final_answer
                    notes.append("inline reasoning extracted from raw_text.")

    return {
        "raw_text": _normalize_text(answer_text),
        "reasoning_text": _join_nonempty(reasoning_parts),
        "notes": notes,
    }


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
        provider_reasoning_parts: list[str] = []
        if isinstance(message, dict):
            raw_text = str(message.get("content", "")).strip()
            for key in ("thinking", "reasoning", "reasoning_content"):
                value = message.get(key)
                if value:
                    provider_reasoning_parts.append(str(value))

        for key in ("thinking", "reasoning", "reasoning_content"):
            value = response.get(key)
            if value:
                provider_reasoning_parts.append(str(value))

        extracted = _extract_reasoning_from_text(
            raw_text,
            "\n\n".join(provider_reasoning_parts),
        )
        raw_text = extracted["raw_text"]

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
        notes.extend(extracted["notes"])

        return {
            "model_id": self.config.model_id,
            "raw_text": raw_text,
            "reasoning_text": extracted["reasoning_text"],
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
