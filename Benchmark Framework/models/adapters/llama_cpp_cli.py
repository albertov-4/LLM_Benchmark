"""llama.cpp CLI adapter for local GGUF model execution."""

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
import re
from time import perf_counter
from typing import Any


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
class LlamaCppCLIConfig:
    model_id: str
    executable_path: str
    model_path: str
    temperature: float = 0.0
    top_k: int | None = 10
    top_p: float | None = 0.95
    max_tokens: int = 4096
    context_size: int | None = None
    threads: int | None = None
    timeout_seconds: int = 300
    extra_args: list[str] = field(default_factory=list)


class LlamaCppCLIAdapter:
    """Adapter that invokes a llama.cpp command-line binary per generation."""

    def __init__(self, config: LlamaCppCLIConfig):
        self.config = config

    @staticmethod
    def _render_messages(messages: list[dict[str, str]]) -> str:
        rendered_lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", "")).strip()
            rendered_lines.append(f"{role}:\n{content}")
        rendered_lines.append("ASSISTANT:\n")
        return "\n\n".join(rendered_lines).strip()

    def _validate_paths(self) -> None:
        executable = Path(self.config.executable_path)
        if not executable.exists():
            raise FileNotFoundError(f"llama.cpp executable not found: {executable}")

        model = Path(self.config.model_path)
        if not model.exists():
            raise FileNotFoundError(f"llama.cpp model file not found: {model}")

    def _build_command(self, prompt: str) -> list[str]:
        command = [
            self.config.executable_path,
            "-m",
            self.config.model_path,
            "-p",
            prompt,
            "-n",
            str(self.config.max_tokens),
            "--temp",
            str(self.config.temperature),
        ]

        if self.config.top_k is not None:
            command.extend(["--top-k", str(self.config.top_k)])
        if self.config.top_p is not None:
            command.extend(["--top-p", str(self.config.top_p)])
        if self.config.context_size is not None:
            command.extend(["-c", str(self.config.context_size)])
        if self.config.threads is not None:
            command.extend(["-t", str(self.config.threads)])

        command.extend(self.config.extra_args)
        return command

    @staticmethod
    def _clean_output(stdout_text: str, prompt: str) -> str:
        cleaned = stdout_text.strip()
        if cleaned.startswith(prompt):
            cleaned = cleaned[len(prompt):].strip()
        return cleaned

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Run one llama.cpp CLI generation and return a normalized payload."""
        self._validate_paths()
        prompt = self._render_messages(messages)
        command = self._build_command(prompt)

        start_time = perf_counter()
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"llama.cpp generation timed out after {self.config.timeout_seconds} seconds."
            ) from exc
        latency_s = perf_counter() - start_time

        if completed.returncode != 0:
            raise RuntimeError(
                "llama.cpp generation failed with exit code "
                f"{completed.returncode}: {completed.stderr.strip()}"
            )

        raw_text = self._clean_output(completed.stdout, prompt)
        extracted = _extract_reasoning_from_text(raw_text)

        return {
            "model_id": self.config.model_id,
            "raw_text": extracted["raw_text"],
            "reasoning_text": extracted["reasoning_text"],
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "latency_s": latency_s,
            "notes": extracted["notes"],
        }
