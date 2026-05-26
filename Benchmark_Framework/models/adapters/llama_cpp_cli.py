"""llama.cpp CLI adapter for local GGUF model execution."""

from dataclasses import dataclass, field
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any


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

        return {
            "model_id": self.config.model_id,
            "raw_text": self._clean_output(completed.stdout, prompt),
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "latency_s": latency_s,
            "notes": [],
        }
