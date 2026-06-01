"""NVIDIA API adapter using the OpenAI-compatible client."""

from dataclasses import dataclass, field
from importlib import import_module
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class NvidiaAPIConfig:
    model_id: str
    api_model_name: str
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key_env: str = "NVIDIA_API_KEY"
    secrets_path: str = "secrets.local.json"
    api_mode: str = "chat_completions"
    stream: bool = False
    temperature: float = 0.0
    top_p: float = 0.95
    max_tokens: int = 4096
    timeout_seconds: int = 300
    job_timeout_seconds: int | None = None
    debug_stream: bool = False
    debug_stream_interval_seconds: int = 10
    extra_body: dict[str, Any] = field(default_factory=dict)


class NvidiaAPIAdapter:
    """Adapter for NVIDIA-hosted chat models exposed through OpenAI API shape."""

    def __init__(self, config: NvidiaAPIConfig):
        self.config = config
        self.client: Any | None = None

    def _load_openai_client_class(self):
        try:
            openai_module = import_module("openai")
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError("NvidiaAPIAdapter requires `openai` to be installed.") from exc
        return getattr(openai_module, "OpenAI")

    def _get_api_key(self) -> str:
        api_key = os.environ.get(self.config.api_key_env, "").strip()
        if api_key:
            return api_key

        api_key = self._get_api_key_from_local_secrets()
        if api_key:
            return api_key

        raise RuntimeError(
            f"NVIDIA API key not found. Set `{self.config.api_key_env}` "
            f"or add it to {self._resolve_secrets_path()}."
        )

    def _resolve_secrets_path(self) -> Path:
        secrets_path = Path(self.config.secrets_path)
        if secrets_path.is_absolute():
            return secrets_path
        return Path(__file__).resolve().parents[2] / secrets_path

    def _get_api_key_from_local_secrets(self) -> str:
        secrets_path = self._resolve_secrets_path()
        if not secrets_path.exists():
            return ""

        try:
            payload = json.loads(secrets_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSON in local secrets file: {secrets_path}") from exc

        if not isinstance(payload, dict):
            raise RuntimeError(f"Local secrets file must contain a JSON object: {secrets_path}")

        value = payload.get(self.config.api_key_env, "")
        if not isinstance(value, str):
            raise RuntimeError(
                f"Secret `{self.config.api_key_env}` in {secrets_path} must be a string."
            )
        return value.strip()

    def _get_client(self):
        if self.client is not None:
            return self.client

        OpenAI = self._load_openai_client_class()
        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self._get_api_key(),
            timeout=self.config.timeout_seconds,
        )
        return self.client

    @staticmethod
    def _extract_usage(completion: Any) -> dict[str, Any]:
        usage = getattr(completion, "usage", None)
        if usage is None:
            return {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            }

        return {
            "prompt_tokens": getattr(usage, "prompt_tokens", None),
            "completion_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }

    @staticmethod
    def _extract_raw_text(completion: Any) -> str:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return ""

        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is None:
            return ""

        return str(getattr(message, "content", "") or "").strip()

    @staticmethod
    def _extract_finish_reason(completion: Any) -> str | None:
        choices = getattr(completion, "choices", None) or []
        if not choices:
            return None

        finish_reason = getattr(choices[0], "finish_reason", None)
        return str(finish_reason) if finish_reason is not None else None

    @staticmethod
    def _render_messages_as_input(messages: list[dict[str, str]]) -> str:
        rendered_lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", "")).strip()
            rendered_lines.append(f"{role}:\n{content}")
        return "\n\n".join(rendered_lines).strip()

    @staticmethod
    def _extract_response_text(response: Any) -> str:
        output_text = getattr(response, "output_text", None)
        if output_text is not None:
            return str(output_text).strip()

        output = getattr(response, "output", None)
        if isinstance(output, list):
            chunks: list[str] = []
            for item in output:
                content = getattr(item, "content", None)
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    text = getattr(content_item, "text", None)
                    if text:
                        chunks.append(str(text))
            if chunks:
                return "".join(chunks).strip()

        return str(response).strip()

    def _generate_chat_completion(self, client, messages: list[dict[str, str]]) -> dict[str, Any]:
        if self.config.debug_stream:
            print(
                f"[NVIDIA REQUEST START] model={self.config.model_id} "
                f"api_model={self.config.api_model_name} stream={self.config.stream} "
                f"timeout={self.config.timeout_seconds} "
                f"job_timeout={self.config.job_timeout_seconds}",
                flush=True,
            )

        completion = client.chat.completions.create(
            model=self.config.api_model_name,
            messages=messages,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            extra_body=self.config.extra_body,
            stream=self.config.stream,
        )

        if not self.config.stream:
            if self.config.debug_stream:
                print(
                    f"[NVIDIA REQUEST DONE] model={self.config.model_id} stream=False",
                    flush=True,
                )
            return {
                "raw_text": self._extract_raw_text(completion),
                "reasoning_text": "",
                "usage": self._extract_usage(completion),
                "finish_reason": self._extract_finish_reason(completion),
                "stop_reason": self._extract_finish_reason(completion),
                "reached_max_tokens": self._extract_finish_reason(completion) == "length",
                "stream_complete": True,
                "stream_error": None,
                "partial_output": False,
                "timed_out_by_job_limit": False,
            }

        content_chunks: list[str] = []
        reasoning_chunks: list[str] = []
        stream_complete = True
        stream_error: str | None = None
        partial_output = False
        timed_out_by_job_limit = False
        finish_reason: str | None = None
        stream_start = perf_counter()
        last_debug_print = stream_start
        chunk_count = 0

        if self.config.debug_stream:
            print(
                f"[NVIDIA STREAM START] model={self.config.model_id} "
                f"max_tokens={self.config.max_tokens}",
                flush=True,
            )

        try:
            for chunk in completion:
                chunk_count += 1
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    chunk_finish_reason = getattr(choices[0], "finish_reason", None)
                    if chunk_finish_reason is not None:
                        finish_reason = str(chunk_finish_reason)

                    delta = getattr(choices[0], "delta", None)
                    if delta is not None:
                        reasoning = getattr(delta, "reasoning_content", None)
                        if reasoning:
                            reasoning_chunks.append(str(reasoning))

                        content = getattr(delta, "content", None)
                        if content is not None:
                            content_chunks.append(str(content))

                elapsed_seconds = perf_counter() - stream_start
                if (
                    self.config.debug_stream
                    and elapsed_seconds - last_debug_print
                    >= self.config.debug_stream_interval_seconds
                ):
                    last_debug_print = perf_counter()
                    print(
                        f"[NVIDIA STREAM PROGRESS] model={self.config.model_id} "
                        f"elapsed={elapsed_seconds:.1f}s chunks={chunk_count} "
                        f"content_chars={len(''.join(content_chunks))} "
                        f"reasoning_chars={len(''.join(reasoning_chunks))}",
                        flush=True,
                    )

                if self.config.job_timeout_seconds is not None:
                    if elapsed_seconds > self.config.job_timeout_seconds:
                        stream_complete = False
                        stream_error = (
                            "JobTimeout: generation exceeded "
                            f"{self.config.job_timeout_seconds} seconds"
                        )
                        partial_output = True
                        timed_out_by_job_limit = True
                        if self.config.debug_stream:
                            print(
                                f"[NVIDIA STREAM TIMEOUT] model={self.config.model_id} "
                                f"elapsed={elapsed_seconds:.1f}s "
                                f"limit={self.config.job_timeout_seconds}s "
                                f"content_chars={len(''.join(content_chunks))}",
                                flush=True,
                            )
                        break
        except Exception as exc:
            stream_complete = False
            stream_error = f"{type(exc).__name__}: {exc}"
            partial_output = True
            if self.config.debug_stream:
                print(
                    f"[NVIDIA STREAM ERROR] model={self.config.model_id} "
                    f"{stream_error} content_chars={len(''.join(content_chunks))} "
                    f"reasoning_chars={len(''.join(reasoning_chunks))}",
                    flush=True,
                )

        if self.config.debug_stream:
            elapsed_seconds = perf_counter() - stream_start
            print(
                f"[NVIDIA STREAM DONE] model={self.config.model_id} "
                f"complete={stream_complete} elapsed={elapsed_seconds:.1f}s "
                f"chunks={chunk_count} content_chars={len(''.join(content_chunks))} "
                f"reasoning_chars={len(''.join(reasoning_chunks))}",
                flush=True,
            )

        return {
            "raw_text": "".join(content_chunks).strip(),
            "reasoning_text": "".join(reasoning_chunks).strip(),
            "usage": {
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
            },
            "finish_reason": finish_reason,
            "stop_reason": finish_reason,
            "reached_max_tokens": finish_reason == "length",
            "stream_complete": stream_complete,
            "stream_error": stream_error,
            "partial_output": partial_output,
            "timed_out_by_job_limit": timed_out_by_job_limit,
        }

    def _generate_response(self, client, messages: list[dict[str, str]]) -> dict[str, Any]:
        response = client.responses.create(
            model=self.config.api_model_name,
            input=self._render_messages_as_input(messages),
            max_output_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
            temperature=self.config.temperature,
            stream=self.config.stream,
        )

        if self.config.stream:
            chunks = [str(chunk) for chunk in response]
            raw_text = "".join(chunks).strip()
        else:
            raw_text = self._extract_response_text(response)

        return {
            "raw_text": raw_text,
            "reasoning_text": "",
            "usage": self._extract_usage(response),
            "finish_reason": None,
            "stop_reason": None,
            "reached_max_tokens": False,
            "stream_complete": True,
            "stream_error": None,
            "partial_output": False,
            "timed_out_by_job_limit": False,
        }

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Run one NVIDIA API generation and return a normalized payload."""
        client = self._get_client()

        start_time = perf_counter()
        if self.config.api_mode == "responses":
            generation = self._generate_response(client, messages)
        else:
            generation = self._generate_chat_completion(client, messages)
        latency_s = perf_counter() - start_time

        notes: list[str] = []
        if generation.get("reasoning_text"):
            notes.append("reasoning_content captured separately from raw_text.")
        if generation.get("partial_output"):
            notes.append("stream interrupted; raw_text contains partial output.")
        if generation.get("timed_out_by_job_limit"):
            notes.append("generation stopped by job timeout.")
        if generation.get("reached_max_tokens"):
            notes.append("generation likely stopped because it reached the token limit.")

        return {
            "model_id": self.config.model_id,
            "raw_text": generation["raw_text"],
            "usage": generation["usage"],
            "latency_s": latency_s,
            "notes": notes,
            "reasoning_text": generation["reasoning_text"],
            "finish_reason": generation.get("finish_reason"),
            "stop_reason": generation.get("stop_reason"),
            "reached_max_tokens": generation.get("reached_max_tokens", False),
            "stream_complete": generation.get("stream_complete"),
            "stream_error": generation.get("stream_error"),
            "partial_output": generation.get("partial_output", False),
            "timed_out_by_job_limit": generation.get("timed_out_by_job_limit", False),
        }
