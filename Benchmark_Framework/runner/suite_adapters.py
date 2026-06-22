"""Model adapter construction for benchmark suites."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable


class _UnavailableAdapter:
    """Minimal adapter used when a configured adapter cannot be created."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "raw_text": "",
            "usage": {},
            "latency_s": None,
            "notes": [
                "No concrete adapter factory was provided.",
                "The configured adapter could not be initialized for this run.",
            ],
            "message_count": len(messages),
        }


@lru_cache(maxsize=None)
def _load_framework_module(module_key: str, relative_path: str):
    """Load a framework module without requiring package installation."""
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_optional_int(value: Any, field_name: str) -> int | None:
    """Parse optional integer config fields from registry entries."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {field_name}: {value!r}") from exc


def build_model_adapter(
    model_entry: dict[str, Any],
    protocol_config: Any,
    adapter_factory: Callable[[dict[str, Any], Any], Any] | None = None,
):
    """Build the adapter used by one suite job."""
    if adapter_factory is not None:
        return adapter_factory(model_entry, protocol_config)

    adapter_name = str(model_entry.get("adapter", "")).strip()
    model_id = str(model_entry.get("model_id", "unknown-model"))

    if adapter_name == "hf_local":
        hf_module = _load_framework_module(
            "benchmark_framework_hf_local_adapter",
            "models/adapters/hf_local.py",
        )
        generation_config = protocol_config.raw_config.get("generation", {})
        if not isinstance(generation_config, dict):
            generation_config = {}

        top_k = generation_config.get("top_k", 10)
        if top_k is None:
            top_k = 10
        top_p = model_entry.get("top_p", generation_config.get("top_p"))
        if top_p in {"", "none", "null"}:
            top_p = None

        hf_config = hf_module.HFLocalConfig(
            model_id=model_id,
            weights_path=str(model_entry.get("weights_path", "")),
            model_loader=str(model_entry.get("model_loader", "causal_lm") or "causal_lm"),
            temperature=float(model_entry.get("temperature", generation_config.get("temperature", 0.0)) or 0.0),
            top_k=int(top_k),
            top_p=float(top_p) if top_p is not None else None,
            max_tokens=int(model_entry.get("max_tokens", generation_config.get("max_tokens", 4096)) or 4096),
            device_map=None if model_entry.get("device_map") in {None, "", "none"} else str(model_entry.get("device_map", "auto")),
            torch_dtype=None if model_entry.get("torch_dtype") in {None, ""} else str(model_entry.get("torch_dtype", "auto")),
            trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
            use_chat_template=bool(model_entry.get("use_chat_template", True)),
            add_generation_prompt=bool(model_entry.get("add_generation_prompt", True)),
            thinking_key=str(model_entry.get("thinking_key", "") or ""),
            thinking_enabled=bool(model_entry.get("thinking_enabled", False)),
        )
        return hf_module.HFLocalAdapter(hf_config)

    if adapter_name == "ollama":
        ollama_module = _load_framework_module(
            "benchmark_framework_ollama_adapter",
            "models/adapters/ollama.py",
        )
        generation_config = protocol_config.raw_config.get("generation", {})
        if not isinstance(generation_config, dict):
            generation_config = {}

        top_k = generation_config.get("top_k", 10)
        if top_k is None:
            top_k = 10

        ollama_config = ollama_module.OllamaConfig(
            model_id=model_id,
            ollama_model=str(model_entry.get("ollama_model", model_entry.get("api_model_name", ""))),
            base_url=str(model_entry.get("base_url", "http://localhost:11434")),
            temperature=float(generation_config.get("temperature", 0.0) or 0.0),
            top_k=int(top_k),
            max_tokens=int(generation_config.get("max_tokens", 4096) or 4096),
            timeout_seconds=int(model_entry.get("timeout_seconds", 300) or 300),
        )
        return ollama_module.OllamaAdapter(ollama_config)

    if adapter_name == "nvidia_api":
        nvidia_module = _load_framework_module(
            "benchmark_framework_nvidia_api_adapter",
            "models/adapters/nvidia_api.py",
        )
        generation_config = protocol_config.raw_config.get("generation", {})
        if not isinstance(generation_config, dict):
            generation_config = {}

        extra_body: dict[str, Any] = {}
        thinking_key = str(model_entry.get("thinking_key", "") or "").strip()
        if thinking_key:
            extra_body["chat_template_kwargs"] = {
                thinking_key: bool(model_entry.get("thinking_enabled", False)),
            }
        reasoning_budget = _parse_optional_int(
            model_entry.get("reasoning_budget"),
            "reasoning_budget",
        )
        if reasoning_budget is not None:
            extra_body["reasoning_budget"] = reasoning_budget

        nvidia_config = nvidia_module.NvidiaAPIConfig(
            model_id=model_id,
            api_model_name=str(model_entry.get("api_model_name", model_id)),
            base_url=str(model_entry.get("base_url", "https://integrate.api.nvidia.com/v1")),
            api_key_env=str(model_entry.get("api_key_env", "NVIDIA_API_KEY")),
            api_mode=str(model_entry.get("api_mode", "chat_completions")),
            stream=bool(model_entry.get("stream", False)),
            temperature=float(model_entry.get("temperature", generation_config.get("temperature", 0.0)) or 0.0),
            top_p=float(model_entry.get("top_p", generation_config.get("top_p", 0.95)) or 0.95),
            max_tokens=int(model_entry.get("max_tokens", generation_config.get("max_tokens", 4096)) or 4096),
            timeout_seconds=int(model_entry.get("timeout_seconds", 300) or 300),
            job_timeout_seconds=_parse_optional_int(
                model_entry.get("job_timeout_seconds"),
                "job_timeout_seconds",
            ),
            debug_stream=bool(model_entry.get("debug_stream", False)),
            debug_stream_interval_seconds=int(
                model_entry.get("debug_stream_interval_seconds", 10) or 10
            ),
            extra_body=extra_body,
        )
        return nvidia_module.NvidiaAPIAdapter(nvidia_config)

    if adapter_name == "llama_cpp_cli":
        llama_cpp_module = _load_framework_module(
            "benchmark_framework_llama_cpp_cli_adapter",
            "models/adapters/llama_cpp_cli.py",
        )
        generation_config = protocol_config.raw_config.get("generation", {})
        if not isinstance(generation_config, dict):
            generation_config = {}

        top_k = generation_config.get("top_k", 10)
        if top_k is None:
            top_k = 10

        llama_cpp_config = llama_cpp_module.LlamaCppCLIConfig(
            model_id=model_id,
            executable_path=str(model_entry.get("executable_path", "llama-cli")),
            model_path=str(model_entry.get("model_path", model_entry.get("weights_path", ""))),
            temperature=float(generation_config.get("temperature", 0.0) or 0.0),
            top_k=int(top_k),
            top_p=float(generation_config.get("top_p", model_entry.get("top_p", 0.95)) or 0.95),
            max_tokens=int(generation_config.get("max_tokens", 4096) or 4096),
            context_size=_parse_optional_int(model_entry.get("context_size"), "context_size"),
            threads=_parse_optional_int(model_entry.get("threads"), "threads"),
            timeout_seconds=int(model_entry.get("timeout_seconds", 300) or 300),
            extra_args=[],
        )
        return llama_cpp_module.LlamaCppCLIAdapter(llama_cpp_config)

    return _UnavailableAdapter(model_id)
