"""Local Hugging Face adapter for benchmark runs.

This adapter accepts either a local model directory or a Hugging Face repo id.
For HPC runs, prepare models first with `scripts/prepare_models.py`; when
`weights_path` is a Hugging Face repo id, the adapter checks `models_cache`
before falling back to the Hub. The standard `HF_HUB_OFFLINE` and
`TRANSFORMERS_OFFLINE` environment variables are honored by the underlying
Hugging Face libraries.
"""

import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from time import perf_counter
from collections.abc import Iterable
from typing import Any


HF_REPO_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)


def looks_like_hf_repo_id(value: str) -> bool:
    """Return whether a value has the shape of a Hugging Face repo id."""
    text = value.strip()
    if not text or "\\" in text or text.startswith(("/", ".")):
        return False
    if ":" in text or " " in text:
        return False
    return HF_REPO_ID_PATTERN.match(text) is not None


@dataclass(slots=True)
class HFLocalConfig:
    model_id: str
    weights_path: str = ""
    temperature: float = 0.0
    top_k: int | None = 10
    top_p: float | None = None
    max_tokens: int = 4096
    device_map: str | None = "auto"
    torch_dtype: str | None = "auto"
    trust_remote_code: bool = False
    use_chat_template: bool = True
    add_generation_prompt: bool = True
    thinking_key: str = ""
    thinking_enabled: bool = False


class HFLocalAdapter:
    """Adapter used by the benchmark runner for local HF generation."""

    def __init__(self, config: HFLocalConfig):
        self.config = config
        self.model: Any | None = None
        self.tokenizer: Any | None = None

    def _load_backend(self):
        """Import the HF backend lazily so tests can run without the deps."""
        try:
            torch = import_module("torch")
            transformers = import_module("transformers")
            AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
            AutoTokenizer = getattr(transformers, "AutoTokenizer")
        except ImportError as exc:  # pragma: no cover - depends on local env
            raise RuntimeError(
                "HFLocalAdapter requires `torch` and `transformers` to be installed."
            ) from exc

        return torch, AutoModelForCausalLM, AutoTokenizer

    def _framework_root(self) -> Path:
        """Return the Benchmark Framework root directory."""
        return Path(__file__).resolve().parents[2]

    def _models_cache_dir(self) -> Path:
        """Return the project-managed prepared model cache directory."""
        return self._framework_root() / "models_cache"

    def _candidate_local_paths(self, path_text: str) -> list[Path]:
        """Return plausible local paths for a registry path value."""
        raw_path = Path(path_text).expanduser()
        if raw_path.is_absolute():
            return [raw_path]

        return [
            Path.cwd() / raw_path,
            self._framework_root() / raw_path,
        ]

    def _existing_local_model_path(self, path_text: str) -> Path | None:
        """Return an existing local model path for a registry path, if any."""
        if not path_text.strip():
            return None

        for candidate in self._candidate_local_paths(path_text):
            if candidate.exists() and candidate.is_dir():
                return candidate.resolve()
        return None

    def _prepared_model_path_for_repo(self, repo_id: str) -> Path:
        """Return the models_cache directory used by prepare_models.py."""
        return self._models_cache_dir() / repo_id.replace("/", "__")

    def _existing_prepared_model_path(self, repo_id: str) -> Path | None:
        """Return a prepared models_cache path for a repo id, if present."""
        candidate = self._prepared_model_path_for_repo(repo_id)
        if candidate.exists() and candidate.is_dir() and any(candidate.iterdir()):
            return candidate.resolve()
        return None

    def _resolve_model_source(self) -> str:
        """Return the local path or repo id passed to `from_pretrained(...)`.

        Resolution order:
        1. explicit existing local path from `weights_path` or `model_id`
        2. prepared `models_cache/<namespace>__<repo>` for HF repo ids
        3. original repo id/path, allowing Hugging Face to download or use its cache
        """
        model_source = self.config.weights_path.strip() or self.config.model_id.strip()

        local_path = self._existing_local_model_path(model_source)
        if local_path is not None:
            return str(local_path)

        if looks_like_hf_repo_id(model_source):
            prepared_path = self._existing_prepared_model_path(model_source)
            if prepared_path is not None:
                return str(prepared_path)

        return model_source

    def _resolve_torch_dtype(self, torch_module):
        """Translate the config string into a torch dtype when possible."""
        raw_dtype = (self.config.torch_dtype or "auto").strip().lower()
        if raw_dtype in {"", "auto"}:
            return "auto"

        dtype_map = {
            "float16": getattr(torch_module, "float16", None),
            "fp16": getattr(torch_module, "float16", None),
            "bfloat16": getattr(torch_module, "bfloat16", None),
            "bf16": getattr(torch_module, "bfloat16", None),
            "float32": getattr(torch_module, "float32", None),
            "fp32": getattr(torch_module, "float32", None),
        }
        resolved = dtype_map.get(raw_dtype)
        if resolved is None:
            raise ValueError(f"Unsupported torch_dtype value: {self.config.torch_dtype!r}")
        return resolved

    def _build_prompt_text(self, messages: list[dict[str, str]]) -> str:
        """Format chat messages for causal generation."""
        if (
            self.config.use_chat_template
            and self.tokenizer is not None
            and hasattr(self.tokenizer, "apply_chat_template")
        ):
            chat_template_kwargs: dict[str, Any] = {}
            thinking_key = self.config.thinking_key.strip()
            if thinking_key:
                chat_template_kwargs[thinking_key] = self.config.thinking_enabled

            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=self.config.add_generation_prompt,
                    **chat_template_kwargs,
                )
            except Exception:
                if chat_template_kwargs:
                    try:
                        return self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=self.config.add_generation_prompt,
                        )
                    except Exception:
                        pass

        rendered_lines: list[str] = []
        for message in messages:
            role = str(message.get("role", "user")).upper()
            content = str(message.get("content", "")).strip()
            rendered_lines.append(f"{role}:\n{content}")

        if self.config.add_generation_prompt:
            rendered_lines.append("ASSISTANT:\n")

        return "\n\n".join(rendered_lines).strip()

    def _get_model_device(self):
        """Best-effort inference device detection for encoded inputs."""
        if self.model is None:
            return None

        device = getattr(self.model, "device", None)
        if device is not None:
            return device

        parameters = getattr(self.model, "parameters", None)
        if callable(parameters):
            try:
                parameter_values = parameters()
            except Exception:
                return None
            if not isinstance(parameter_values, Iterable):
                return None
            first_param = next(iter(parameter_values), None)
            if first_param is None:
                return None
            return getattr(first_param, "device", None)

        return None

    @staticmethod
    def _move_inputs_to_device(encoded_inputs: dict[str, Any], device) -> dict[str, Any]:
        """Move tensor-like values to the model device when supported."""
        if device is None:
            return encoded_inputs

        moved_inputs: dict[str, Any] = {}
        for key, value in encoded_inputs.items():
            if hasattr(value, "to"):
                try:
                    moved_inputs[key] = value.to(device)
                    continue
                except Exception:
                    pass
            moved_inputs[key] = value
        return moved_inputs

    @staticmethod
    def _count_tokens(token_like: Any) -> int | None:
        """Count tokens for common list- and tensor-like structures."""
        shape = getattr(token_like, "shape", None)
        if shape is not None:
            try:
                if len(shape) >= 2:
                    return int(shape[-1])
                if len(shape) == 1:
                    return int(shape[0])
            except Exception:
                pass

        if isinstance(token_like, list):
            if token_like and isinstance(token_like[0], list):
                return len(token_like[0])
            return len(token_like)

        return None

    @staticmethod
    def _extract_generated_tokens(generation_output: Any, prompt_token_count: int) -> Any:
        """Return only the newly generated tokens from one generation output."""
        sequence = generation_output
        if isinstance(sequence, list) and sequence and isinstance(sequence[0], list):
            sequence = sequence[0]
        elif hasattr(sequence, "__getitem__"):
            try:
                sequence = sequence[0]
            except Exception:
                pass

        try:
            return sequence[prompt_token_count:]
        except Exception:
            return sequence

    def load_model(self) -> None:
        """Load the local model and tokenizer on first use."""
        if self.model is not None and self.tokenizer is not None:
            return

        torch, AutoModelForCausalLM, AutoTokenizer = self._load_backend()
        model_source = self._resolve_model_source()
        resolved_dtype = self._resolve_torch_dtype(torch)

        tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            trust_remote_code=self.config.trust_remote_code,
        )

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": self.config.trust_remote_code,
        }
        if self.config.device_map is not None:
            model_kwargs["device_map"] = self.config.device_map
        if resolved_dtype != "auto":
            model_kwargs["torch_dtype"] = resolved_dtype

        model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

        if self.config.device_map is None:
            target_device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
            if hasattr(model, "to"):
                model = model.to(target_device)

        if hasattr(model, "eval"):
            model.eval()

        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Run one local generation and return a normalized payload."""
        self.load_model()
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("HFLocalAdapter failed to initialize model or tokenizer.")

        model = self.model
        tokenizer = self.tokenizer

        prompt_text = self._build_prompt_text(messages)
        encoded_inputs = tokenizer(prompt_text, return_tensors="pt")
        prompt_token_count = self._count_tokens(encoded_inputs.get("input_ids")) or 0

        model_device = self._get_model_device()
        encoded_inputs = self._move_inputs_to_device(encoded_inputs, model_device)

        do_sample = self.config.temperature > 0.0
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_tokens,
            "do_sample": do_sample,
        }

        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if pad_token_id is not None:
            generation_kwargs["pad_token_id"] = pad_token_id
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id

        if do_sample:
            generation_kwargs["temperature"] = self.config.temperature
            if self.config.top_k is not None:
                generation_kwargs["top_k"] = self.config.top_k
            if self.config.top_p is not None:
                generation_kwargs["top_p"] = self.config.top_p

        torch_module, _, _ = self._load_backend()
        start_time = perf_counter()
        with torch_module.inference_mode():
            generation_output = model.generate(**encoded_inputs, **generation_kwargs)
        latency_s = perf_counter() - start_time

        generated_tokens = self._extract_generated_tokens(generation_output, prompt_token_count)
        raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        completion_tokens = self._count_tokens(generated_tokens)

        usage = {
            "prompt_tokens": prompt_token_count,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_token_count + completion_tokens
                if completion_tokens is not None
                else None
            ),
        }

        return {
            "model_id": self.config.model_id,
            "raw_text": raw_text,
            "usage": usage,
            "latency_s": latency_s,
            "notes": [],
        }
