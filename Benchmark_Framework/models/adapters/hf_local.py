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
import sys


HF_REPO_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)
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


def _install_transformers_generation_compatibility_shim() -> None:
    """Restore legacy generation aliases expected by older mamba-ssm releases."""
    try:
        generation_module = import_module("transformers.generation")
    except Exception:
        return

    generate_decoder_only_output = getattr(generation_module, "GenerateDecoderOnlyOutput", None)
    if generate_decoder_only_output is None:
        return

    for legacy_name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        if not hasattr(generation_module, legacy_name):
            setattr(generation_module, legacy_name, generate_decoder_only_output)


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
    model_loader: str = "causal_lm"
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
            _install_transformers_generation_compatibility_shim()
            AutoModelForCausalLM = getattr(transformers, "AutoModelForCausalLM")
            AutoTokenizer = getattr(transformers, "AutoTokenizer")
            AutoModel = getattr(transformers, "AutoModel", None)
            AutoModelForImageTextToText = getattr(transformers, "AutoModelForImageTextToText", None)
            AutoProcessor = getattr(transformers, "AutoProcessor", None)
        except ImportError as exc:
            raise RuntimeError(
                "HFLocalAdapter backend import failed. "
                f"python={sys.executable}; "
                f"original_error={type(exc).__name__}: {exc}"
            ) from exc

        return torch, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForImageTextToText, AutoProcessor

    def _framework_root(self) -> Path:
        """Return the Benchmark_Framework root directory."""
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

    @staticmethod
    def _processor_messages(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
        """Return messages in the content-list format expected by multimodal processors."""
        processor_messages: list[dict[str, Any]] = []
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            processor_messages.append(
                {
                    "role": role,
                    "content": [{"type": "text", "text": content}],
                }
            )
        return processor_messages

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

    def _install_nemotron_cache_patch(self, model: Any, torch_module: Any) -> None:
        """Patch Nemotron-H generation to provide its required hybrid cache."""
        if "nemotron" not in self.config.model_id.lower():
            return
        if getattr(model, "_benchmark_nemotron_cache_patch_installed", False):
            return
        if not hasattr(model, "prepare_inputs_for_generation"):
            return

        try:
            model_module = import_module(model.__class__.__module__)
        except Exception:
            return

        cache_cls = getattr(model_module, "NemotronHHybridDynamicCache", None)
        if cache_cls is None:
            return

        original_prepare = model.prepare_inputs_for_generation

        def patched_prepare_inputs_for_generation(
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            **kwargs,
        ):
            use_cache = kwargs.get("use_cache", True)
            sequence_source = inputs_embeds if inputs_embeds is not None else input_ids
            sequence_shape = getattr(sequence_source, "shape", None)

            if past_key_values is None and use_cache is not False and sequence_shape is not None:
                dtype = getattr(model, "dtype", None)
                if dtype is None:
                    try:
                        dtype = next(model.parameters()).dtype
                    except Exception:
                        dtype = torch_module.bfloat16

                device = getattr(sequence_source, "device", None) or getattr(model, "device", None)
                batch_size = int(sequence_shape[0])
                past_key_values = cache_cls(
                    model.config,
                    batch_size,
                    dtype=dtype,
                    device=device,
                )

            if cache_position is None and sequence_shape is not None:
                sequence_length = int(sequence_shape[1])
                past_length = 0
                if past_key_values is not None and hasattr(past_key_values, "get_seq_length"):
                    try:
                        past_length = int(past_key_values.get_seq_length())
                    except Exception:
                        past_length = 0
                device = getattr(sequence_source, "device", None) or getattr(model, "device", None)
                cache_position = torch_module.arange(
                    past_length,
                    past_length + sequence_length,
                    dtype=torch_module.long,
                    device=device,
                )

            return original_prepare(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                **kwargs,
            )

        model.prepare_inputs_for_generation = patched_prepare_inputs_for_generation
        model._benchmark_nemotron_cache_patch_installed = True

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

        torch, AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForImageTextToText, AutoProcessor = self._load_backend()
        model_source = self._resolve_model_source()
        resolved_dtype = self._resolve_torch_dtype(torch)
        model_loader = self.config.model_loader.strip().lower()

        if model_loader == "image_text_to_text":
            if AutoModelForImageTextToText is None or AutoProcessor is None:
                raise RuntimeError(
                    "This model requires AutoModelForImageTextToText and AutoProcessor. "
                    "Upgrade transformers."
                )
            tokenizer = AutoProcessor.from_pretrained(
                model_source,
                trust_remote_code=self.config.trust_remote_code,
            )
        else:
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

        if model_loader == "image_text_to_text":
            model = AutoModelForImageTextToText.from_pretrained(model_source, **model_kwargs)
        elif model_loader == "auto_model":
            if AutoModel is None:
                raise RuntimeError("This model requires AutoModel. Upgrade transformers.")
            model = AutoModel.from_pretrained(model_source, **model_kwargs)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_source, **model_kwargs)

        if self.config.device_map is None:
            target_device = "cuda" if getattr(torch, "cuda", None) and torch.cuda.is_available() else "cpu"
            if hasattr(model, "to"):
                model = model.to(target_device)

        if hasattr(model, "eval"):
            model.eval()

        self._install_nemotron_cache_patch(model, torch)

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

        model_loader = self.config.model_loader.strip().lower()
        if model_loader == "image_text_to_text" and hasattr(tokenizer, "apply_chat_template"):
            processor_messages = self._processor_messages(messages)
            encoded_inputs = tokenizer.apply_chat_template(
                processor_messages,
                tokenize=True,
                add_generation_prompt=self.config.add_generation_prompt,
                return_dict=True,
                return_tensors="pt",
            )
        else:
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

        tokenizer_like = getattr(tokenizer, "tokenizer", tokenizer)
        pad_token_id = getattr(tokenizer_like, "pad_token_id", None)
        eos_token_id = getattr(tokenizer_like, "eos_token_id", None)
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

        torch_module, _, _, _, _, _ = self._load_backend()
        start_time = perf_counter()
        with torch_module.inference_mode():
            generation_output = model.generate(**encoded_inputs, **generation_kwargs)
        latency_s = perf_counter() - start_time

        generated_tokens = self._extract_generated_tokens(generation_output, prompt_token_count)
        raw_text = tokenizer_like.decode(generated_tokens, skip_special_tokens=True).strip()
        extracted = _extract_reasoning_from_text(raw_text)
        raw_text = extracted["raw_text"]
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
            "reasoning_text": extracted["reasoning_text"],
            "usage": usage,
            "latency_s": latency_s,
            "notes": extracted["notes"],
        }
