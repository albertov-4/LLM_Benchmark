"""Model loading and response generation utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, logging as hf_logging

from prompts.prompts import system_prompt_pddl
from utils.answer_postprocessor import formatter, clean_response_text
from utils.logging_utils import get_logger
from utils.validator import validate_plan_from_text

hf_logging.set_verbosity_warning()

logger = get_logger(__name__)


class ModelManager:
    """Wrapper around Hugging Face models tailored for PDDL planning."""

    def __init__(self, weights_path: str):
        self.weights_path = weights_path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.model_type = self._detect_model_type(weights_path)
        logger.info("Detected model type: %s", self.model_type)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> Tuple[Any, Any]:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Using %s backend", "GPU" if torch.cuda.is_available() else "CPU")

        try:
            self._log_gpu_inventory()
            self.model = self._load_model()
            self.tokenizer = self._load_tokenizer()
            self._ensure_pad_token()
        except Exception as exc:  # pragma: no cover - fatal
            logger.exception("Unable to load model from %s", self.weights_path)
            raise SystemExit(1) from exc

        logger.info(
            "Model loaded: %s (~%.1fB params)",
            self.model_type,
            sum(p.numel() for p in self.model.parameters()) / 1e9,
        )
        return self.model, self.tokenizer

    def _load_model(self):
        kwargs = {
            "torch_dtype": torch.bfloat16,
            "trust_remote_code": True,
        }
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = {"": self.device}

        logger.debug("Loading model from %s", self.weights_path)
        return AutoModelForCausalLM.from_pretrained(self.weights_path, **kwargs)

    def _load_tokenizer(self):
        logger.debug("Loading tokenizer from %s", self.weights_path)
        tokenizer = AutoTokenizer.from_pretrained(self.weights_path)
        return tokenizer

    def _ensure_pad_token(self) -> None:
        if self.tokenizer and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.debug("Pad token missing – defaulting to eos token")

    def _log_gpu_inventory(self) -> None:
        if not torch.cuda.is_available():
            return
        gpu_count = torch.cuda.device_count()
        for idx in range(gpu_count):
            props = torch.cuda.get_device_properties(idx)
            logger.info(
                "GPU %d: %s (%.1f GB)", idx, props.name, props.total_memory / 1e9
            )

    # ------------------------------------------------------------------
    # Response generation
    # ------------------------------------------------------------------

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        sampling: bool,
        temperature: float,
        top_k: int,
        include_prompt: bool,
        skip_special_tokens: bool,
    ) -> str:
        if not self.model or not self.tokenizer:
            raise ValueError("ModelManager.load() must be called before generation")

        formatted = self._format_messages(messages)
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            truncation=True,
            max_length=min(getattr(self.tokenizer, "model_max_length", 4096), 4096),
        )

        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        config = self._build_generation_config(
            max_tokens=max_tokens,
            sampling=sampling,
            temperature=temperature,
            top_k=top_k,
        )

        try:
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **config)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                logger.error("CUDA OOM during generation")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                return "Generation failed: CUDA out of memory"
            logger.exception("Error during generation")
            return f"Generation failed: {exc}"

        sequence = outputs[0] if not isinstance(outputs, list) else outputs[0][0]
        if include_prompt:
            decoded_tokens = sequence
        else:
            decoded_tokens = sequence[inputs["input_ids"].shape[1] :]

        return self.tokenizer.decode(
            decoded_tokens, skip_special_tokens=skip_special_tokens
        ).strip()

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Minimal fallback for non-chat tokenizers
            output = []
            for message in messages:
                output.append(f"{message['role'].capitalize()}: {message['content']}")
            output.append("Assistant:")
            return "\n\n".join(output)

    def _build_generation_config(
        self,
        *,
        max_tokens: int,
        sampling: bool,
        temperature: float,
        top_k: int,
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "max_new_tokens": int(max_tokens),
            "do_sample": bool(sampling),
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
        }
        if sampling and temperature > 0:
            config.update({
                "temperature": float(temperature),
                "top_k": int(top_k),
                "top_p": 0.9,
            })
        return config

    # ------------------------------------------------------------------
    # Iterative planning loop
    # ------------------------------------------------------------------

    def iterative_planning_with_validation(
        self,
        *,
        domain_path: str,
        problem_path: str,
        initial_prompt: str,
        max_iterations: int = 3,
        add_system_prompt: bool = True,
        validation_feedback_fn: Optional[Callable[[str, str, str], str]] = None,
        **generation_kwargs,
    ) -> Tuple[str, int, bool]:
        messages = self._build_initial_messages(initial_prompt, add_system_prompt)
        defaults = {
            "max_tokens": 5000,
            "sampling": False,
            "temperature": 0.0,
            "top_k": 50,
            "include_prompt": False,
            "skip_special_tokens": True,
        }
        config = {**defaults, **generation_kwargs}
        # Force include_prompt to False for iterative planning to avoid context duplication
        # and ensure clean response extraction
        config["include_prompt"] = False

        last_response = ""

        for iteration in range(1, max_iterations + 1):
            logger.info(
                "Planning iteration %d/%d (%s / %s)",
                iteration,
                max_iterations,
                Path(domain_path).name,
                Path(problem_path).name,
            )

            response = self.generate_response(messages, **config)
            last_response = response
            logger.debug(f"Iteration {iteration}: generated response length: {len(response)}")
            formatted = formatter(response, include_reasoning=True)
            plan_actions = formatted.get("plan", [])
            logger.debug("Iteration %d: extracted %d plan actions", iteration, len(plan_actions))

            if not plan_actions:
                logger.warning("Iteration %d: no plan extracted", iteration)
                self._append_feedback(
                    messages,
                    assistant=response,
                    user=(
                        "The response doesn't contain a valid plan. "
                        "Please output a sequence of PDDL actions."
                    ),
                )
                continue

            plan_text = "\n".join(plan_actions)
            validation = validate_plan_from_text(domain_path, problem_path, plan_text)

            if validation.get("valid"):
                logger.info("Valid plan produced in %d iteration(s)", iteration)
                return plan_text, iteration, True

            error_msg = validation.get("error") or "Plan validation failed"
            feedback = self._build_validation_feedback(
                validation_feedback_fn, initial_prompt, plan_text, error_msg
            )
            logger.warning(
                "Invalid plan (iteration %d): %s", iteration, error_msg
            )
            self._append_feedback(messages, assistant=response, user=feedback)

        logger.error("No valid plan found after %d iterations", max_iterations)
        formatted = formatter(last_response, include_reasoning=True)
        plan_text = "\n".join(formatted.get("plan", []))
        return plan_text, max_iterations, False

    def _build_initial_messages(
        self, prompt: str, add_system_prompt: bool
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if add_system_prompt:
            messages.append({"role": "system", "content": system_prompt_pddl})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _append_feedback(
        self,
        messages: List[Dict[str, str]],
        *,
        assistant: str,
        user: str,
    ) -> None:
        # Clean the assistant response to remove long reasoning traces (e.g. <think> blocks)
        # This prevents the context from growing too large and confusing the model in the next turn
        cleaned_assistant = clean_response_text(assistant)
        if not cleaned_assistant:
            # If cleaning removed everything (e.g. only thoughts were present), use a placeholder
            cleaned_assistant = "No plan generated."
            
        messages.append({"role": "assistant", "content": cleaned_assistant})
        messages.append({"role": "user", "content": user})

    def _build_validation_feedback(
        self,
        feedback_fn: Optional[Callable[[str, str, str], str]],
        initial_prompt: str,
        plan_text: str,
        error_msg: str,
    ) -> str:
        if feedback_fn:
            try:
                return feedback_fn(initial_prompt, plan_text, error_msg)
            except Exception:
                logger.exception("Custom validation feedback failed")
        try:
            from prompts.prompts import validation_feedback_prompt

            return validation_feedback_prompt(initial_prompt, plan_text, error_msg)
        except Exception:
            return f"The plan is invalid. Error: {error_msg}. Please provide a corrected plan."

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _detect_model_type(self, weights_path: str) -> str:
        name = Path(weights_path).name.lower()
        if "phi" in name:
            return "phi4"
        if "llama" in name:
            return "llama3"
        if "gemma" in name:
            return "gemma3"
        if "kimi" in name:
            return "kimi"
        return "unknown"

    def get_model_info(self) -> Dict[str, Any]:
        if not self.model:
            return {"loaded": False, "weights_path": self.weights_path}
        params = sum(p.numel() for p in self.model.parameters()) / 1e9
        return {
            "loaded": True,
            "model_type": self.model_type,
            "weights_path": self.weights_path,
            "device": str(self.device),
            "parameters_b": f"~{params:.1f}B",
            "torch_dtype": getattr(self.model, "dtype", "unknown"),
            "vocab_size": getattr(self.tokenizer, "vocab_size", "unknown"),
        }


__all__ = ["ModelManager"]