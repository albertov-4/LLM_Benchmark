"""Configuration and prompt loading helpers for benchmark suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LoadedProtocolConfig:
    protocol_id: str
    max_iterations: int
    require_final_plan_only: bool
    raw_config: dict[str, Any]


@dataclass(slots=True)
class LoadedPromptBundle:
    system_prompt: str
    domain_prompt: str
    feedback_prompt: str


def load_protocol_config(
    protocol_id: str,
    protocols_root: str | Path,
) -> LoadedProtocolConfig:
    """Load protocol metadata needed by the runner."""
    protocol_path = Path(protocols_root) / f"{protocol_id}.yaml"
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol config not found: {protocol_path}")

    loaded = yaml.safe_load(protocol_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        loaded = {}

    evaluation = loaded.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}

    return LoadedProtocolConfig(
        protocol_id=str(loaded.get("protocol_id", protocol_id)),
        max_iterations=int(evaluation.get("max_iterations", 1)),
        require_final_plan_only=bool(evaluation.get("require_final_plan_only", True)),
        raw_config=loaded,
    )


def _read_text_if_exists(file_path: Path) -> str:
    """Read a UTF-8 text file when present, otherwise return an empty string."""
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8").strip()


def _read_required_text(file_path: Path, description: str) -> str:
    """Read a required UTF-8 text file and fail clearly when it is missing or empty."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {file_path}. "
            "Create this prompt file or disable the related protocol option."
        )

    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(
            f"Empty {description}: {file_path}. "
            "Fill this prompt file or disable the related protocol option."
        )
    return content


def load_prompt_bundle(
    task_family: str,
    protocol_config: LoadedProtocolConfig,
    prompts_root: str | Path,
) -> LoadedPromptBundle:
    """Load the prompt texts used by one suite job."""
    prompts_root_path = Path(prompts_root)
    prompting_config = protocol_config.raw_config.get("prompting", {})
    if not isinstance(prompting_config, dict):
        prompting_config = {}

    use_system_prompt = bool(prompting_config.get("use_system_prompt", True))
    include_domain_prompt = bool(prompting_config.get("include_domain_prompt", True))
    include_examples = bool(prompting_config.get("include_examples", False))
    include_external_feedback = bool(prompting_config.get("include_external_feedback", False))

    system_prompt = ""
    if use_system_prompt:
        system_prompt = _read_text_if_exists(prompts_root_path / "system.txt")

    domain_prompt = ""
    if include_domain_prompt:
        domain_prompt = _read_required_text(
            prompts_root_path / f"{task_family}.txt",
            f"domain prompt for task family '{task_family}'",
        )

    if include_examples:
        example_prompt = _read_text_if_exists(prompts_root_path / "examples" / f"{task_family}.txt")
        if example_prompt:
            if domain_prompt:
                domain_prompt = f"{domain_prompt}\n\n=== EXAMPLES ===\n{example_prompt}"
            else:
                domain_prompt = f"=== EXAMPLES ===\n{example_prompt}"

    feedback_prompt = ""
    if include_external_feedback:
        feedback_prompt = _read_text_if_exists(prompts_root_path / "feedback.txt")

    return LoadedPromptBundle(
        system_prompt=system_prompt,
        domain_prompt=domain_prompt,
        feedback_prompt=feedback_prompt,
    )
