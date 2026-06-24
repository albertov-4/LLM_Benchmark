"""Repair feedback generation for iterative PDDL planning attempts."""

from __future__ import annotations

import sys
from dataclasses import dataclass, fields
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

try:
    from evaluators.plan_alignment import compute_sequence_alignment
except ModuleNotFoundError:  # Loaded by file path from runner/run_case.py.
    module_path = Path(__file__).resolve().parent / "plan_alignment.py"
    spec = spec_from_file_location("benchmark_framework_plan_alignment", module_path)
    if spec is None or spec.loader is None:
        raise
    module = module_from_spec(spec)
    sys.modules["benchmark_framework_plan_alignment"] = module
    spec.loader.exec_module(module)
    compute_sequence_alignment = module.compute_sequence_alignment


@dataclass(slots=True)
class RepairFeedbackConfig:
    history_mode: str = "last_only"
    include_previous_raw_text: bool = True
    include_previous_reasoning_text: bool = True
    max_raw_chars: int = 4000
    max_reasoning_chars: int = 6000
    high_alignment_threshold: float = 0.85
    low_alignment_threshold: float = 0.55
    final_output_instruction: str = (
        "Return only the complete PDDL action sequence, one parenthesized action per line, "
        "with no explanation, headings, markdown, bullets, numbering, or comments."
    )

    @classmethod
    def from_mapping(cls, values: Any) -> "RepairFeedbackConfig":
        if isinstance(values, cls):
            return values
        if not isinstance(values, dict):
            return cls()
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


def build_repair_feedback(
    *,
    attempt_record: dict[str, Any],
    feedback_prompt: str = "",
    config: RepairFeedbackConfig | dict[str, Any] | None = None,
) -> str:
    cfg = RepairFeedbackConfig.from_mapping(config)
    generation = _dict(attempt_record.get("generation"))
    raw_text = _text(generation.get("raw_text", attempt_record.get("raw_output", "")))
    reasoning_text = _text(generation.get("reasoning_text", ""))

    sections: list[str] = []
    if cfg.include_previous_raw_text:
        sections.extend(["[PREVIOUS FINAL ANSWER]", _truncate(raw_text, cfg.max_raw_chars) or "[empty]"])
    if cfg.include_previous_reasoning_text and reasoning_text.strip():
        sections.extend(["[PREVIOUS REASONING TEXT]", _truncate(reasoning_text, cfg.max_reasoning_chars)])

    feedback = _diagnostic_text(attempt_record, cfg)
    if feedback_prompt.strip():
        feedback = f"{feedback_prompt.strip()} {feedback}"
    sections.extend(["[FEEDBACK FOR THE NEXT ATTEMPT]", feedback])
    return "\n\n".join(sections)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _truncate(text: str, max_chars: int) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    marker = "\n[truncated]\n"
    if max_chars <= len(marker):
        return "[truncated]"
    keep = max_chars - len(marker)
    head = keep // 2
    tail = keep - head
    return f"{text[:head].rstrip()}{marker}{text[-tail:].lstrip()}"


def _actions(parsed_plan: dict[str, Any], key: str) -> list[Any]:
    section = parsed_plan.get(key)
    if isinstance(section, dict) and isinstance(section.get("actions"), list):
        return section["actions"]
    return []


def _format_issues(attempt_record: dict[str, Any]) -> list[str]:
    parsed_plan = _dict(attempt_record.get("parsed_plan"))
    raw = _dict(parsed_plan.get("raw"))
    result = _dict(attempt_record.get("validation_result"))
    details = _dict(result.get("details"))
    issues = []
    for source in (raw.get("format_issues"), details.get("raw_format_issues")):
        if isinstance(source, list):
            issues.extend(str(item) for item in source if item)
    return list(dict.fromkeys(issues))


def _validation_sentence(result: dict[str, Any]) -> str:
    feedback = _text(result.get("feedback_text")).strip()
    error_type = _text(result.get("error_type") or "unknown").strip()
    failed_step = result.get("failed_step")
    failed_action = _text(result.get("failed_action")).strip()
    parts = [feedback or "The validator rejected the previous final answer."]
    if error_type and error_type != "unknown":
        parts.append(f"The validator classified the failure as {error_type}.")
    if failed_step is not None:
        parts.append(f"It reported the failure at step {failed_step}.")
    if failed_action:
        parts.append(f"The problematic action was {failed_action}.")
    return " ".join(parts)


def _diagnostic_text(attempt_record: dict[str, Any], cfg: RepairFeedbackConfig) -> str:
    result = _dict(attempt_record.get("validation_result"))
    parsed_plan = _dict(attempt_record.get("parsed_plan"))
    raw_actions = _actions(parsed_plan, "raw")
    reasoning_actions = _actions(parsed_plan, "reasoning")
    reasoning_result = _dict(attempt_record.get("reasoning_validation_result"))
    raw_valid = bool(result.get("valid"))
    reasoning_valid = bool(attempt_record.get("reasoning_final_plan_valid") or reasoning_result.get("valid"))
    final_plan_valid = bool(attempt_record.get("final_plan_valid"))
    error_type = _text(result.get("error_type"))
    status = _text(result.get("status"))
    issues = _format_issues(attempt_record)
    validation = _validation_sentence(result)
    alignment = compute_sequence_alignment(raw_actions, reasoning_actions) if raw_actions and reasoning_actions else {}

    technical = status in {"generation_error", "validator_error", "timeout"} or error_type in {
        "validator_crash",
        "validator_unavailable",
        "timeout",
    }
    format_failure = status == "parse_error" or not raw_actions or any(
        issue in {
            "empty_output",
            "no_valid_domain_actions_found",
            "no_parenthesized_actions_found",
            "raw_text_contains_reasoning_like_content",
            "reasoning_before_plan_removed",
        }
        for issue in issues
    )

    if technical:
        core = (
            f"The previous attempt failed because of a technical generation or validation problem, not a confirmed planning mistake. "
            f"{validation} For the next attempt, produce a clean complete final action sequence so it can be checked again."
        )
    elif reasoning_valid and not raw_valid:
        core = (
            "A valid action sequence was decoded from the reasoning text, but the final answer/raw plan did not validate. "
            "This points to a transfer problem from reasoning to final answer"
            + (", plus final-answer formatting issues" if format_failure else "")
            + ". For the next attempt, use the valid plan already present in the reasoning text as the final answer and do not add explanation."
        )
    elif format_failure:
        core = (
            f"The previous final answer was not a clean parseable PDDL action sequence. {validation} "
            "For the next attempt, output only the corrected final action sequence."
        )
    elif not final_plan_valid and isinstance(attempt_record.get("first_valid_prefix_length"), int) and (attempt_record.get("extra_actions_after_first_valid") or 0) > 0:
        core = (
            "The previous final answer reached a valid solution prefix, then continued with extra actions. "
            "For the next attempt, stop as soon as the goal is reached and do not append actions after a valid plan."
        )
    elif error_type == "unsatisfied_goal":
        core = (
            f"The previous final answer was executable but did not satisfy the goal. {validation} "
            "For the next attempt, extend or modify the plan so the final state satisfies the problem goal."
        )
    elif alignment and error_type == "invalid_precondition" and alignment.get("action_bag_overlap_score", 0.0) >= cfg.high_alignment_threshold and not alignment.get("exact_sequence_match"):
        core = (
            f"The previous final answer and reasoning used mostly the same actions, but the order appears inconsistent with the validator failure. {validation} "
            "For the next attempt, rebuild the order so every action is applicable before it is executed."
        )
    elif alignment and alignment.get("structural_alignment", 0.0) >= cfg.high_alignment_threshold:
        core = (
            f"The previous final answer and reasoning were coherent with each other, but the shared plan was still invalid. {validation} "
            "For the next attempt, repair the plan itself using the validator failure rather than changing only the wording of the final answer."
        )
    elif alignment and alignment.get("structural_alignment", 1.0) <= cfg.low_alignment_threshold:
        core = (
            f"The previous final answer did not closely reflect the separate reasoning text. {validation} "
            "For the next attempt, rebuild one coherent plan and make the final answer match it."
        )
    elif not reasoning_actions:
        core = (
            f"No separate reasoning plan was available, so the repair should rely on the validator result. {validation} "
            "For the next attempt, correct the plan according to that failure."
        )
    else:
        core = (
            f"The previous final answer did not validate. {validation} "
            "For the next attempt, repair the plan and return one coherent corrected sequence."
        )

    return f"{core} {cfg.final_output_instruction}"
