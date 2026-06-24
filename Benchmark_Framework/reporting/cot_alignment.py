"""Reusable CoT plan-alignment utilities for post-run reporting."""

from __future__ import annotations

import math
import re
from typing import Any, Iterable, Optional

from evaluators.plan_alignment import (
    action_bag_overlap_score,
    common_prefix_length,
    compute_sequence_alignment,
    contiguous_repetition_blocks,
    detect_adjacent_swaps,
    lcs_length,
    normalize_action_sequence,
    repetition_similarity,
)


_ACT_RE = re.compile(r"\(\s*([^\s()]+)((?:\s+[^\s()]+)*)\s*\)")


def parse_action(action_str: str) -> tuple[Optional[str], list[str]]:
    """Parse a PDDL action string "(name arg1 arg2 ...)" into (name, [args]).

    Metric correlation: hallucination rate, executability ratio, PAS - all metrics
    are computed by iterating over parsed (name, args) pairs from the model's plan.
    Rationale: the model outputs free-form text containing action tuples. Extracting
    the action name and argument list with a single regex avoids per-character
    string splitting that would break on nested parentheses or extra whitespace.
    Code purpose: lowest-level tokeniser for a single plan step; called in
    ``compute_hallucination_metrics`` and ``compute_precondition_metrics``.
    Detail: ``_ACT_RE`` matches one parenthesized PDDL action atom.
    captures the head token as group 1 and the argument token list as group 2.
    Returns ``(None, [])`` if the string does not match PDDL atom syntax.
    """
    match = _ACT_RE.match(action_str.strip())
    if not match:
        return None, []
    name = match.group(1).lower()
    args = match.group(2).strip().lower().split() if match.group(2).strip() else []
    return name, args


def safe_get(data: Any, path: str | Iterable[Any], default: Any = None) -> Any:
    parts = path.split(".") if isinstance(path, str) else list(path)
    current = data
    missing = object()
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part, missing)
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            current = current[part]
        else:
            return default
        if current is missing:
            return default
    return current


def parsed_plan_raw_actions(parsed_plan: dict[str, Any]) -> list[str]:
    raw_plan = parsed_plan.get("raw")
    if isinstance(raw_plan, dict) and isinstance(raw_plan.get("actions"), list):
        return raw_plan["actions"]
    actions = parsed_plan.get("actions")
    return actions if isinstance(actions, list) else []


def parsed_plan_reasoning_actions(parsed_plan: dict[str, Any]) -> list[str]:
    reasoning = parsed_plan.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("actions"), list):
        return reasoning["actions"]
    return []


def parsed_plan_reasoning_text(parsed_plan: dict[str, Any]) -> str:
    reasoning = parsed_plan.get("reasoning")
    return reasoning if isinstance(reasoning, str) else ""


def raw_attempt_reasoning_text(raw_attempt: dict[str, Any] | None) -> str:
    value = safe_get(raw_attempt or {}, "generation.reasoning_text", "")
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def compute_cot_semantic_support(cot_text: str, actions: list[str], d_info: dict[str, Any], p_info: dict[str, Any]) -> dict[str, Any]:
    legal_action_names = d_info.get("action_names", set())
    legal_objects = p_info.get("objects", set())
    cot_tokens = set(re.findall(r"[a-z][a-z0-9_-]*", (cot_text or "").lower()))

    plan_action_names: set[str] = set()
    plan_objects: set[str] = set()
    for action in actions:
        name, args = parse_action(action)
        if name:
            plan_action_names.add(name)
            plan_objects.update(args)

    plan_terms = plan_action_names | plan_objects
    cot_action_mentioned = cot_tokens & legal_action_names
    cot_object_mentioned = cot_tokens & legal_objects
    cot_action_cov = len(cot_action_mentioned & plan_action_names) / max(len(plan_action_names), 1)
    cot_object_cov = len(cot_object_mentioned & plan_objects) / max(len(plan_objects), 1)
    cot_term_cov = len(cot_tokens & plan_terms) / max(len(plan_terms), 1)
    return {
        "cot_action_coverage": cot_action_cov,
        "cot_object_coverage": cot_object_cov,
        "cot_term_coverage": cot_term_cov,
        "cot_semantic_support_score": (cot_action_cov + cot_object_cov) / 2,
    }


def compute_cot_alignment(cot_text: str, actions: list[str], d_info: dict[str, Any], p_info: dict[str, Any]) -> dict[str, Any]:
    return compute_cot_semantic_support(cot_text, actions, d_info, p_info)


def _bool_or_none(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _prefix_ratio(prefix: Any, length: int) -> Optional[float]:
    if isinstance(prefix, int) and length > 0:
        return prefix / length
    return None


def compute_cot_alignment_for_attempt(
    parsed_attempt: dict[str, Any],
    scored_attempt: dict[str, Any] | None,
    raw_attempt: dict[str, Any] | None,
    d_info: dict[str, Any],
    p_info: dict[str, Any],
    semantic_proxy_cap: float = 0.35,
) -> dict[str, Any]:
    parsed_plan = parsed_attempt.get("parsed_plan") if isinstance(parsed_attempt, dict) else {}
    parsed_plan = parsed_plan if isinstance(parsed_plan, dict) else {}
    scored_attempt = scored_attempt if isinstance(scored_attempt, dict) else {}
    raw_actions = normalize_action_sequence(parsed_plan_raw_actions(parsed_plan))
    reasoning_actions = normalize_action_sequence(parsed_plan_reasoning_actions(parsed_plan))
    reasoning_text = raw_attempt_reasoning_text(raw_attempt) or parsed_plan_reasoning_text(parsed_plan)
    semantic = (
        compute_cot_semantic_support(reasoning_text, raw_actions, d_info, p_info)
        if reasoning_text.strip()
        else {
            "cot_action_coverage": None,
            "cot_object_coverage": None,
            "cot_term_coverage": None,
            "cot_semantic_support_score": None,
        }
    )

    raw_valid = _bool_or_none(safe_get(scored_attempt, "final_plan_valid"))
    raw_valid_inferred = False
    if raw_valid is None:
        raw_valid = _bool_or_none(safe_get(scored_attempt, "validation_result.valid"))
        raw_valid_inferred = raw_valid is not None
    reasoning_valid = _bool_or_none(safe_get(scored_attempt, "reasoning_final_plan_valid"))
    raw_prefix = safe_get(scored_attempt, "first_valid_prefix_length")
    reasoning_prefix = safe_get(scored_attempt, "reasoning_first_valid_prefix_length")

    sequence: dict[str, Any] = {}
    plan_score = None
    proxy_score = None
    exact_match = None
    confidence = "none"
    status = "no_reasoning_text"
    basis = "none"

    if not reasoning_text.strip() and not reasoning_actions:
        status = "no_reasoning_text"
    elif not raw_actions:
        status = "no_raw_plan"
        confidence = "low"
    elif raw_actions and reasoning_actions:
        sequence = compute_sequence_alignment(raw_actions, reasoning_actions)
        plan_score = sequence["structural_alignment"]
        exact_match = sequence["exact_sequence_match"]
        confidence = "high"
        if raw_valid is True and reasoning_valid is True:
            status = "comparable_and_both_valid"
        elif raw_valid is False and reasoning_valid is False:
            status = "comparable_but_both_invalid"
        elif raw_valid is False:
            status = "comparable_but_raw_invalid"
        elif reasoning_valid is False:
            status = "comparable_but_reasoning_invalid"
        else:
            status = "comparable_plans"
    else:
        score = semantic.get("cot_semantic_support_score")
        proxy_score = score * semantic_proxy_cap if isinstance(score, (int, float)) and math.isfinite(score) else None
        status = "semantic_proxy_only"
        confidence = "low"

    if raw_actions and reasoning_actions:
        if raw_valid is True and reasoning_valid is False:
            basis = "raw"
        elif raw_valid is False and reasoning_valid is True:
            basis = "reasoning"
        elif raw_valid is True and reasoning_valid is True:
            basis = "raw" if len(raw_actions) <= len(reasoning_actions) else "reasoning"
        elif raw_valid is False and reasoning_valid is False:
            basis = "reasoning"
        else:
            basis = "raw"

    raw_prefix_ratio = _prefix_ratio(raw_prefix, len(raw_actions))
    reasoning_prefix_ratio = _prefix_ratio(reasoning_prefix, len(reasoning_actions))
    iteration = parsed_attempt.get("iteration") if isinstance(parsed_attempt, dict) else None
    result = {
        "iteration": iteration,
        "cot_plan_alignment_score": plan_score,
        "cot_plan_alignment_proxy_score": proxy_score,
        "cot_alignment_status": status,
        "cot_alignment_confidence": confidence,
        "cot_reasoning_plan_available": bool(reasoning_actions),
        "cot_exact_sequence_match": exact_match,
        "strict_or_proxy_alignment_value": plan_score if plan_score is not None else proxy_score,
        "raw_valid": raw_valid,
        "raw_valid_inferred": raw_valid_inferred,
        "reasoning_valid": reasoning_valid,
        "raw_first_valid_prefix_length": raw_prefix,
        "reasoning_first_valid_prefix_length": reasoning_prefix,
        "raw_prefix_ratio": raw_prefix_ratio,
        "reasoning_prefix_ratio": reasoning_prefix_ratio,
        "raw_has_shorter_valid_prefix": isinstance(raw_prefix, int) and 0 < raw_prefix < len(raw_actions),
        "reasoning_has_shorter_valid_prefix": isinstance(reasoning_prefix, int) and 0 < reasoning_prefix < len(reasoning_actions),
        "basis": basis,
        **semantic,
        **sequence,
    }
    result.setdefault("raw_plan_length", len(raw_actions))
    result.setdefault("reasoning_plan_length", len(reasoning_actions))
    return result


def select_cot_alignment_attempt(by_iteration: list[dict[str, Any]], solved: bool) -> dict[str, Any]:
    if not by_iteration:
        return {"selected_attempt": None, "selected_attempt_reason": "none", "final": {}}
    if len(by_iteration) == 1:
        selected_index = 0
        reason = "only_attempt"
    elif solved:
        selected_index = next((index for index, item in enumerate(by_iteration) if item.get("raw_valid") is True), len(by_iteration) - 1)
        reason = "first_solved_attempt" if selected_index < len(by_iteration) - 1 or by_iteration[selected_index].get("raw_valid") is True else "last_attempt"
    else:
        selected_index = len(by_iteration) - 1
        reason = "last_attempt"
    final = by_iteration[selected_index]
    return {
        "selected_attempt": final.get("iteration") or selected_index + 1,
        "selected_attempt_reason": reason,
        "final": final,
    }
