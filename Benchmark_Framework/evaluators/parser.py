"""Shared parser for turning model output into structured benchmark plans."""

from __future__ import annotations

from typing import Any
import re


ACTION_PATTERN = re.compile(r"\([^()\n]+\)")
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[\).\]:-]?\s+)")
PLAN_MARKER_PATTERNS = (
    re.compile(r"^\s*(?:final\s+plan|final\s+answer|plan|actions?|action\s+sequence|answer)\s*:?\s*(.*)$", re.IGNORECASE),
)
FINAL_MARKER_PATTERN = re.compile(r"\bfinal\s+(?:answer|plan|action\s+sequence)\b|\bnow\s+produce\s+final\b", re.IGNORECASE)
SOURCE_REF = {"artifact": "raw", "field": "generation.reasoning_text"}
WORD_COUNTS = {"once": 1, "twice": 2, "thrice": 3}


class ParsedPlan:
    def __init__(self, raw: dict[str, Any] | None = None, reasoning: dict[str, Any] | None = None) -> None:
        self.raw = raw or {}
        self.reasoning = reasoning or {}

    @property
    def actions(self) -> list[str]:
        actions = self.raw.get("actions", [])
        return actions if isinstance(actions, list) else []

    @property
    def format_issues(self) -> list[str]:
        issues = self.raw.get("format_issues", [])
        return issues if isinstance(issues, list) else []


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _strip_markdown_fences(text: str) -> tuple[str, list[str]]:
    if "```" not in text:
        return text, []

    cleaned_lines: list[str] = []
    issues: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("```"):
            issues.append("markdown_fences_removed")
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines), _dedupe_preserve_order(issues)


def _strip_list_prefix(line: str) -> str:
    previous = ""
    stripped = line.strip()
    while stripped != previous:
        previous = stripped
        stripped = LIST_PREFIX_PATTERN.sub("", stripped).strip()
    return stripped


def _strip_pddl_comments(text: str) -> str:
    return re.sub(r";.*", "", text)


def _parse_domain_actions(domain_text: str | None) -> dict[str, dict[str, Any]] | None:
    if domain_text is None:
        return None

    text = _strip_pddl_comments(domain_text)
    matches = list(re.finditer(r"\(:action\s+([^\s()]+)", text, re.IGNORECASE))
    actions: dict[str, dict[str, Any]] = {}
    for index, match in enumerate(matches):
        name = match.group(1).lower()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start() : end]
        params = re.search(r":parameters\s*\(([^()]*)\)", block, re.IGNORECASE | re.DOTALL)
        arity = len(re.findall(r"\?[A-Za-z0-9_-]+", params.group(1))) if params else 0
        actions[name] = {"name": name, "arity": arity}
    return actions


def _normalize_action(action_text: str, schemas: dict[str, dict[str, Any]] | None) -> tuple[str | None, list[str]]:
    inner = re.sub(r"\s+", " ", action_text[1:-1].strip())
    if not inner:
        return None, ["empty_parenthesized_expression_found"]

    tokens = inner.split()
    action_name = tokens[0].lower()
    args = tokens[1:]
    if schemas is None:
        return f"({inner})", []

    schema = schemas.get(action_name)
    if schema is None:
        return None, ["unknown_action_names_found"]
    if len(args) != schema["arity"]:
        return None, ["wrong_action_arity"]
    return f"({schema['name']} {' '.join(args)})" if args else f"({schema['name']})", []


def _repeat_count(line: str, start: int, end: int) -> int:
    before = line[:start].lower()
    after = line[end:].lower()
    word_pattern = "|".join(WORD_COUNTS)
    patterns = (
        re.search(r"\(?\s*(?:repeat\s+exactly|repeat(?:ed)?)\s+(\d+)\s+times?\s*\)?\s*$", before),
        re.search(r"(\d+)\s+times?\s*$", before),
        re.search(r"\b(" + word_pattern + r")\s*$", before),
        re.search(r"^\s*(?:x|\*)\s*(\d+)\b", after),
        re.search(r"^\s*(?:repeated\s+|for\s+)?(\d+)\s+times?\b", after),
        re.search(r"^\s*(" + word_pattern + r")\b", after),
    )
    for match in patterns:
        if match:
            value = match.group(1)
            return WORD_COUNTS[value] if value in WORD_COUNTS else max(1, int(value))
    return 1


def _build_alias_map(schemas: dict[str, dict[str, Any]] | None) -> dict[str, str]:
    if not schemas:
        return {}

    seen: dict[str, str | None] = {}
    for name, schema in schemas.items():
        if schema["arity"] != 1:
            continue
        parts = re.split(r"[_-]+", name)
        for alias in {name, parts[-1]}:
            seen[alias] = name if alias not in seen else None
    return {alias: name for alias, name in seen.items() if name is not None}


def _one_arg_action_object(action: str, schemas: dict[str, dict[str, Any]] | None) -> str | None:
    if schemas is None:
        return None

    tokens = action.strip("()").split()
    if len(tokens) != 2:
        return None
    schema = schemas.get(tokens[0].lower())
    return tokens[1] if schema and schema["arity"] == 1 else None


def _parenthesized_repeat_actions(
    expression: str,
    schemas: dict[str, dict[str, Any]] | None,
    alias_map: dict[str, str],
    current_object: str | None,
) -> tuple[list[str] | None, list[str]]:
    if not schemas:
        return None, []

    inner = re.sub(r"\s+", " ", expression[1:-1].strip())
    match = re.match(r"^repeat\s+([A-Za-z][\w-]*)\s+(\d+)\s+times?$", inner, re.IGNORECASE)
    if not match:
        return None, []

    action_name = alias_map.get(match.group(1).lower())
    if action_name is None or current_object is None:
        return [], ["ambiguous_compressed_action"]

    count = max(1, int(match.group(2)))
    issues = ["compressed_actions_expanded"] if count > 1 else []
    return [f"({action_name} {current_object})"] * count, issues


def _compressed_line_actions(
    line: str,
    schemas: dict[str, dict[str, Any]] | None,
    alias_map: dict[str, str],
    current_object: str | None,
) -> tuple[list[str], list[str], str | None]:
    if not schemas or not alias_map:
        return [], [], current_object

    stripped = _strip_list_prefix(line).strip(" \t.;")
    if not stripped:
        return [], [], current_object

    current_step = re.match(r"^([A-Za-z][\w-]*)\s+\d+\s*(?:times?)?$", stripped, re.IGNORECASE)
    if current_object and current_step and current_step.group(1).lower() in alias_map:
        obj = current_object
        rest = stripped
    else:
        match = re.match(r"^(?:then\s+)?([A-Za-z][\w-]*)\s*:?\s+(.+)$", stripped, re.IGNORECASE)
        if not match:
            return [], [], current_object
        obj = match.group(1)
        rest = match.group(2)

    actions: list[str] = []
    issues: list[str] = []
    for part in re.split(r"[,;]", rest):
        step = part.strip(" \t.")
        step_match = re.match(r"^([A-Za-z][\w-]*)\s+(\d+)\s*(?:times?)?$", step, re.IGNORECASE)
        if not step_match:
            continue
        alias = step_match.group(1).lower()
        action_name = alias_map.get(alias)
        if action_name is None:
            issues.append("ambiguous_compressed_action")
            continue
        count = max(1, int(step_match.group(2)))
        actions.extend([f"({action_name} {obj})"] * count)
        if count > 1:
            issues.append("compressed_actions_expanded")

    return actions, issues, obj if actions else current_object


def _line_actions(
    line: str,
    schemas: dict[str, dict[str, Any]] | None,
    alias_map: dict[str, str],
    current_object: str | None,
) -> tuple[list[str], list[str], str | None]:
    stripped = re.sub(r"^\s*\d+\s*-\s*\d+\s*:\s*", "", line.strip())
    if not re.match(r"^\d+\s+times?\b", stripped, re.IGNORECASE):
        stripped = _strip_list_prefix(stripped)
    matches = list(ACTION_PATTERN.finditer(stripped))
    actions: list[str] = []
    issues: list[str] = []

    for match in matches:
        repeated, repeat_issues = _parenthesized_repeat_actions(match.group(0), schemas, alias_map, current_object)
        if repeated is not None:
            actions.extend(repeated)
            issues.extend(repeat_issues)
            continue

        action, action_issues = _normalize_action(match.group(0), schemas)
        issues.extend(action_issues)
        if action is None:
            continue
        count = _repeat_count(stripped, match.start(), match.end())
        actions.extend([action] * count)
        if count > 1:
            issues.append("compressed_actions_expanded")
        current_object = _one_arg_action_object(action, schemas) or current_object

    if matches:
        remainder = ACTION_PATTERN.sub("", stripped).strip(" \t-:;,.>")
        remainder = re.sub(r"(?:repeat\s+exactly|repeat(?:ed)?|for)\s+\d+\s+times?", "", remainder, flags=re.IGNORECASE).strip()
        remainder = re.sub(r"\d+\s+times?", "", remainder, flags=re.IGNORECASE).strip()
        remainder = re.sub(r"^(?:x|\*)\s*\d+\b", "", remainder, flags=re.IGNORECASE).strip()
        if remainder:
            issues.append("actions_embedded_in_text")
        return actions, issues, current_object

    compressed, compressed_issues, current_object = _compressed_line_actions(
        stripped,
        schemas,
        alias_map,
        current_object,
    )
    return compressed, compressed_issues, current_object


def _line_has_truncated_action(line: str, schemas: dict[str, dict[str, Any]] | None) -> bool:
    stripped = line.strip()
    if stripped.count("(") <= stripped.count(")"):
        return False
    fragment = stripped[stripped.rfind("(") + 1 :].strip()
    if not fragment:
        return True
    action_name = fragment.split()[0].lower()
    return schemas is None or action_name in schemas


def _leading_list_number(line: str) -> int | None:
    if re.match(r"^\s*\d+\s+times?\b", line, re.IGNORECASE):
        return None
    match = re.match(r"^\s*(\d+)[\).\]:-]?\s+", line)
    return int(match.group(1)) if match else None


def _candidate(
    actions: list[str],
    issues: list[str],
    source_index: int,
    near_final_marker: bool,
    truncated: bool = False,
    numbered: bool = False,
) -> dict[str, Any]:
    candidate_issues = list(issues)
    if truncated:
        candidate_issues.append("truncated_reasoning_candidate")
    return {
        "actions": list(actions),
        "format_issues": _dedupe_preserve_order(candidate_issues),
        "source_index": source_index,
        "near_final_marker": near_final_marker,
        "truncated": truncated,
        "numbered": numbered,
    }


def _extract_action_candidates(
    text: str,
    schemas: dict[str, dict[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    candidates: list[dict[str, Any]] = []
    current_segment: list[str] = []
    current_issues: list[str] = []
    current_numbered_segment: list[str] = []
    current_numbered_issues: list[str] = []
    current_near_final = False
    numbered_near_final = False
    expected_number: int | None = None
    current_object: str | None = None
    alias_map = _build_alias_map(schemas)

    def flush_segment(truncated: bool = False) -> None:
        nonlocal current_segment, current_issues, current_near_final
        if current_segment:
            candidates.append(_candidate(current_segment, current_issues, len(candidates), current_near_final, truncated))
        current_segment = []
        current_issues = []
        current_near_final = False

    def flush_numbered(truncated: bool = False) -> None:
        nonlocal current_numbered_segment, current_numbered_issues, numbered_near_final
        if current_numbered_segment:
            candidates.append(
                _candidate(current_numbered_segment, current_numbered_issues, len(candidates), numbered_near_final, truncated, True)
            )
        current_numbered_segment = []
        current_numbered_issues = []
        numbered_near_final = False

    marker_pending = False
    for line in text.splitlines():
        marker_pending = marker_pending or bool(FINAL_MARKER_PATTERN.search(line))
        line_actions, line_issues, current_object = _line_actions(line, schemas, alias_map, current_object)
        issues.extend(line_issues)
        line_number = _leading_list_number(line)
        if line_actions:
            current_segment.extend(line_actions)
            current_issues.extend(line_issues)
            current_near_final = current_near_final or marker_pending
            if line_number is not None:
                if expected_number is not None and line_number != expected_number:
                    flush_numbered()
                current_numbered_segment.extend(line_actions)
                current_numbered_issues.extend(line_issues)
                numbered_near_final = numbered_near_final or marker_pending
                expected_number = line_number + 1
            elif current_numbered_segment:
                flush_numbered()
                expected_number = None
            marker_pending = False
            continue

        truncated = _line_has_truncated_action(line, schemas)
        if current_segment:
            flush_segment(truncated)
        if truncated:
            flush_numbered(True)
            expected_number = None

    flush_segment()
    flush_numbered()
    return candidates, _dedupe_preserve_order(issues)


def _extract_actions(
    text: str,
    schemas: dict[str, dict[str, Any]] | None,
    *,
    select_candidate: bool = False,
) -> tuple[list[str], list[str]]:
    candidates, issues = _extract_action_candidates(text, schemas)
    unnumbered = [candidate for candidate in candidates if not candidate.get("numbered")] or candidates
    actions = [action for candidate in unnumbered for action in candidate["actions"]]

    if select_candidate and candidates:
        candidate_pool = [candidate for candidate in candidates if candidate.get("numbered")] or candidates
        if len(candidate_pool) > 1:
            issues.append("multiple_reasoning_candidate_plans")
        best = max(
            candidate_pool,
            key=lambda candidate: (
                bool(candidate["near_final_marker"]),
                int(candidate["source_index"]),
                len(candidate["actions"]),
            ),
        )
        if len(candidate_pool) > 1 and sum(1 for candidate in candidate_pool if len(candidate["actions"]) == len(best["actions"])) > 1:
            issues.append("ambiguous_reasoning_plan_selection")
        actions = list(best["actions"])
        issues.extend(best["format_issues"])

    return actions, _dedupe_preserve_order(issues)


def extract_reasoning_candidates(reasoning_text: str, domain_text: str | None = None) -> list[dict[str, Any]]:
    """Return domain-valid reasoning plan candidates without selecting one."""
    if not reasoning_text or not reasoning_text.strip():
        return []
    schemas = _parse_domain_actions(domain_text)
    cleaned_text, fence_issues = _strip_markdown_fences(reasoning_text)
    candidates, shared_issues = _extract_action_candidates(cleaned_text, schemas)
    for candidate in candidates:
        candidate["format_issues"] = _dedupe_preserve_order(fence_issues + shared_issues + candidate["format_issues"])
    return candidates


def _split_reasoning_and_plan(text: str) -> tuple[str, str, list[str]]:
    if not text.strip():
        return "", "", []

    lines = text.splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        for pattern in PLAN_MARKER_PATTERNS:
            match = pattern.match(stripped)
            if not match:
                continue
            reasoning = "\n".join(lines[:index]).strip()
            trailing = match.group(1).strip()
            remaining_lines = lines[index + 1 :]
            if trailing:
                remaining_lines = [trailing] + remaining_lines
            issues = ["plan_section_marker_removed"]
            if reasoning:
                issues.append("reasoning_before_plan_removed")
            return reasoning, "\n".join(remaining_lines).strip(), issues

    for index, line in enumerate(lines):
        if ACTION_PATTERN.search(line):
            reasoning = "\n".join(lines[:index]).strip()
            issues: list[str] = []
            if reasoning:
                issues.append("reasoning_before_plan_removed")
            return reasoning, "\n".join(lines[index:]).strip(), issues

    return text.strip(), text.strip(), []


def _raw_section(raw_text: str, schemas: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not raw_text or not raw_text.strip():
        return {
            "actions": [],
            "format_issues": ["empty_output"],
            "contains_reasoning": False,
            "source_kind": "empty_raw_plan",
        }

    cleaned_text, fence_issues = _strip_markdown_fences(raw_text)
    parser_reasoning, plan_text, split_issues = _split_reasoning_and_plan(cleaned_text)
    actions, extraction_issues = _extract_actions(plan_text or cleaned_text, schemas)
    format_issues = fence_issues + split_issues + extraction_issues
    contains_reasoning = bool(parser_reasoning) or "actions_embedded_in_text" in extraction_issues

    if contains_reasoning:
        format_issues.append("raw_text_contains_reasoning_like_content")
    if not actions:
        format_issues.append("no_valid_domain_actions_found" if schemas is not None else "no_parenthesized_actions_found")

    return {
        "actions": actions,
        "format_issues": _dedupe_preserve_order(format_issues),
        "contains_reasoning": contains_reasoning,
        "source_kind": "reasoning_like_raw" if contains_reasoning else "clean_raw_plan",
    }


def _reasoning_section(reasoning_text: str, schemas: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not reasoning_text or not reasoning_text.strip():
        return {
            "actions": [],
            "format_issues": ["reasoning_text_empty"],
            "source_ref": dict(SOURCE_REF),
        }

    cleaned_text, fence_issues = _strip_markdown_fences(reasoning_text)
    actions, extraction_issues = _extract_actions(cleaned_text, schemas, select_candidate=True)
    format_issues = fence_issues + extraction_issues
    if not actions:
        format_issues.append("no_valid_domain_actions_found" if schemas is not None else "no_parenthesized_actions_found")

    return {
        "actions": actions,
        "format_issues": _dedupe_preserve_order(format_issues),
        "source_ref": dict(SOURCE_REF),
    }


def parse_plan_text(raw_text: str, reasoning_text: str = "", domain_text: str | None = None) -> ParsedPlan:
    """Parse model text into raw-official and reasoning-diagnostic plans."""
    schemas = _parse_domain_actions(domain_text)
    return ParsedPlan(
        raw=_raw_section(raw_text, schemas),
        reasoning=_reasoning_section(reasoning_text, schemas),
    )
