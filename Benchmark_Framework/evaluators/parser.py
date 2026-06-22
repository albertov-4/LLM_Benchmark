"""Shared parser for turning model output into structured benchmark plans."""

from __future__ import annotations

from typing import Any
import re


ACTION_PATTERN = re.compile(r"\([^()\n]+\)")
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[\).\]:-]?\s+)")
PLAN_MARKER_PATTERNS = (
    re.compile(r"^\s*(?:final\s+plan|final\s+answer|plan|actions?|action\s+sequence|answer)\s*:?\s*(.*)$", re.IGNORECASE),
)
SOURCE_REF = {"artifact": "raw", "field": "generation.reasoning_text"}


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
    patterns = (
        re.search(r"(?:repeat\s+exactly|repeat(?:ed)?)\s+(\d+)\s+times?\s*$", before),
        re.search(r"^\s*(?:x|\*)\s*(\d+)\b", after),
        re.search(r"^\s*(?:repeated\s+|for\s+)?(\d+)\s+times?\b", after),
    )
    for match in patterns:
        if match:
            return max(1, int(match.group(1)))
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
    stripped = _strip_list_prefix(line)
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


def _leading_list_number(line: str) -> int | None:
    match = re.match(r"^\s*(\d+)[\).\]:-]?\s+", line)
    return int(match.group(1)) if match else None


def _extract_actions(
    text: str,
    schemas: dict[str, dict[str, Any]] | None,
    *,
    select_candidate: bool = False,
) -> tuple[list[str], list[str]]:
    actions: list[str] = []
    issues: list[str] = []
    segments: list[list[str]] = []
    numbered_segments: list[list[str]] = []
    current_segment: list[str] = []
    current_numbered_segment: list[str] = []
    expected_number: int | None = None
    current_object: str | None = None
    alias_map = _build_alias_map(schemas)

    for line in text.splitlines():
        line_actions, line_issues, current_object = _line_actions(line, schemas, alias_map, current_object)
        issues.extend(line_issues)
        line_number = _leading_list_number(line)
        if line_actions:
            actions.extend(line_actions)
            current_segment.extend(line_actions)
            if line_number is not None:
                if expected_number is not None and line_number != expected_number:
                    if current_numbered_segment:
                        numbered_segments.append(current_numbered_segment)
                    current_numbered_segment = []
                current_numbered_segment.extend(line_actions)
                expected_number = line_number + 1
            elif current_numbered_segment:
                numbered_segments.append(current_numbered_segment)
                current_numbered_segment = []
                expected_number = None
            continue
        if current_segment:
            segments.append(current_segment)
            current_segment = []
    if current_segment:
        segments.append(current_segment)
    if current_numbered_segment:
        numbered_segments.append(current_numbered_segment)

    if select_candidate and (numbered_segments or segments):
        candidate_segments = numbered_segments or segments
        if len(candidate_segments) > 1:
            issues.append("multiple_reasoning_candidate_plans")
        best_index, best_segment = max(enumerate(candidate_segments), key=lambda item: (len(item[1]), item[0]))
        if len(candidate_segments) > 1 and sum(1 for segment in candidate_segments if len(segment) == len(best_segment)) > 1:
            issues.append("ambiguous_reasoning_plan_selection")
        actions = list(best_segment)

    return actions, _dedupe_preserve_order(issues)


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
