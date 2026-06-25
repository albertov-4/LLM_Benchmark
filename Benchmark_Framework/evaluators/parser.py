"""Shared parser for turning model output into structured benchmark plans."""

from __future__ import annotations

from typing import Any
import re


ACTION_PATTERN = re.compile(r"\([^()\n]+\)")
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]\s+|\d+[\).\]:-]?\s+)")
PLAN_MARKER_PATTERNS = (
    re.compile(r"^\s*(?:final\s+plan|final\s+answer|plan|actions?|action\s+sequence|answer)\s*:?\s*(.*)$", re.IGNORECASE),
)
FINAL_MARKER_PATTERN = re.compile(r"\bfinal\s+(?:answer|plan|list|action\s+sequence)\b|\bnow\s+produce\s+final\b", re.IGNORECASE)
SOURCE_REF = {"artifact": "raw", "field": "generation.reasoning_text"}
WORD_COUNTS = {
    "once": 1,
    "twice": 2,
    "thrice": 3,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "one hundred": 100,
}


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


def _parse_typed_list(text: str, *, variables_only: bool = False) -> list[tuple[str, str | None]]:
    tokens = re.findall(r"[^\s()]+", text)
    result: list[tuple[str, str | None]] = []
    pending: list[str] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "-":
            type_name = tokens[index + 1].lower() if index + 1 < len(tokens) else None
            result.extend((name, type_name) for name in pending)
            pending = []
            index += 2
            continue
        if not token.startswith(":") and (not variables_only or token.startswith("?")):
            pending.append(token)
        index += 1
    result.extend((name, None) for name in pending)
    return result


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
        parameter_types = [type_name for _name, type_name in _parse_typed_list(params.group(1), variables_only=True)] if params else []
        actions[name] = {"name": name, "arity": len(parameter_types), "parameter_types": parameter_types}
    return actions


def _balanced_section(text: str, marker: str) -> str | None:
    match = re.search(r"\(" + re.escape(marker) + r"\b", text, re.IGNORECASE)
    if not match:
        return None
    depth = 1
    index = match.end()
    while index < len(text):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return text[match.end() : index]
        index += 1
    return text[match.end() :]


def _parse_problem_objects(problem_text: str | None) -> dict[str | None, list[str]] | None:
    if problem_text is None:
        return None

    section = _balanced_section(_strip_pddl_comments(problem_text), ":objects")
    if section is None:
        return None

    objects: dict[str | None, list[str]] = {"*": []}
    for name, type_name in _parse_typed_list(section):
        if name.startswith("?"):
            continue
        objects.setdefault(type_name, []).append(name)
        objects["*"].append(name)
    return objects if objects["*"] else None


def _infer_missing_args(
    action_name: str,
    args: list[str],
    schemas: dict[str, dict[str, Any]],
    objects_by_type: dict[str | None, list[str]] | None,
) -> list[str] | None:
    if not objects_by_type:
        return None

    schema = schemas[action_name]
    parameter_types = schema.get("parameter_types", [None] * schema["arity"])
    if len(args) >= schema["arity"]:
        return None

    inferred = list(args)
    for type_name in parameter_types[len(args) :]:
        candidates = objects_by_type.get(type_name) if type_name else objects_by_type.get("*")
        if not candidates or len(candidates) != 1:
            return None
        inferred.append(candidates[0])
    return inferred


def _normalize_action(
    action_text: str,
    schemas: dict[str, dict[str, Any]] | None,
    objects_by_type: dict[str | None, list[str]] | None = None,
) -> tuple[str | None, list[str]]:
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
        inferred_args = _infer_missing_args(action_name, args, schemas, objects_by_type)
        if inferred_args is None or len(inferred_args) != schema["arity"]:
            return None, ["wrong_action_arity"]
        args = inferred_args
        return f"({schema['name']} {' '.join(args)})" if args else f"({schema['name']})", ["compressed_actions_expanded"]
    return f"({schema['name']} {' '.join(args)})" if args else f"({schema['name']})", []


def _repeat_count(line: str, start: int, end: int) -> int:
    before = line[:start].lower()
    after = line[end:].lower()
    word_pattern = "|".join(re.escape(word) for word in sorted(WORD_COUNTS, key=len, reverse=True))
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


def _parse_count(text: str) -> int | None:
    cleaned = re.sub(r"[-_]+", " ", text.lower()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    if cleaned.isdigit():
        return max(1, int(cleaned))
    if cleaned in WORD_COUNTS:
        return WORD_COUNTS[cleaned]
    parts = cleaned.split()
    if len(parts) == 2 and parts[0] in WORD_COUNTS and parts[1] == "hundred":
        return max(1, WORD_COUNTS[parts[0]] * 100)
    if len(parts) == 2 and parts[0] in WORD_COUNTS and parts[1] in WORD_COUNTS:
        return max(1, WORD_COUNTS[parts[0]] + WORD_COUNTS[parts[1]])
    return None


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
    objects_by_type: dict[str | None, list[str]] | None,
) -> tuple[list[str] | None, list[str]]:
    if not schemas:
        return None, []

    inner = re.sub(r"\s+", " ", expression[1:-1].strip())
    match = re.match(r"^repeat\s+([A-Za-z][\w-]*)\s+(.+?)\s+times?$", inner, re.IGNORECASE)
    if not match:
        return None, []

    action_name = alias_map.get(match.group(1).lower())
    count = _parse_count(match.group(2))
    if action_name is None or count is None:
        return [], ["ambiguous_compressed_action"]
    if current_object is None:
        inferred_action, inferred_issues = _normalize_action(f"({action_name})", schemas, objects_by_type)
        if inferred_action is None:
            return [], ["ambiguous_compressed_action"]
        issues = ["compressed_actions_expanded"] if count > 1 else []
        return [inferred_action] * count, _dedupe_preserve_order(issues + inferred_issues)

    issues = ["compressed_actions_expanded"] if count > 1 else []
    return [f"({action_name} {current_object})"] * count, issues


def _non_parenthesized_repeat_actions(
    line: str,
    schemas: dict[str, dict[str, Any]] | None,
    objects_by_type: dict[str | None, list[str]] | None,
) -> tuple[list[str], list[str]]:
    if not schemas:
        return [], []

    stripped = line.strip(" \t.;")
    prefix = re.match(r"^(?:repeat\s+)?(.+?)\s+times?\s+(.+)$", stripped, re.IGNORECASE)
    if prefix:
        pairs = [(prefix.group(2), prefix.group(1))]
    else:
        repeated = re.match(r"^(.+?)\s+repeated\s+(.+?)\s+times?$", stripped, re.IGNORECASE)
        pairs = [(repeated.group(1), repeated.group(2))] if repeated else []
        tokens = stripped.split()
        for tail_size in (2, 1):
            if len(tokens) > tail_size + 1 and tokens[-1].lower().startswith("time"):
                pairs.append((" ".join(tokens[:-tail_size - 1]), " ".join(tokens[-tail_size - 1:-1])))

    for action_text, count_text in pairs:
        count = _parse_count(count_text)
        if count is None:
            continue
        action, action_issues = _normalize_action(f"({action_text})", schemas, objects_by_type)
        if action is None:
            action_name = action_text.split()[0].lower() if action_text.split() else ""
            if action_name in schemas and "wrong_action_arity" in action_issues:
                return [], ["ambiguous_compressed_action"]
            continue
        issues = ["compressed_actions_expanded"] if count > 1 else []
        return [action] * count, _dedupe_preserve_order(issues + action_issues)
    return [], []

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
        step_match = re.match(r"^([A-Za-z][\w-]*)\s+(?:x|\*)?\s*(.+?)\s*(?:times?)?$", step, re.IGNORECASE)
        if not step_match:
            continue
        alias = step_match.group(1).lower()
        action_name = alias_map.get(alias)
        count = _parse_count(step_match.group(2))
        if action_name is None or count is None:
            issues.append("ambiguous_compressed_action")
            continue
        actions.extend([f"({action_name} {obj})"] * count)
        if count > 1:
            issues.append("compressed_actions_expanded")

    return actions, issues, obj if actions else current_object


def _line_actions(
    line: str,
    schemas: dict[str, dict[str, Any]] | None,
    alias_map: dict[str, str],
    current_object: str | None,
    objects_by_type: dict[str | None, list[str]] | None,
) -> tuple[list[str], list[str], str | None]:
    stripped = line.strip()
    if not re.match(r"^\d+\s+times?\b", stripped, re.IGNORECASE):
        stripped = re.sub(r"^\s*\d+\s*-\s*\d+\s*:\s*", "", stripped)
    word_pattern = "|".join(re.escape(word) for word in sorted(WORD_COUNTS, key=len, reverse=True))
    if not re.match(r"^\d+\s+times?\b", stripped, re.IGNORECASE):
        stripped = _strip_list_prefix(stripped)
    matches = list(ACTION_PATTERN.finditer(stripped))
    actions: list[str] = []
    issues: list[str] = []

    for match in matches:
        repeated, repeat_issues = _parenthesized_repeat_actions(match.group(0), schemas, alias_map, current_object, objects_by_type)
        if repeated is not None:
            actions.extend(repeated)
            issues.extend(repeat_issues)
            continue

        action, action_issues = _normalize_action(match.group(0), schemas, objects_by_type)
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
        remainder = re.sub(r"(?:" + word_pattern + r")\s+times?", "", remainder, flags=re.IGNORECASE).strip()
        remainder = re.sub(r"^(?:x|\*)\s*\d+\b", "", remainder, flags=re.IGNORECASE).strip()
        if remainder:
            issues.append("actions_embedded_in_text")
        return actions, issues, current_object

    repeated, repeat_issues = _non_parenthesized_repeat_actions(stripped, schemas, objects_by_type)
    if repeated:
        current_object = _one_arg_action_object(repeated[-1], schemas) or current_object
        return repeated, repeat_issues, current_object
    if repeat_issues:
        return [], repeat_issues, current_object

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


BOUNDARY_PATTERN = re.compile(
    r"\b(?:alternative|maybe|could|try|instead|check|verify|count|wrong|invalid)\b",
    re.IGNORECASE,
)


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


def _is_composite_clause_line(line: str, line_number: int | None) -> bool:
    stripped = re.sub(r"^\s*\d+\s*-\s*\d+\s*:\s*", "", line.strip())
    if line_number is not None:
        return True
    if not re.match(r"^\d+\s+times?\b", stripped, re.IGNORECASE):
        stripped = _strip_list_prefix(stripped)
    stripped = stripped.strip()
    if stripped.startswith("("):
        return True
    if re.match(r"^(?:repeat\b|\d+\s+times?\b)", stripped, re.IGNORECASE):
        return True
    return bool(re.match(r"^[A-Za-z][\w-]*\s*:?\s+[A-Za-z][\w-]*(?:\s+(?:x|\*)?[A-Za-z0-9_-]+(?:\s+times?)?)?(?:\s*[,;]|\s*$)", stripped, re.IGNORECASE))


def _extract_action_candidates(
    text: str,
    schemas: dict[str, dict[str, Any]] | None,
    objects_by_type: dict[str | None, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    issues: list[str] = []
    candidates: list[dict[str, Any]] = []
    current_segment: list[str] = []
    current_issues: list[str] = []
    current_numbered_segment: list[str] = []
    current_numbered_issues: list[str] = []
    composite_segment: list[str] = []
    composite_issues: list[str] = []
    current_near_final = False
    numbered_near_final = False
    composite_near_final = False
    composite_start_index = 0
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

    def flush_composite(truncated: bool = False) -> None:
        nonlocal composite_segment, composite_issues, composite_near_final, composite_start_index
        if composite_segment and not any(candidate["actions"] == composite_segment for candidate in candidates):
            candidate = _candidate(composite_segment, composite_issues, composite_start_index, composite_near_final, truncated)
            candidate["composite"] = True
            candidates.append(candidate)
        composite_segment = []
        composite_issues = []
        composite_near_final = False
        composite_start_index = len(candidates)

    marker_pending = False
    for line in text.splitlines():
        has_boundary = bool(BOUNDARY_PATTERN.search(line))
        has_final_marker = bool(FINAL_MARKER_PATTERN.search(line))
        if (has_boundary or has_final_marker) and composite_segment:
            flush_composite()
        marker_pending = marker_pending or has_final_marker
        line_actions, line_issues, current_object = _line_actions(line, schemas, alias_map, current_object, objects_by_type)
        issues.extend(line_issues)
        line_number = _leading_list_number(line)
        if line_actions:
            if _is_composite_clause_line(line, line_number):
                if not composite_segment:
                    composite_start_index = len(candidates)
                composite_segment.extend(line_actions)
                composite_issues.extend(line_issues)
                composite_near_final = composite_near_final or marker_pending
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

        if line_issues:
            current_issues.extend(line_issues)
        truncated = _line_has_truncated_action(line, schemas)
        if truncated and composite_segment:
            flush_composite(True)
        if current_segment:
            flush_segment(truncated)
        if truncated:
            flush_numbered(True)
            expected_number = None

    flush_segment()
    flush_numbered()
    flush_composite()
    return candidates, _dedupe_preserve_order(issues)

def _extract_actions(
    text: str,
    schemas: dict[str, dict[str, Any]] | None,
    objects_by_type: dict[str | None, list[str]] | None = None,
    *,
    select_candidate: bool = False,
) -> tuple[list[str], list[str]]:
    candidates, issues = _extract_action_candidates(text, schemas, objects_by_type)
    unnumbered = [candidate for candidate in candidates if not candidate.get("numbered") and not candidate.get("composite")] or candidates
    actions = [action for candidate in unnumbered for action in candidate["actions"]]

    if select_candidate and candidates:
        candidate_pool = candidates
        if len(candidate_pool) > 1:
            issues.append("multiple_reasoning_candidate_plans")
        best = max(
            candidate_pool,
            key=lambda candidate: (
                bool(candidate["near_final_marker"]),
                len(candidate["actions"]),
                int(candidate["source_index"]),
            ),
        )
        if len(candidate_pool) > 1 and sum(1 for candidate in candidate_pool if len(candidate["actions"]) == len(best["actions"])) > 1:
            issues.append("ambiguous_reasoning_plan_selection")
        actions = list(best["actions"])
        issues.extend(best["format_issues"])

    return actions, _dedupe_preserve_order(issues)


def extract_reasoning_candidates(
    reasoning_text: str,
    domain_text: str | None = None,
    problem_text: str | None = None,
) -> list[dict[str, Any]]:
    """Return domain-valid reasoning plan candidates without selecting one."""
    if not reasoning_text or not reasoning_text.strip():
        return []
    schemas = _parse_domain_actions(domain_text)
    objects_by_type = _parse_problem_objects(problem_text)
    cleaned_text, fence_issues = _strip_markdown_fences(reasoning_text)
    candidates, shared_issues = _extract_action_candidates(cleaned_text, schemas, objects_by_type)
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


def _raw_section(
    raw_text: str,
    schemas: dict[str, dict[str, Any]] | None,
    objects_by_type: dict[str | None, list[str]] | None,
) -> dict[str, Any]:
    if not raw_text or not raw_text.strip():
        return {
            "actions": [],
            "format_issues": ["empty_output"],
            "contains_reasoning": False,
            "source_kind": "empty_raw_plan",
        }

    cleaned_text, fence_issues = _strip_markdown_fences(raw_text)
    parser_reasoning, plan_text, split_issues = _split_reasoning_and_plan(cleaned_text)
    actions, extraction_issues = _extract_actions(plan_text or cleaned_text, schemas, objects_by_type)
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


def _reasoning_section(
    reasoning_text: str,
    schemas: dict[str, dict[str, Any]] | None,
    objects_by_type: dict[str | None, list[str]] | None,
) -> dict[str, Any]:
    if not reasoning_text or not reasoning_text.strip():
        return {
            "actions": [],
            "format_issues": ["reasoning_text_empty"],
            "source_ref": dict(SOURCE_REF),
        }

    cleaned_text, fence_issues = _strip_markdown_fences(reasoning_text)
    actions, extraction_issues = _extract_actions(cleaned_text, schemas, objects_by_type, select_candidate=True)
    format_issues = fence_issues + extraction_issues
    if not actions:
        format_issues.append("no_valid_domain_actions_found" if schemas is not None else "no_parenthesized_actions_found")

    return {
        "actions": actions,
        "format_issues": _dedupe_preserve_order(format_issues),
        "source_ref": dict(SOURCE_REF),
    }


def parse_plan_text(
    raw_text: str,
    reasoning_text: str = "",
    domain_text: str | None = None,
    problem_text: str | None = None,
) -> ParsedPlan:
    """Parse model text into raw-official and reasoning-diagnostic plans."""
    schemas = _parse_domain_actions(domain_text)
    objects_by_type = _parse_problem_objects(problem_text)
    return ParsedPlan(
        raw=_raw_section(raw_text, schemas, objects_by_type),
        reasoning=_reasoning_section(reasoning_text, schemas, objects_by_type),
    )
