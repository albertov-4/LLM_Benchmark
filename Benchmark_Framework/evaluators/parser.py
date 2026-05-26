"""Shared parser for turning raw model output into a structured plan.

The parser is intentionally lightweight, but it is now robust enough for
real-world LLM outputs that often include:
- reasoning before the final plan
- markdown code fences
- numbered or bulleted action lists
- actions embedded in otherwise noisy text
"""

from dataclasses import dataclass, field
import re


ACTION_PATTERN = re.compile(r"\([^()\n]+\)")
LIST_PREFIX_PATTERN = re.compile(r"^\s*(?:[-*+]|(?:\d+[\).\]:-]))\s*")
PLAN_MARKER_PATTERNS = (
    re.compile(r"^\s*(?:final\s+plan|plan|actions?|action\s+sequence|answer)\s*:?\s*(.*)$", re.IGNORECASE),
)


@dataclass
class ParsedPlan:
    actions: list[str] = field(default_factory=list)
    reasoning: str = ""
    format_issues: list[str] = field(default_factory=list)


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """Return unique items while keeping their original order."""
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _strip_markdown_fences(text: str) -> tuple[str, list[str]]:
    """Remove markdown fence lines while keeping the enclosed content."""
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


def _normalize_action(action_text: str) -> str:
    """Normalize whitespace inside one PDDL-style action."""
    inner = re.sub(r"\s+", " ", action_text[1:-1].strip())
    return f"({inner})"


def _strip_list_prefix(line: str) -> str:
    """Remove common bullet and numbering prefixes from one line."""
    previous = ""
    stripped = line.strip()
    while stripped != previous:
        previous = stripped
        stripped = LIST_PREFIX_PATTERN.sub("", stripped).strip()
    return stripped


def _split_reasoning_and_plan(text: str) -> tuple[str, str, list[str]]:
    """Split free-form reasoning from the plan-oriented portion of the text."""
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
            plan_text = "\n".join(remaining_lines).strip()

            issues = ["plan_section_marker_removed"]
            if reasoning:
                issues.append("reasoning_before_plan_removed")
            return reasoning, plan_text, issues

    for index, line in enumerate(lines):
        if ACTION_PATTERN.search(line):
            reasoning = "\n".join(lines[:index]).strip()
            plan_text = "\n".join(lines[index:]).strip()
            issues: list[str] = []
            if reasoning:
                issues.append("reasoning_before_plan_removed")
            return reasoning, plan_text, issues

    return text.strip(), text.strip(), []


def _extract_actions(text: str) -> tuple[list[str], list[str]]:
    """Extract PDDL-style actions from noisy text using simple heuristics."""
    actions: list[str] = []
    issues: list[str] = []

    for line in text.splitlines():
        stripped = _strip_list_prefix(line)
        if not stripped:
            continue

        matches = ACTION_PATTERN.findall(stripped)
        if not matches:
            continue

        remainder = ACTION_PATTERN.sub("", stripped).strip(" \t-:;,.")
        if remainder:
            issues.append("actions_embedded_in_text")

        actions.extend(_normalize_action(match) for match in matches)

    return actions, _dedupe_preserve_order(issues)


def parse_plan_text(raw_text: str) -> ParsedPlan:
    """Parse raw model text into a normalized benchmark plan payload."""
    if not raw_text or not raw_text.strip():
        return ParsedPlan(format_issues=["empty_output"])

    cleaned_text, fence_issues = _strip_markdown_fences(raw_text)
    reasoning, plan_text, split_issues = _split_reasoning_and_plan(cleaned_text)
    actions, extraction_issues = _extract_actions(plan_text or cleaned_text)
    format_issues = fence_issues + split_issues + extraction_issues

    if not actions:
        format_issues.append("no_parenthesized_actions_found")

    return ParsedPlan(
        actions=actions,
        reasoning=reasoning,
        format_issues=_dedupe_preserve_order(format_issues),
    )
