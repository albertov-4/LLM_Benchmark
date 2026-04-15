"""Shared parser scaffold.

This parser is intentionally simple:
- it is useful immediately
- it stays readable
- it can later be replaced with a stronger domain-aware parser
"""

from dataclasses import dataclass, field


@dataclass
class ParsedPlan:
    actions: list[str] = field(default_factory=list)
    reasoning: str = ""
    format_issues: list[str] = field(default_factory=list)


def _extract_parenthesized_lines(text: str) -> list[str]:
    """Keep only lines that already look like PDDL actions."""
    actions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("(") and stripped.endswith(")"):
            actions.append(stripped)
    return actions


def parse_plan_text(raw_text: str) -> ParsedPlan:
    """Parse raw model text into a lightweight structured plan.

    Pseudocode for future upgrades:
    1. Remove markdown fences and obvious boilerplate
    2. Split reasoning from final answer when the protocol allows rationale
    3. Extract plan lines using multiple heuristics
    4. Return both actions and format issues
    """
    if not raw_text:
        return ParsedPlan(format_issues=["empty_output"])

    actions = _extract_parenthesized_lines(raw_text)
    format_issues: list[str] = []

    if not actions:
        format_issues.append("no_parenthesized_actions_found")

    return ParsedPlan(actions=actions, reasoning="", format_issues=format_issues)
