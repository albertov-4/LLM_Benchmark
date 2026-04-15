"""Shared parser scaffold."""

from dataclasses import dataclass, field


@dataclass
class ParsedPlan:
    actions: list[str] = field(default_factory=list)
    reasoning: str = ""
    format_issues: list[str] = field(default_factory=list)


def parse_plan_text(raw_text: str) -> ParsedPlan:
    """Placeholder parser to be aligned with the common benchmark format."""
    if not raw_text:
        return ParsedPlan(format_issues=["empty_output"])
    return ParsedPlan(actions=[], reasoning="")
