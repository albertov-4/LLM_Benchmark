"""Shared error taxonomy for benchmark scoring and validation."""

from enum import Enum


class ErrorType(str, Enum):
    """Controlled vocabulary for benchmark failures.

    The goal is to keep failure categories stable across models and tasks,
    while still being expressive enough for repair loops and analysis.
    """

    EMPTY_PLAN = "empty_plan"
    SYNTAX_ERROR = "syntax_error"
    PARSE_ERROR = "parse_error"
    UNKNOWN_ACTION = "unknown_action"
    INVALID_PRECONDITION = "invalid_precondition"
    UNSATISFIED_GOAL = "unsatisfied_goal"
    TIMEOUT = "timeout"
    VALIDATOR_CRASH = "validator_crash"
    VALIDATOR_UNAVAILABLE = "validator_unavailable"
    UNKNOWN = "unknown"


LOGICAL_ERROR_TYPES = frozenset(
    {
        ErrorType.EMPTY_PLAN,
        ErrorType.SYNTAX_ERROR,
        ErrorType.UNKNOWN_ACTION,
        ErrorType.INVALID_PRECONDITION,
        ErrorType.UNSATISFIED_GOAL,
    }
)

TECHNICAL_ERROR_TYPES = frozenset(
    {
        ErrorType.PARSE_ERROR,
        ErrorType.TIMEOUT,
        ErrorType.VALIDATOR_CRASH,
        ErrorType.VALIDATOR_UNAVAILABLE,
    }
)
