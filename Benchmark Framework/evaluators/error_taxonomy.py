"""Shared error taxonomy for benchmark scoring."""

from enum import Enum


class ErrorType(str, Enum):
    FORMAT_ERROR = "format_error"
    PARSE_ERROR = "parse_error"
    INVALID_ACTION = "invalid_action"
    PRECONDITION_VIOLATION = "precondition_violation"
    GOAL_NOT_REACHED = "goal_not_reached"
    LOOPING_PLAN = "looping_plan"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"
