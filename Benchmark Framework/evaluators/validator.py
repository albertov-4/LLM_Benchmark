"""Validator interface scaffold."""

from dataclasses import dataclass


@dataclass
class ValidationResult:
    valid: bool
    error_type: str | None = None
    error_message: str | None = None


class ValidatorAdapter:
    """Common interface for a symbolic validator."""

    def validate(self, domain_file: str, problem_file: str, plan_text: str) -> ValidationResult:
        raise NotImplementedError
