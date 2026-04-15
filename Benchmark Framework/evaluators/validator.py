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
        """Pseudocode contract for a symbolic validator.

        Expected future implementation:
        1. materialize `plan_text` into a temporary file if needed
        2. invoke the external validator
        3. normalize its response into `ValidationResult`
        """
        raise NotImplementedError


def build_feedback_from_validation(result: ValidationResult) -> str:
    """Turn a validation failure into a repair-oriented feedback string."""
    if result.valid:
        return "The previous plan validated successfully."

    return (
        "The previous plan is invalid.\n"
        f"Error type: {result.error_type or 'unknown'}\n"
        f"Validator message: {result.error_message or 'No extra details provided.'}\n"
        "Please return a corrected action sequence."
    )
