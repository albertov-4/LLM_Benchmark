"""Mock validator utilities for benchmark smoke tests."""

from __future__ import annotations


class MockValidator:
    """Simulate validator behavior with optional repair-loop support."""

    def __init__(
        self,
        *,
        valid_on_attempt: int = 1,
        accepted_actions: list[str] | None = None,
    ) -> None:
        self.valid_on_attempt = valid_on_attempt
        self.accepted_actions = accepted_actions or ["(move a b)"]
        self.call_count = 0

    def validate(self, domain_file: str, problem_file: str, plan_text: str) -> dict[str, object]:
        self.call_count += 1
        cleaned_plan = "\n".join(
            line.strip()
            for line in plan_text.splitlines()
            if line.strip()
        )

        if self.call_count >= self.valid_on_attempt and cleaned_plan in self.accepted_actions:
            return {
                "valid": True,
                "status": "valid",
                "error_type": None,
                "feedback_text": "Plan validated successfully.",
                "failed_step": None,
                "failed_action": None,
                "goal_satisfied": True,
                "plan_length": len([line for line in cleaned_plan.splitlines() if line.strip()]),
                "validation_time_ms": 1,
                "raw_validator_output": "mock-valid",
                "details": {
                    "domain_file": domain_file,
                    "problem_file": problem_file,
                    "validator_kind": "mock",
                },
            }

        return {
            "valid": False,
            "status": "invalid",
            "error_type": "invalid_precondition",
            "feedback_text": "Mock validator rejected the current action sequence.",
            "failed_step": 1,
            "failed_action": cleaned_plan.splitlines()[0] if cleaned_plan else None,
            "goal_satisfied": False,
            "plan_length": len([line for line in cleaned_plan.splitlines() if line.strip()]) or 0,
            "validation_time_ms": 1,
            "raw_validator_output": "mock-invalid",
            "details": {
                "domain_file": domain_file,
                "problem_file": problem_file,
                "validator_kind": "mock",
            },
        }


def build_mock_validator_for_suite() -> MockValidator:
    """Factory compatible with `run_suite(validator_factory=...)`."""
    return MockValidator(valid_on_attempt=1)
