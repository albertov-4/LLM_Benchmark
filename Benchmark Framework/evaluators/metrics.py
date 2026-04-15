"""Benchmark metrics scaffold."""

from dataclasses import dataclass


@dataclass
class RunMetrics:
    validity_at_1: bool
    validity_at_k: bool
    repair_success: bool
    iterations_to_valid: int | None
    plan_length: int | None
    error_type: str | None


def compute_core_metrics(valid: bool, iterations: int, plan_length: int | None, error_type: str | None) -> RunMetrics:
    """Compute the common metrics used across all models."""
    return RunMetrics(
        validity_at_1=valid and iterations == 1,
        validity_at_k=valid,
        repair_success=valid and iterations > 1,
        iterations_to_valid=iterations if valid else None,
        plan_length=plan_length,
        error_type=error_type,
    )
