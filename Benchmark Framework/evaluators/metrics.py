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


def metrics_to_dict(metrics: RunMetrics) -> dict[str, object]:
    """Serialize a metric bundle into a plain dictionary."""
    return {
        "validity_at_1": metrics.validity_at_1,
        "validity_at_k": metrics.validity_at_k,
        "repair_success": metrics.repair_success,
        "iterations_to_valid": metrics.iterations_to_valid,
        "plan_length": metrics.plan_length,
        "error_type": metrics.error_type,
    }
