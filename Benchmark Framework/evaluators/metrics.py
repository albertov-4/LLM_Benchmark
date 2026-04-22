"""Benchmark metrics scaffold."""

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RunMetrics:
    validity_at_1: bool
    validity_at_k: bool
    repair_success: bool
    iterations_to_valid: int | None
    plan_length: int | None
    error_type: str | None
    hit_iteration_limit: bool


def _extract_plan_length(
    parsed_plan: dict[str, Any] | None,
    validation_result: dict[str, Any] | None,
) -> int | None:
    """Infer plan length from the most reliable available source."""
    if validation_result:
        maybe_length = validation_result.get("plan_length")
        if isinstance(maybe_length, int):
            return maybe_length

    if not parsed_plan:
        return None

    actions = parsed_plan.get("actions")
    if isinstance(actions, list):
        return len(actions)

    return None


def _extract_error_type(validation_result: dict[str, Any] | None) -> str | None:
    """Read the final normalized error type from validator output."""
    if not validation_result:
        return None

    error_type = validation_result.get("error_type")
    return error_type if isinstance(error_type, str) else None


def compute_core_metrics(
    *,
    solved: bool,
    iterations_used: int,
    max_iterations: int,
    stopped_by_iteration_limit: bool,
    parsed_plan: dict[str, Any] | None = None,
    validation_result: dict[str, Any] | None = None,
) -> RunMetrics:
    """Compute the common benchmark metrics from one normalized run.

    This function intentionally consumes run-level fields rather than raw model
    output, so every model is scored on the same basis.
    """
    plan_length = _extract_plan_length(parsed_plan, validation_result)
    error_type = _extract_error_type(validation_result)

    return RunMetrics(
        validity_at_1=solved and iterations_used == 1,
        validity_at_k=solved,
        repair_success=solved and iterations_used > 1,
        iterations_to_valid=iterations_used if solved else None,
        plan_length=plan_length,
        error_type=error_type,
        hit_iteration_limit=stopped_by_iteration_limit or iterations_used > max_iterations,
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
        "hit_iteration_limit": metrics.hit_iteration_limit,
    }
