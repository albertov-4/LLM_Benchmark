"""Suite result normalization and aggregation."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def _normalize_record(value: Any) -> dict[str, Any]:
    """Convert dataclass-like results into plain dictionaries."""
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if hasattr(value, "__dict__"):
        return {
            key: item
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    raise TypeError(f"Unsupported record type: {type(value)!r}")


def _build_suite_result_summary(result: dict[str, Any]) -> dict[str, Any]:
    """Keep suite-level results compact; details live in per-job artifacts."""
    return {
        "model_id": result.get("model_id"),
        "task_id": result.get("task_id"),
        "protocol_id": result.get("protocol_id"),
        "task_family": result.get("task_family"),
        "tier": result.get("tier"),
        "instance_id": result.get("instance_id"),
        "solved": result.get("solved"),
        "iterations_used": result.get("iterations_used"),
        "max_iterations": result.get("max_iterations"),
        "stopped_by_iteration_limit": result.get("stopped_by_iteration_limit"),
        "generation_time_seconds": result.get("generation_time_seconds"),
        "metrics": result.get("metrics", {}),
        "raw_output_path": result.get("raw_output_path"),
        "parsed_output_path": result.get("parsed_output_path"),
        "scored_output_path": result.get("scored_output_path"),
    }


def _new_aggregate_bucket() -> dict[str, Any]:
    """Create one aggregation bucket."""
    return {
        "num_runs": 0,
        "num_solved": 0,
        "solve_rate": 0.0,
        "avg_iterations_used": 0.0,
        "error_counts": {},
        "_iterations_total": 0,
    }


def _update_aggregate_bucket(bucket: dict[str, Any], record: dict[str, Any]) -> None:
    """Update one aggregation bucket in place."""
    solved = bool(record.get("solved", False))
    iterations_used = int(record.get("iterations_used", 0) or 0)

    bucket["num_runs"] += 1
    bucket["_iterations_total"] += iterations_used
    if solved:
        bucket["num_solved"] += 1

    validation_result = record.get("validation_result")
    error_type = None
    if isinstance(validation_result, dict):
        error_type = validation_result.get("error_type")

    if not error_type:
        metrics = record.get("metrics")
        if isinstance(metrics, dict):
            error_type = metrics.get("error_type")

    if error_type:
        error_counts = bucket["error_counts"]
        error_counts[error_type] = error_counts.get(error_type, 0) + 1


def _finalize_aggregate_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    """Finalize one aggregation bucket for output."""
    num_runs = bucket["num_runs"]
    iterations_total = bucket.pop("_iterations_total", 0)
    bucket["solve_rate"] = (bucket["num_solved"] / num_runs) if num_runs else 0.0
    bucket["avg_iterations_used"] = (iterations_total / num_runs) if num_runs else 0.0
    return bucket


def aggregate_suite_results(results: list[dict[str, Any] | Any]) -> dict[str, Any]:
    """Aggregate normalized run results into benchmark summaries."""
    normalized_results = [_normalize_record(result) for result in results]

    overall = _new_aggregate_bucket()
    by_model: dict[str, dict[str, Any]] = {}
    by_protocol: dict[str, dict[str, Any]] = {}
    by_tier: dict[str, dict[str, Any]] = {}

    for record in normalized_results:
        model_id = str(record.get("model_id", "unknown-model"))
        protocol_id = str(record.get("protocol_id", "unknown-protocol"))
        tier = str(record.get("tier", "unknown-tier"))

        _update_aggregate_bucket(overall, record)
        _update_aggregate_bucket(by_model.setdefault(model_id, _new_aggregate_bucket()), record)
        _update_aggregate_bucket(by_protocol.setdefault(protocol_id, _new_aggregate_bucket()), record)
        _update_aggregate_bucket(by_tier.setdefault(tier, _new_aggregate_bucket()), record)

    return {
        "overall": _finalize_aggregate_bucket(overall),
        "by_model": {key: _finalize_aggregate_bucket(value) for key, value in by_model.items()},
        "by_protocol": {key: _finalize_aggregate_bucket(value) for key, value in by_protocol.items()},
        "by_tier": {key: _finalize_aggregate_bucket(value) for key, value in by_tier.items()},
    }
