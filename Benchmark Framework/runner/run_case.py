"""Single benchmark case scaffold."""

from dataclasses import dataclass


@dataclass
class TaskSpec:
    task_family: str
    tier: str
    instance_id: str
    domain_file: str
    problem_file: str


@dataclass
class ProtocolSpec:
    protocol_id: str
    max_iterations: int
    require_final_plan_only: bool


@dataclass
class ResultRecord:
    model_id: str
    protocol_id: str
    task_family: str
    tier: str
    instance_id: str
    valid: bool
    iterations: int
    error_type: str | None
    raw_output_path: str
    parsed_output_path: str
    scored_output_path: str


def run_case() -> None:
    """Placeholder entry point for a single benchmark execution."""
    raise NotImplementedError("Implement case execution using adapters and evaluators.")
