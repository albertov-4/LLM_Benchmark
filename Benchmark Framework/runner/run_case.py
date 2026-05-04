"""Single benchmark case scaffold.

This file contains guided pseudocode for the smallest benchmark unit:
one model, one protocol, one task instance.
"""

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, TypedDict


@dataclass(slots=True)
class TaskSpec:
    task_family: str
    tier: str
    instance_id: str
    domain_file: str
    problem_file: str

    @property
    def task_id(self) -> str:
        """Return a stable benchmark id for one concrete task instance."""
        return f"{self.task_family}_{self.tier}_{self.instance_id}"


@dataclass(slots=True)
class ProtocolSpec:
    protocol_id: str
    max_iterations: int
    require_final_plan_only: bool
    include_external_feedback: bool = False


@dataclass(slots=True)
class ResultRecord:
    """Normalized result for one model x protocol x task run."""

    model_id: str
    task_id: str
    protocol_id: str
    task_family: str
    tier: str
    instance_id: str
    solved: bool
    iterations_used: int
    max_iterations: int
    stopped_by_iteration_limit: bool
    raw_output: str | None = None
    parsed_plan: dict[str, Any] | None = None
    validation_result: dict[str, Any] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_output_path: str | None = None
    parsed_output_path: str | None = None
    scored_output_path: str | None = None


class CasePayload(TypedDict):
    solved: bool
    iterations_used: int
    max_iterations: int
    stopped_by_iteration_limit: bool
    raw_output: str | None
    parsed_plan: dict[str, Any] | None
    validation_result: dict[str, Any] | None
    metrics: dict[str, Any]
    raw_generations: list[dict[str, Any]]


@lru_cache(maxsize=None)
def _load_framework_module(module_key: str, relative_path: str):
    """Load a sibling framework module without requiring package installation."""
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _normalize_payload(value: Any) -> dict[str, Any] | None:
    """Convert dataclass-like payloads into plain dictionaries."""
    if value is None:
        return None
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
    raise TypeError(f"Unsupported payload type: {type(value)!r}")


def _json_safe(value: Any) -> Any:
    """Convert benchmark payloads into JSON-serializable values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _build_artifact_paths(
    output_root: Path,
    model_id: str,
    protocol_id: str,
    task_spec: TaskSpec,
) -> tuple[Path, Path, Path]:
    """Return the canonical raw/parsed/scored paths for one run."""
    relative_dir = Path(model_id) / protocol_id / task_spec.task_family / task_spec.tier
    file_name = f"{task_spec.instance_id}.json"
    return (
        output_root / "raw" / relative_dir / file_name,
        output_root / "parsed" / relative_dir / file_name,
        output_root / "scored" / relative_dir / file_name,
    )


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Persist one artifact as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _parse_generation_output(raw_text: str) -> dict[str, Any]:
    """Parse raw model text using the shared benchmark parser."""
    parser_module = _load_framework_module(
        "benchmark_framework_parser",
        "evaluators/parser.py",
    )
    parsed_plan = parser_module.parse_plan_text(raw_text)
    normalized = _normalize_payload(parsed_plan)
    return normalized or {"actions": [], "reasoning": "", "format_issues": []}


def _build_parse_failure_result(
    raw_text: str,
    parsed_plan: dict[str, Any],
) -> dict[str, Any]:
    """Create a normalized validation-style error for parser failures."""
    format_issues = parsed_plan.get("format_issues", [])
    has_text = bool(raw_text and raw_text.strip())

    if has_text:
        error_type = "syntax_error"
        feedback_text = (
            "No valid PDDL action lines were detected. "
            "Please return only parenthesized actions, one per line."
        )
    else:
        error_type = "empty_plan"
        feedback_text = (
            "The model did not return any plan. "
            "Please provide only a sequence of actions."
        )

    return {
        "valid": False,
        "status": "parse_error",
        "error_type": error_type,
        "feedback_text": feedback_text,
        "failed_step": None,
        "failed_action": None,
        "goal_satisfied": None,
        "plan_length": 0,
        "validation_time_ms": None,
        "raw_validator_output": None,
        "details": {"format_issues": format_issues},
    }


def _run_validator(
    validator,
    task_spec: TaskSpec,
    plan_text: str,
) -> dict[str, Any]:
    """Validate a parsed plan and normalize the validator response."""
    try:
        validation = validator.validate(
            task_spec.domain_file,
            task_spec.problem_file,
            plan_text,
        )
    except Exception as exc:  # pragma: no cover - defensive scaffold behavior
        return {
            "valid": False,
            "status": "validator_error",
            "error_type": "validator_crash",
            "feedback_text": "The validator crashed while checking the plan.",
            "failed_step": None,
            "failed_action": None,
            "goal_satisfied": None,
            "plan_length": None,
            "validation_time_ms": None,
            "raw_validator_output": str(exc),
            "details": {"exception_type": type(exc).__name__},
        }

    normalized = _normalize_payload(validation)
    if normalized is not None:
        return normalized

    return {
        "valid": False,
        "status": "validator_error",
        "error_type": "validator_unavailable",
        "feedback_text": "The validator did not return a normalized result.",
        "failed_step": None,
        "failed_action": None,
        "goal_satisfied": None,
        "plan_length": None,
        "validation_time_ms": None,
        "raw_validator_output": None,
        "details": {},
    }


def build_task_spec(task_family: str, tier: str, instance_id: str, domain_file: str, problem_file: str) -> TaskSpec:
    """Create a normalized task description from the folder-based benchmark structure."""
    return TaskSpec(
        task_family=task_family,
        tier=tier,
        instance_id=instance_id,
        domain_file=domain_file,
        problem_file=problem_file,
    )


def load_task_inputs(task_spec: TaskSpec) -> dict[str, str]:
    """Read the domain and problem text from disk.

    This helper is already real code because it is stable and easy to maintain.
    """
    return {
        "domain_text": Path(task_spec.domain_file).read_text(encoding="utf-8"),
        "problem_text": Path(task_spec.problem_file).read_text(encoding="utf-8"),
    }


def build_messages(
    task_spec: TaskSpec,
    task_inputs: dict[str, str],
    protocol_spec: ProtocolSpec,
    system_prompt: str = "",
    domain_prompt: str = "",
    feedback_messages: list[str] | None = None,
) -> list[dict[str, str]]:
    """Build a chat-style message list shared across adapters.

    Pseudocode decisions:
    - always keep a chat-like representation
    - adapter implementations can convert messages to provider-specific format
    """
    feedback_messages = feedback_messages or []
    protocol_instruction_lines: list[str] = []

    if protocol_spec.require_final_plan_only:
        protocol_instruction_lines.append(
            "FINAL ANSWER FORMAT: return only parenthesized PDDL actions, one per line."
        )
        protocol_instruction_lines.append(
            "Do not include explanations, headings, markdown fences, bullets, or numbering in the final answer."
        )
    else:
        protocol_instruction_lines.append(
            "You may include brief reasoning, but end with a clean action sequence."
        )
        protocol_instruction_lines.append(
            "Keep each action on its own line in parenthesized PDDL form so the parser can extract it."
        )

    protocol_instructions = "\n".join(protocol_instruction_lines)

    user_content = (
        f"TASK FAMILY: {task_spec.task_family}\n"
        f"DIFFICULTY: {task_spec.tier}\n"
        f"INSTANCE: {task_spec.instance_id}\n\n"
        f"{domain_prompt}\n\n"
        f"{protocol_instructions}\n\n"
        f"=== DOMAIN ===\n{task_inputs['domain_text']}\n\n"
        f"=== PROBLEM ===\n{task_inputs['problem_text']}\n"
    )

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_content})

    for feedback in feedback_messages:
        messages.append({"role": "user", "content": feedback})

    return messages


def build_repair_feedback(
    validation_result: dict[str, Any],
    feedback_prompt: str = "",
) -> str:
    """Convert validator output into a user-facing repair message.

    In a future implementation this could delegate to a domain-specific
    feedback module.
    """
    status = validation_result.get("status", "invalid")
    error_type = validation_result.get("error_type", "unknown")
    feedback_text = validation_result.get("feedback_text", "Unknown validation error.")
    failed_step = validation_result.get("failed_step")
    failed_action = validation_result.get("failed_action")

    lines: list[str] = []
    if feedback_prompt.strip():
        lines.append(feedback_prompt.strip())

    lines.extend(
        [
            "The previous plan did not validate.",
            f"Validation status: {status}",
            f"Error type: {error_type}",
        ]
    )
    if failed_step is not None:
        lines.append(f"Failed step: {failed_step}")
    if failed_action:
        lines.append(f"Failed action: {failed_action}")
    lines.append(f"Feedback: {feedback_text}")
    lines.append("Please provide a corrected action sequence.")
    return "\n".join(lines)


def run_generation_loop(
    adapter,
    validator,
    task_spec: TaskSpec,
    task_inputs: dict[str, str],
    protocol_spec: ProtocolSpec,
    system_prompt: str = "",
    domain_prompt: str = "",
    feedback_prompt: str = "",
) -> CasePayload:
    """Core pseudocode loop for one benchmark case.

    Expected adapter contract:
    - adapter.generate(messages) -> {"raw_text": "...", ...}

    Expected validator contract:
    - validator.validate(domain_file, problem_file, plan_text) ->
      ValidationResult-like payload with at least:
      `valid`, `status`, `error_type`, `feedback_text`
    """
    feedback_messages: list[str] = []
    raw_generations: list[dict[str, Any]] = []
    last_parsed_plan: dict[str, Any] | None = None
    last_validation_result: dict[str, Any] | None = None

    for iteration in range(1, protocol_spec.max_iterations + 1):
        messages = build_messages(
            task_spec=task_spec,
            task_inputs=task_inputs,
            protocol_spec=protocol_spec,
            system_prompt=system_prompt,
            domain_prompt=domain_prompt,
            feedback_messages=feedback_messages,
        )

        generation = adapter.generate(messages)
        if not isinstance(generation, dict):
            generation = {"raw_text": str(generation)}
        raw_generations.append(generation)

        raw_text = generation.get("raw_text", "")
        raw_text = raw_text if isinstance(raw_text, str) else str(raw_text)

        parsed_plan = _parse_generation_output(raw_text)
        last_parsed_plan = parsed_plan

        actions = parsed_plan.get("actions", [])
        if not isinstance(actions, list) or not actions:
            validation_result = _build_parse_failure_result(raw_text, parsed_plan)
            last_validation_result = validation_result
            if protocol_spec.include_external_feedback:
                feedback_messages.append(
                    build_repair_feedback(validation_result, feedback_prompt=feedback_prompt)
                )
            continue

        plan_text = "\n".join(action for action in actions if isinstance(action, str))
        validation_result = _run_validator(validator, task_spec, plan_text)
        last_validation_result = validation_result

        if bool(validation_result.get("valid", False)):
            return {
                "solved": True,
                "iterations_used": iteration,
                "max_iterations": protocol_spec.max_iterations,
                "stopped_by_iteration_limit": False,
                "raw_output": raw_text,
                "parsed_plan": parsed_plan,
                "validation_result": validation_result,
                "metrics": {},
                "raw_generations": raw_generations,
            }

        if protocol_spec.include_external_feedback:
            feedback_messages.append(
                build_repair_feedback(validation_result, feedback_prompt=feedback_prompt)
            )

    final_raw_output = None
    if raw_generations:
        final_raw_output = raw_generations[-1].get("raw_text")

    return {
        "solved": False,
        "iterations_used": len(raw_generations),
        "max_iterations": protocol_spec.max_iterations,
        "stopped_by_iteration_limit": len(raw_generations) >= protocol_spec.max_iterations,
        "raw_output": final_raw_output,
        "parsed_plan": last_parsed_plan,
        "validation_result": last_validation_result,
        "metrics": {},
        "raw_generations": raw_generations,
    }


def build_result_record(
    model_id: str,
    protocol_spec: ProtocolSpec,
    task_spec: TaskSpec,
    solved: bool,
    iterations_used: int,
    stopped_by_iteration_limit: bool,
    raw_output: str | None = None,
    parsed_plan: dict[str, Any] | None = None,
    validation_result: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    raw_output_path: str | None = None,
    parsed_output_path: str | None = None,
    scored_output_path: str | None = None,
) -> ResultRecord:
    """Normalize the final case result into a serializable record."""
    return ResultRecord(
        model_id=model_id,
        task_id=task_spec.task_id,
        protocol_id=protocol_spec.protocol_id,
        task_family=task_spec.task_family,
        tier=task_spec.tier,
        instance_id=task_spec.instance_id,
        solved=solved,
        iterations_used=iterations_used,
        max_iterations=protocol_spec.max_iterations,
        stopped_by_iteration_limit=stopped_by_iteration_limit,
        raw_output=raw_output,
        parsed_plan=parsed_plan,
        validation_result=validation_result,
        metrics=metrics or {},
        raw_output_path=raw_output_path,
        parsed_output_path=parsed_output_path,
        scored_output_path=scored_output_path,
    )


def run_case(
    model_id: str,
    adapter,
    validator,
    task_spec: TaskSpec,
    protocol_spec: ProtocolSpec,
    system_prompt: str = "",
    domain_prompt: str = "",
    feedback_prompt: str = "",
    output_root: str | Path | None = None,
) -> ResultRecord:
    """Pseudocode entry point for one benchmark case.

    Current implemented flow:
    1. load task files
    2. run the generation/repair loop
    3. parse and validate each attempt
    4. compute core metrics from the normalized run payload
    5. return a normalized record

    Still left as future work:
    - support richer attempt history serialization
    """
    task_inputs = load_task_inputs(task_spec)

    case_payload = run_generation_loop(
        adapter=adapter,
        validator=validator,
        task_spec=task_spec,
        task_inputs=task_inputs,
        protocol_spec=protocol_spec,
        system_prompt=system_prompt,
        domain_prompt=domain_prompt,
        feedback_prompt=feedback_prompt,
    )

    metrics_module = _load_framework_module(
        "benchmark_framework_metrics",
        "evaluators/metrics.py",
    )
    metrics = metrics_module.compute_core_metrics(
        solved=case_payload["solved"],
        iterations_used=case_payload["iterations_used"],
        max_iterations=protocol_spec.max_iterations,
        stopped_by_iteration_limit=case_payload["stopped_by_iteration_limit"],
        parsed_plan=case_payload["parsed_plan"],
        validation_result=case_payload["validation_result"],
    )
    metrics_dict = metrics_module.metrics_to_dict(metrics)
    raw_output_path: str | None = None
    parsed_output_path: str | None = None
    scored_output_path: str | None = None

    if output_root is not None:
        output_root_path = Path(output_root)
        raw_path, parsed_path, scored_path = _build_artifact_paths(
            output_root=output_root_path,
            model_id=model_id,
            protocol_id=protocol_spec.protocol_id,
            task_spec=task_spec,
        )

        raw_payload = {
            "model_id": model_id,
            "task_id": task_spec.task_id,
            "protocol_id": protocol_spec.protocol_id,
            "task_family": task_spec.task_family,
            "tier": task_spec.tier,
            "instance_id": task_spec.instance_id,
            "solved": case_payload["solved"],
            "iterations_used": case_payload["iterations_used"],
            "raw_output": case_payload["raw_output"],
            "raw_generations": case_payload["raw_generations"],
        }
        parsed_payload = {
            "model_id": model_id,
            "task_id": task_spec.task_id,
            "protocol_id": protocol_spec.protocol_id,
            "task_family": task_spec.task_family,
            "tier": task_spec.tier,
            "instance_id": task_spec.instance_id,
            "parsed_plan": case_payload["parsed_plan"],
            "validation_result": case_payload["validation_result"],
        }

        _write_json_artifact(raw_path, raw_payload)
        _write_json_artifact(parsed_path, parsed_payload)

        raw_output_path = str(raw_path)
        parsed_output_path = str(parsed_path)
        scored_output_path = str(scored_path)

    result_record = build_result_record(
        model_id=model_id,
        protocol_spec=protocol_spec,
        task_spec=task_spec,
        solved=case_payload["solved"],
        iterations_used=case_payload["iterations_used"],
        stopped_by_iteration_limit=case_payload["stopped_by_iteration_limit"],
        raw_output=case_payload["raw_output"],
        parsed_plan=case_payload["parsed_plan"],
        validation_result=case_payload["validation_result"],
        metrics=metrics_dict,
        raw_output_path=raw_output_path,
        parsed_output_path=parsed_output_path,
        scored_output_path=scored_output_path,
    )

    if output_root is not None and scored_output_path is not None:
        _write_json_artifact(Path(scored_output_path), asdict(result_record))

    return result_record
