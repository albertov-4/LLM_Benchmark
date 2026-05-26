"""Single benchmark case execution.

This module runs the smallest benchmark unit: one model, one protocol,
one task instance.
"""

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from time import perf_counter
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
    include_chain_of_thought: bool = False


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
    generation_time_seconds: float = 0.0
    raw_output: str | None = None
    parsed_plan: dict[str, Any] | None = None
    validation_result: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    raw_output_path: str | None = None
    parsed_output_path: str | None = None
    scored_output_path: str | None = None


class CasePayload(TypedDict):
    solved: bool
    iterations_used: int
    max_iterations: int
    stopped_by_iteration_limit: bool
    generation_time_seconds: float
    raw_output: str | None
    parsed_plan: dict[str, Any] | None
    validation_result: dict[str, Any] | None
    metrics: dict[str, Any]
    raw_generations: list[dict[str, Any]]
    attempts: list[dict[str, Any]]


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
    run_id: str,
    model_id: str,
    protocol_id: str,
    task_spec: TaskSpec,
) -> tuple[Path, Path, Path]:
    """Return the canonical raw/parsed/scored paths for one run."""
    relative_dir = Path(model_id) / protocol_id / task_spec.task_family / task_spec.tier
    file_name = f"{task_spec.instance_id}.json"
    return (
        output_root / "raw" / run_id / relative_dir / file_name,
        output_root / "parsed" / run_id / relative_dir / file_name,
        output_root / "scored" / run_id / relative_dir / file_name,
    )


def _write_json_artifact(path: Path, payload: dict[str, Any]) -> None:
    """Persist one artifact as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _build_raw_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only prompt and model-generation data for raw artifacts."""
    raw_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        generation = dict(attempt.get("generation", {}))
        generation.pop("model_id", None)
        raw_attempts.append(
            {
                "iteration": attempt.get("iteration"),
                "generation_time_seconds": attempt.get("generation_time_seconds"),
                "messages": attempt.get("messages", []),
                "generation": generation,
            }
        )
    return raw_attempts


def _build_parsed_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only parser output for parsed artifacts."""
    parsed_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        parsed_attempts.append(
            {
                "iteration": attempt.get("iteration"),
                "generation_time_seconds": attempt.get("generation_time_seconds"),
                "parsed_plan": attempt.get("parsed_plan"),
            }
        )
    return parsed_attempts


def _build_scored_attempts(attempts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only validation and repair data for scored artifacts."""
    scored_attempts: list[dict[str, Any]] = []
    for attempt in attempts:
        scored_attempts.append(
            {
                "iteration": attempt.get("iteration"),
                "generation_time_seconds": attempt.get("generation_time_seconds"),
                "validation_result": attempt.get("validation_result"),
                "first_valid_prefix_length": attempt.get("first_valid_prefix_length"),
                "first_valid_plan_text": attempt.get("first_valid_plan_text"),
                "final_plan_valid": attempt.get("final_plan_valid"),
                "extra_actions_after_first_valid": attempt.get("extra_actions_after_first_valid"),
                "feedback_to_next_iteration": attempt.get("feedback_to_next_iteration"),
            }
        )
    return scored_attempts


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
    except Exception as exc:  # pragma: no cover - defensive validator handling
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


def _validate_action_prefixes(
    validator,
    task_spec: TaskSpec,
    actions: list[str],
) -> dict[str, Any]:
    """Validate every non-empty action prefix and summarize the outcome."""
    first_valid_prefix: dict[str, Any] | None = None
    final_prefix: dict[str, Any] | None = None

    for prefix_length in range(1, len(actions) + 1):
        prefix_actions = actions[:prefix_length]
        plan_text = "\n".join(prefix_actions)
        validation_result = _run_validator(validator, task_spec, plan_text)
        prefix_record = {
            "prefix_length": prefix_length,
            "actions": prefix_actions,
            "plan_text": plan_text,
            "validation_result": validation_result,
            "valid": bool(validation_result.get("valid", False)),
            "goal_satisfied": validation_result.get("goal_satisfied"),
            "error_type": validation_result.get("error_type"),
            "raw_validator_output": validation_result.get("raw_validator_output"),
        }
        final_prefix = prefix_record

        if first_valid_prefix is None and prefix_record["valid"]:
            first_valid_prefix = prefix_record

    first_valid_prefix_length = (
        int(first_valid_prefix["prefix_length"])
        if first_valid_prefix is not None
        else None
    )
    first_valid_plan_text = (
        str(first_valid_prefix["plan_text"])
        if first_valid_prefix is not None
        else None
    )
    final_validation_result = (
        final_prefix["validation_result"]
        if final_prefix is not None
        else None
    )
    synthetic_validation_result = (
        first_valid_prefix["validation_result"]
        if first_valid_prefix is not None
        else final_validation_result
    )

    return {
        "validation_result": synthetic_validation_result,
        "first_valid_prefix_length": first_valid_prefix_length,
        "first_valid_plan_text": first_valid_plan_text,
        "final_plan_valid": bool(final_prefix["valid"]) if final_prefix is not None else False,
        "extra_actions_after_first_valid": (
            len(actions) - first_valid_prefix_length
            if first_valid_prefix_length is not None
            else None
        ),
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
    """Build a chat-style message list shared across adapters."""
    feedback_messages = feedback_messages or []
    protocol_instruction_lines: list[str] = []

    if protocol_spec.require_final_plan_only:
        if protocol_spec.include_chain_of_thought:
            protocol_instruction_lines.append(
                "You may reason internally, but do not include reasoning in the final answer."
            )
        protocol_instruction_lines.append(
            "FINAL ANSWER FORMAT: return only parenthesized PDDL actions, one per line."
        )
        protocol_instruction_lines.append(
            "Do not include explanations, headings, markdown fences, bullets, or numbering in the final answer."
        )
    else:
        if protocol_spec.include_chain_of_thought:
            protocol_instruction_lines.append(
                "You may include a brief rationale before the final plan."
            )
            protocol_instruction_lines.append(
                "Keep the rationale concise and focused on action applicability and goal achievement."
            )
        else:
            protocol_instruction_lines.append(
                "Do not include reasoning unless it is necessary to make the final plan understandable."
            )
        protocol_instruction_lines.append(
            "End with a clean final action sequence, with each action on its own line in parenthesized PDDL form."
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
    model_id: str = "",
    system_prompt: str = "",
    domain_prompt: str = "",
    feedback_prompt: str = "",
) -> CasePayload:
    """Run the generation, parsing, validation and repair loop for one case.

    Expected adapter contract:
    - adapter.generate(messages) -> {"raw_text": "...", ...}

    Expected validator contract:
    - validator.validate(domain_file, problem_file, plan_text) ->
      ValidationResult-like payload with at least:
      `valid`, `status`, `error_type`, `feedback_text`
    """
    feedback_messages: list[str] = []
    raw_generations: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    last_parsed_plan: dict[str, Any] | None = None
    last_validation_result: dict[str, Any] | None = None
    total_generation_time_seconds = 0.0
    model_label = model_id or getattr(adapter, "model_id", "unknown-model")
    task_label = f"{task_spec.task_family}/{task_spec.tier}/{task_spec.instance_id}"

    for iteration in range(1, protocol_spec.max_iterations + 1):
        messages = build_messages(
            task_spec=task_spec,
            task_inputs=task_inputs,
            protocol_spec=protocol_spec,
            system_prompt=system_prompt,
            domain_prompt=domain_prompt,
            feedback_messages=feedback_messages,
        )

        print(
            f"[GEN START] model={model_label} protocol={protocol_spec.protocol_id} "
            f"task={task_label} iteration={iteration}",
            flush=True,
        )
        generation_start = perf_counter()
        try:
            generation = adapter.generate(messages)
        except Exception as exc:
            elapsed_seconds = perf_counter() - generation_start
            print(
                f"[GEN ERROR] model={model_label} protocol={protocol_spec.protocol_id} "
                f"task={task_label} iteration={iteration} elapsed={elapsed_seconds:.1f}s "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            raise
        generation_elapsed = perf_counter() - generation_start
        if not isinstance(generation, dict):
            generation = {"raw_text": str(generation)}
        else:
            generation = dict(generation)
        generation["generation_time_seconds"] = generation_elapsed
        total_generation_time_seconds += generation_elapsed
        raw_text_preview = generation.get("raw_text", "")
        raw_text_length = len(raw_text_preview) if isinstance(raw_text_preview, str) else len(str(raw_text_preview))
        print(
            f"[GEN DONE] model={model_label} protocol={protocol_spec.protocol_id} "
            f"task={task_label} iteration={iteration} elapsed={generation_elapsed:.1f}s "
            f"raw_chars={raw_text_length}",
            flush=True,
        )
        raw_generations.append(generation)

        raw_text = generation.get("raw_text", "")
        raw_text = raw_text if isinstance(raw_text, str) else str(raw_text)

        parsed_plan = _parse_generation_output(raw_text)
        last_parsed_plan = parsed_plan
        actions = parsed_plan.get("actions", [])
        action_count = len(actions) if isinstance(actions, list) else 0
        print(
            f"[PARSE DONE] model={model_label} protocol={protocol_spec.protocol_id} "
            f"task={task_label} iteration={iteration} actions={action_count}",
            flush=True,
        )
        attempt_record: dict[str, Any] = {
            "iteration": iteration,
            "generation_time_seconds": generation_elapsed,
            "messages": messages,
            "generation": generation,
            "raw_output": raw_text,
            "parsed_plan": parsed_plan,
            "validation_result": None,
            "first_valid_prefix_length": None,
            "first_valid_plan_text": None,
            "final_plan_valid": False,
            "extra_actions_after_first_valid": None,
            "feedback_to_next_iteration": None,
        }

        if not isinstance(actions, list) or not actions:
            validation_result = _build_parse_failure_result(raw_text, parsed_plan)
            last_validation_result = validation_result
            attempt_record["validation_result"] = validation_result
            print(
                f"[VALIDATE SKIP] model={model_label} protocol={protocol_spec.protocol_id} "
                f"task={task_label} iteration={iteration} "
                f"error={validation_result.get('error_type')}",
                flush=True,
            )
            if protocol_spec.include_external_feedback:
                feedback = build_repair_feedback(validation_result, feedback_prompt=feedback_prompt)
                feedback_messages.append(feedback)
                attempt_record["feedback_to_next_iteration"] = feedback
            attempts.append(attempt_record)
            continue

        normalized_actions = [action for action in actions if isinstance(action, str)]
        print(
            f"[VALIDATE START] model={model_label} protocol={protocol_spec.protocol_id} "
            f"task={task_label} iteration={iteration} prefixes={len(normalized_actions)}",
            flush=True,
        )
        prefix_validation_summary = _validate_action_prefixes(
            validator=validator,
            task_spec=task_spec,
            actions=normalized_actions,
        )
        validation_result = prefix_validation_summary["validation_result"]
        last_validation_result = validation_result
        attempt_record["validation_result"] = validation_result
        attempt_record["first_valid_prefix_length"] = prefix_validation_summary["first_valid_prefix_length"]
        attempt_record["first_valid_plan_text"] = prefix_validation_summary["first_valid_plan_text"]
        attempt_record["final_plan_valid"] = prefix_validation_summary["final_plan_valid"]
        attempt_record["extra_actions_after_first_valid"] = prefix_validation_summary["extra_actions_after_first_valid"]
        print(
            f"[VALIDATE DONE] model={model_label} protocol={protocol_spec.protocol_id} "
            f"task={task_label} iteration={iteration} "
            f"valid={bool(validation_result.get('valid', False))} "
            f"first_valid_prefix={prefix_validation_summary['first_valid_prefix_length']} "
            f"final_plan_valid={prefix_validation_summary['final_plan_valid']} "
            f"error={validation_result.get('error_type')}",
            flush=True,
        )

        if bool(validation_result.get("valid", False)):
            attempts.append(attempt_record)
            return {
                "solved": True,
                "iterations_used": iteration,
                "max_iterations": protocol_spec.max_iterations,
                "stopped_by_iteration_limit": False,
                "generation_time_seconds": total_generation_time_seconds,
                "raw_output": raw_text,
                "parsed_plan": parsed_plan,
                "validation_result": validation_result,
                "metrics": {},
                "raw_generations": raw_generations,
                "attempts": attempts,
            }

        if protocol_spec.include_external_feedback:
            feedback = build_repair_feedback(validation_result, feedback_prompt=feedback_prompt)
            feedback_messages.append(feedback)
            attempt_record["feedback_to_next_iteration"] = feedback
        attempts.append(attempt_record)

    final_raw_output = None
    if raw_generations:
        final_raw_output = raw_generations[-1].get("raw_text")

    return {
        "solved": False,
        "iterations_used": len(raw_generations),
        "max_iterations": protocol_spec.max_iterations,
        "stopped_by_iteration_limit": len(raw_generations) >= protocol_spec.max_iterations,
        "generation_time_seconds": total_generation_time_seconds,
        "raw_output": final_raw_output,
        "parsed_plan": last_parsed_plan,
        "validation_result": last_validation_result,
        "metrics": {},
        "raw_generations": raw_generations,
        "attempts": attempts,
    }


def build_result_record(
    model_id: str,
    protocol_spec: ProtocolSpec,
    task_spec: TaskSpec,
    solved: bool,
    iterations_used: int,
    stopped_by_iteration_limit: bool,
    generation_time_seconds: float = 0.0,
    raw_output: str | None = None,
    parsed_plan: dict[str, Any] | None = None,
    validation_result: dict[str, Any] | None = None,
    attempts: list[dict[str, Any]] | None = None,
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
        generation_time_seconds=generation_time_seconds,
        raw_output=raw_output,
        parsed_plan=parsed_plan,
        validation_result=validation_result,
        attempts=attempts or [],
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
    run_id: str = "",
) -> ResultRecord:
    """Run one benchmark case and optionally persist its artifacts.

    Current implemented flow:
    1. load task files
    2. run the generation/repair loop
    3. parse and validate each attempt
    4. compute core metrics from the normalized run payload
    5. return a normalized record

    """
    task_inputs = load_task_inputs(task_spec)

    case_payload = run_generation_loop(
        adapter=adapter,
        validator=validator,
        task_spec=task_spec,
        task_inputs=task_inputs,
        protocol_spec=protocol_spec,
        model_id=model_id,
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
            run_id=run_id,
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
            "generation_time_seconds": case_payload["generation_time_seconds"],
            "attempts": _build_raw_attempts(case_payload["attempts"]),
        }
        parsed_payload = {
            "model_id": model_id,
            "task_id": task_spec.task_id,
            "protocol_id": protocol_spec.protocol_id,
            "task_family": task_spec.task_family,
            "tier": task_spec.tier,
            "instance_id": task_spec.instance_id,
            "generation_time_seconds": case_payload["generation_time_seconds"],
            "raw_output_path": str(raw_path),
            "attempts": _build_parsed_attempts(case_payload["attempts"]),
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
        generation_time_seconds=case_payload["generation_time_seconds"],
        raw_output=case_payload["raw_output"],
        parsed_plan=case_payload["parsed_plan"],
        validation_result=case_payload["validation_result"],
        attempts=case_payload["attempts"],
        metrics=metrics_dict,
        raw_output_path=raw_output_path,
        parsed_output_path=parsed_output_path,
        scored_output_path=scored_output_path,
    )

    if output_root is not None and scored_output_path is not None:
        scored_payload = {
            "model_id": result_record.model_id,
            "task_id": result_record.task_id,
            "protocol_id": result_record.protocol_id,
            "task_family": result_record.task_family,
            "tier": result_record.tier,
            "instance_id": result_record.instance_id,
            "solved": result_record.solved,
            "iterations_used": result_record.iterations_used,
            "max_iterations": result_record.max_iterations,
            "stopped_by_iteration_limit": result_record.stopped_by_iteration_limit,
            "generation_time_seconds": result_record.generation_time_seconds,
            "metrics": result_record.metrics,
            "raw_output_path": raw_output_path,
            "parsed_output_path": parsed_output_path,
            "attempts": _build_scored_attempts(case_payload["attempts"]),
        }
        _write_json_artifact(Path(scored_output_path), scored_payload)

    return result_record
