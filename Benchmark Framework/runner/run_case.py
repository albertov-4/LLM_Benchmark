"""Single benchmark case scaffold.

This file contains guided pseudocode for the smallest benchmark unit:
one model, one protocol, one task instance.
"""

from dataclasses import dataclass
from pathlib import Path


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

    user_content = (
        f"TASK FAMILY: {task_spec.task_family}\n"
        f"DIFFICULTY: {task_spec.tier}\n"
        f"INSTANCE: {task_spec.instance_id}\n\n"
        f"{domain_prompt}\n\n"
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


def build_repair_feedback(validation_result: dict[str, object]) -> str:
    """Convert validator output into a user-facing repair message.

    In a future implementation this could delegate to a domain-specific
    feedback module.
    """
    error_type = validation_result.get("error_type", "unknown")
    error_message = validation_result.get("error_message", "Unknown validation error.")
    return (
        "The previous plan did not validate.\n"
        f"Error type: {error_type}\n"
        f"Validator message: {error_message}\n"
        "Please provide a corrected action sequence."
    )


def run_generation_loop(
    adapter,
    validator,
    task_spec: TaskSpec,
    task_inputs: dict[str, str],
    protocol_spec: ProtocolSpec,
    system_prompt: str = "",
    domain_prompt: str = "",
) -> dict[str, object]:
    """Core pseudocode loop for one benchmark case.

    Expected adapter contract:
    - adapter.generate(messages) -> {"raw_text": "...", ...}

    Expected validator contract:
    - validator.validate(domain_file, problem_file, plan_text) -> {
          "valid": bool,
          "error_type": str | None,
          "error_message": str | None,
      }
    """
    feedback_messages: list[str] = []
    raw_generations: list[dict[str, object]] = []

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
        raw_generations.append(generation)

        # PSEUDOCODE:
        # parsed_plan = parse_plan_text(generation["raw_text"])
        # if parsed_plan.actions is empty:
        #     feedback_messages.append("No valid plan format detected. Return only actions.")
        #     continue
        #
        # plan_text = "\n".join(parsed_plan.actions)
        # validation = validator.validate(task_spec.domain_file, task_spec.problem_file, plan_text)
        # if validation["valid"]:
        #     return success payload
        # feedback_messages.append(build_repair_feedback(validation))

    return {
        "status": "pseudocode_only",
        "iterations_attempted": protocol_spec.max_iterations,
        "raw_generations": raw_generations,
    }


def build_result_record(
    model_id: str,
    protocol_spec: ProtocolSpec,
    task_spec: TaskSpec,
    valid: bool,
    iterations: int,
    error_type: str | None,
    raw_output_path: str,
    parsed_output_path: str,
    scored_output_path: str,
) -> ResultRecord:
    """Normalize the final case result into a serializable record."""
    return ResultRecord(
        model_id=model_id,
        protocol_id=protocol_spec.protocol_id,
        task_family=task_spec.task_family,
        tier=task_spec.tier,
        instance_id=task_spec.instance_id,
        valid=valid,
        iterations=iterations,
        error_type=error_type,
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
) -> dict[str, object]:
    """Pseudocode entry point for one benchmark case.

    Suggested future implementation:
    1. load task files
    2. run the generation/repair loop
    3. parse and validate the final answer
    4. compute metrics
    5. save raw / parsed / scored outputs
    6. return a normalized record
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
    )

    # PSEUDOCODE:
    # parsed_plan = parse_plan_text(...)
    # validation = validator.validate(...)
    # metrics = compute_core_metrics(...)
    # save_json(raw_output_path, raw_payload)
    # save_json(parsed_output_path, parsed_payload)
    # save_json(scored_output_path, metrics_payload)
    # return build_result_record(...)

    return {
        "model_id": model_id,
        "protocol_id": protocol_spec.protocol_id,
        "task_family": task_spec.task_family,
        "tier": task_spec.tier,
        "instance_id": task_spec.instance_id,
        "case_payload": case_payload,
    }
