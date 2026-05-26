"""Task preflight checks for PDDL domain/problem inputs."""

import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TaskPreflightResult:
    """Result of checking one domain/problem pair before model execution."""

    task_family: str
    tier: str
    instance_id: str
    domain_file: str
    problem_file: str
    ok: bool
    status: str
    return_code: int | None
    validation_time_ms: int | None
    raw_validator_output: str | None
    error_message: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize the preflight result as a plain dictionary."""
        return asdict(self)


def run_val_domain_problem_preflight(
    *,
    validator_command: str | list[str] | tuple[str, ...],
    domain_file: str | Path,
    problem_file: str | Path,
    task_family: str,
    tier: str,
    instance_id: str,
    timeout_seconds: int = 30,
    working_directory: str | Path | None = None,
    extra_args: list[str] | None = None,
) -> TaskPreflightResult:
    """Check that VAL can parse one domain/problem pair without a plan file."""
    command = _build_command(
        validator_command=validator_command,
        domain_file=domain_file,
        problem_file=problem_file,
        extra_args=extra_args,
    )
    started_at = time.perf_counter()

    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(working_directory) if working_directory is not None else None,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return TaskPreflightResult(
            task_family=task_family,
            tier=tier,
            instance_id=instance_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            ok=False,
            status="timeout",
            return_code=None,
            validation_time_ms=elapsed_ms,
            raw_validator_output=_combine_process_output(exc.stdout, exc.stderr),
            error_message="VAL timed out while checking the domain/problem pair.",
            details={"timeout_seconds": timeout_seconds, "command": _command_preview(command)},
        )
    except FileNotFoundError as exc:
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return TaskPreflightResult(
            task_family=task_family,
            tier=tier,
            instance_id=instance_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            ok=False,
            status="validator_unavailable",
            return_code=None,
            validation_time_ms=elapsed_ms,
            raw_validator_output=str(exc),
            error_message="The configured VAL executable was not found.",
            details={"command": _command_preview(command)},
        )
    except Exception as exc:  # pragma: no cover - defensive subprocess wrapper
        elapsed_ms = int((time.perf_counter() - started_at) * 1000)
        return TaskPreflightResult(
            task_family=task_family,
            tier=tier,
            instance_id=instance_id,
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            ok=False,
            status="validator_error",
            return_code=None,
            validation_time_ms=elapsed_ms,
            raw_validator_output=str(exc),
            error_message="VAL crashed while checking the domain/problem pair.",
            details={
                "command": _command_preview(command),
                "exception_type": type(exc).__name__,
            },
        )

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)
    raw_output = _combine_process_output(process.stdout, process.stderr)
    ok = process.returncode == 0
    return TaskPreflightResult(
        task_family=task_family,
        tier=tier,
        instance_id=instance_id,
        domain_file=str(domain_file),
        problem_file=str(problem_file),
        ok=ok,
        status="ok" if ok else "invalid_pddl",
        return_code=process.returncode,
        validation_time_ms=elapsed_ms,
        raw_validator_output=raw_output,
        error_message=None if ok else "VAL rejected the domain/problem pair.",
        details={"command": _command_preview(command)},
    )


def _build_command(
    *,
    validator_command: str | list[str] | tuple[str, ...],
    domain_file: str | Path,
    problem_file: str | Path,
    extra_args: list[str] | None,
) -> list[str]:
    if isinstance(validator_command, str):
        command = [validator_command]
    else:
        command = [str(part) for part in validator_command]

    command.extend(str(arg) for arg in extra_args or [])
    command.extend([str(domain_file), str(problem_file)])
    return command


def _command_preview(command: list[str]) -> list[str]:
    return [str(part) for part in command]


def _combine_process_output(
    stdout: str | bytes | None,
    stderr: str | bytes | None,
) -> str | None:
    parts: list[str] = []
    for part in (stdout, stderr):
        normalized = _normalize_output_part(part)
        if normalized is not None:
            parts.append(normalized)
    if not parts:
        return None
    return "\n\n".join(parts)


def _normalize_output_part(part: str | bytes | None) -> str | None:
    if part is None:
        return None
    if isinstance(part, bytes):
        text_part = part.decode("utf-8", errors="replace")
    else:
        text_part = str(part)

    stripped = text_part.strip()
    return stripped or None
