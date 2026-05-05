"""Validator interfaces and VAL integration."""

import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal


ValidationStatus = Literal["valid", "invalid", "parse_error", "timeout", "validator_error"]


@dataclass(slots=True)
class ValidationResult:
    """Normalized validation result for one plan attempt.

    This object should describe only the outcome of validating a single plan,
    not the whole repair loop. Iteration counters belong in the run-level
    result object.
    """

    valid: bool
    status: ValidationStatus
    error_type: str | None = None
    feedback_text: str = ""
    failed_step: int | None = None
    failed_action: str | None = None
    goal_satisfied: bool | None = None
    plan_length: int | None = None
    validation_time_ms: int | None = None
    raw_validator_output: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


class ValidatorAdapter:
    """Common interface for a symbolic validator."""

    def validate(self, domain_file: str, problem_file: str, plan_text: str) -> ValidationResult:
        """Validate a plan and normalize the outcome.

        Semantic contract:
        - `valid=True` implies `status="valid"` and `error_type=None`
        - `status="invalid"` describes a logical plan failure
        - `status="parse_error"`, `status="timeout"` and
          `status="validator_error"` describe technical failures
        """
        raise NotImplementedError


@dataclass(slots=True)
class VALValidatorConfig:
    """Configuration for a real external validator process.

    Notes:
    - `validator_command` should normally be just the executable name or path
    - additional CLI arguments belong in `extra_args`
    - the runner stays agnostic to the concrete validator as long as the final
      output is normalized into `ValidationResult`
    """

    validator_command: str | list[str] | tuple[str, ...]
    timeout_seconds: int = 30
    keep_temp_files: bool = False
    working_directory: str | None = None
    extra_args: list[str] = field(default_factory=list)
    plan_file_suffix: str = ".plan"


class VALValidatorAdapter(ValidatorAdapter):
    """Adapter for running a real external PDDL validator such as VAL.

    The current implementation already handles:
    - temporary plan materialization
    - subprocess invocation with timeout
    - normalization of technical failures
    - heuristic parsing of validator output

    The output parsing is intentionally conservative and should be refined
    against the concrete validator output you decide to adopt.
    """

    def __init__(self, config: VALValidatorConfig):
        self.config = config

    def validate(self, domain_file: str, problem_file: str, plan_text: str) -> ValidationResult:
        plan_file = self._write_temp_plan(plan_text)
        started_at = time.perf_counter()

        try:
            process = self._run_validator_command(domain_file, problem_file, plan_file)
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            return self._parse_validator_process_result(
                process=process,
                plan_text=plan_text,
                validation_time_ms=elapsed_ms,
            )
        except subprocess.TimeoutExpired as exc:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            return ValidationResult(
                valid=False,
                status="timeout",
                error_type="timeout",
                feedback_text="The validator timed out while checking the plan.",
                plan_length=self._count_plan_actions(plan_text),
                validation_time_ms=elapsed_ms,
                raw_validator_output=self._combine_process_output(exc.stdout, exc.stderr),
                details={
                    "validator_kind": "external",
                    "timeout_seconds": self.config.timeout_seconds,
                },
            )
        except FileNotFoundError as exc:
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            return ValidationResult(
                valid=False,
                status="validator_error",
                error_type="validator_unavailable",
                feedback_text="The configured validator executable was not found.",
                plan_length=self._count_plan_actions(plan_text),
                validation_time_ms=elapsed_ms,
                raw_validator_output=str(exc),
                details={
                    "validator_kind": "external",
                    "validator_command": self._command_preview(),
                },
            )
        except Exception as exc:  # pragma: no cover - defensive wrapper behavior
            elapsed_ms = int((time.perf_counter() - started_at) * 1000)
            return ValidationResult(
                valid=False,
                status="validator_error",
                error_type="validator_crash",
                feedback_text="The validator crashed while checking the plan.",
                plan_length=self._count_plan_actions(plan_text),
                validation_time_ms=elapsed_ms,
                raw_validator_output=str(exc),
                details={
                    "validator_kind": "external",
                    "exception_type": type(exc).__name__,
                },
            )
        finally:
            self._cleanup_temp_plan(plan_file)

    def _write_temp_plan(self, plan_text: str) -> Path:
        """Materialize the current plan into a temporary file for validation."""
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            delete=False,
            suffix=self.config.plan_file_suffix,
        ) as handle:
            handle.write(plan_text)
            handle.flush()
            return Path(handle.name)

    def _cleanup_temp_plan(self, plan_file: Path) -> None:
        """Delete the temporary plan file unless debugging requires keeping it."""
        if self.config.keep_temp_files:
            return

        try:
            plan_file.unlink(missing_ok=True)
        except OSError:
            return

    def _run_validator_command(
        self,
        domain_file: str,
        problem_file: str,
        plan_file: Path,
    ) -> subprocess.CompletedProcess[str]:
        """Invoke the external validator command."""
        command = self._build_command(domain_file, problem_file, plan_file)
        return subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            cwd=self.config.working_directory,
            check=False,
        )

    def _build_command(self, domain_file: str, problem_file: str, plan_file: Path) -> list[str]:
        """Build the concrete subprocess command line."""
        if isinstance(self.config.validator_command, str):
            command = [self.config.validator_command]
        else:
            command = [str(part) for part in self.config.validator_command]

        command.extend(str(arg) for arg in self.config.extra_args)
        command.extend([str(domain_file), str(problem_file), str(plan_file)])
        return command

    def _command_preview(self) -> list[str]:
        """Return the configured command without task-specific paths."""
        if isinstance(self.config.validator_command, str):
            command = [self.config.validator_command]
        else:
            command = [str(part) for part in self.config.validator_command]
        command.extend(str(arg) for arg in self.config.extra_args)
        return command

    def _parse_validator_process_result(
        self,
        process: subprocess.CompletedProcess[str],
        plan_text: str,
        validation_time_ms: int,
    ) -> ValidationResult:
        """Translate subprocess output into the benchmark validation schema.

        This is a heuristic parser: once you settle on a concrete validator,
        you should refine these markers against real stdout/stderr examples.
        """
        raw_output = self._combine_process_output(process.stdout, process.stderr)
        normalized_output = raw_output.lower() if raw_output else None
        plan_length = self._count_plan_actions(plan_text)
        failed_step = self._extract_failed_step(raw_output)
        failed_action = self._extract_failed_action(raw_output)

        if self._looks_like_valid_plan(normalized_output, process.returncode):
            return ValidationResult(
                valid=True,
                status="valid",
                error_type=None,
                feedback_text="Plan validated successfully.",
                failed_step=None,
                failed_action=None,
                goal_satisfied=True,
                plan_length=plan_length,
                validation_time_ms=validation_time_ms,
                raw_validator_output=raw_output,
                details={
                    "validator_kind": "external",
                    "return_code": process.returncode,
                },
            )

        error_type = self._infer_error_type(normalized_output)
        status: ValidationStatus = "invalid"
        if error_type in {"validator_crash", "validator_unavailable"}:
            status = "validator_error"

        feedback_text = self._build_feedback_text(
            error_type=error_type,
            failed_step=failed_step,
            failed_action=failed_action,
            raw_output=raw_output,
        )

        return ValidationResult(
            valid=False,
            status=status,
            error_type=error_type,
            feedback_text=feedback_text,
            failed_step=failed_step,
            failed_action=failed_action,
            goal_satisfied=False if error_type == "unsatisfied_goal" else None,
            plan_length=plan_length,
            validation_time_ms=validation_time_ms,
            raw_validator_output=raw_output,
            details={
                "validator_kind": "external",
                "return_code": process.returncode,
            },
        )

    @staticmethod
    def _combine_process_output(
        stdout: str | bytes | None,
        stderr: str | bytes | None,
    ) -> str | None:
        """Merge stdout and stderr into a single debuggable string."""
        normalized_parts: list[str] = []

        for part in (stdout, stderr):
            normalized_part = VALValidatorAdapter._normalize_output_part(part)
            if normalized_part is not None:
                normalized_parts.append(normalized_part)

        parts = normalized_parts
        if not parts:
            return None
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_output_part(part: str | bytes | None) -> str | None:
        """Normalize one stdout/stderr chunk into a stripped string."""
        if part is None:
            return None
        if isinstance(part, bytes):
            text_part = part.decode("utf-8", errors="replace")
        elif isinstance(part, str):
            text_part = part
        else:
            text_part = str(part)

        stripped = text_part.strip()
        return stripped or None

    @staticmethod
    def _count_plan_actions(plan_text: str) -> int:
        """Count non-empty action lines in the submitted plan."""
        return sum(1 for line in plan_text.splitlines() if line.strip())

    @staticmethod
    def _looks_like_valid_plan(normalized_output: str | None, return_code: int) -> bool:
        """Heuristically detect a successful validation."""
        if not normalized_output:
            return return_code == 0

        success_markers = (
            "plan valid",
            "plan has been successfully executed",
            "successful plans",
            "goal achieved",
            "validation successful",
        )
        return any(marker in normalized_output for marker in success_markers)

    @staticmethod
    def _infer_error_type(normalized_output: str | None) -> str:
        """Map validator output to the shared benchmark error taxonomy."""
        if not normalized_output:
            return "unknown"

        if "precondition" in normalized_output or "not applicable" in normalized_output:
            return "invalid_precondition"
        if "goal" in normalized_output and any(
            marker in normalized_output
            for marker in ("not satisfied", "not achieved", "unsatisfied")
        ):
            return "unsatisfied_goal"
        if "unknown operator" in normalized_output or "unknown action" in normalized_output:
            return "unknown_action"
        if "syntax" in normalized_output or "parse error" in normalized_output:
            return "syntax_error"
        if "segmentation fault" in normalized_output or "traceback" in normalized_output:
            return "validator_crash"
        return "unknown"

    @staticmethod
    def _extract_failed_step(raw_output: str | None) -> int | None:
        """Extract the first failing step index when the validator exposes it."""
        if not raw_output:
            return None

        match = re.search(r"step\s+(\d+)", raw_output, flags=re.IGNORECASE)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _extract_failed_action(raw_output: str | None) -> str | None:
        """Extract a failing parenthesized action if present in validator output."""
        if not raw_output:
            return None

        match = re.search(r"(\([^()\n]+\))", raw_output)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _build_feedback_text(
        *,
        error_type: str,
        failed_step: int | None,
        failed_action: str | None,
        raw_output: str | None,
    ) -> str:
        """Build a compact repair-oriented summary from validator output."""
        fragments: list[str] = []

        if error_type == "invalid_precondition":
            fragments.append("The plan contains an action whose preconditions are not satisfied.")
        elif error_type == "unsatisfied_goal":
            fragments.append("The plan finishes without achieving the goal.")
        elif error_type == "unknown_action":
            fragments.append("The plan references an action that is not recognized in the domain.")
        elif error_type == "syntax_error":
            fragments.append("The plan format is not accepted by the validator.")
        elif error_type == "validator_crash":
            fragments.append("The validator crashed while analyzing the plan.")
        else:
            fragments.append("The validator rejected the current plan.")

        if failed_step is not None:
            fragments.append(f"Failure reported at step {failed_step}.")
        if failed_action:
            fragments.append(f"Problematic action: {failed_action}.")
        if raw_output:
            fragments.append(f"Validator output: {raw_output[:240]}")

        return " ".join(fragments)


def build_feedback_from_validation(result: ValidationResult) -> str:
    """Turn a validation failure into a repair-oriented feedback string."""
    if result.valid:
        return result.feedback_text or "The previous plan validated successfully."

    lines = [
        "The previous plan is invalid.",
        f"Validation status: {result.status}",
        f"Error type: {result.error_type or 'unknown'}",
    ]

    if result.failed_step is not None:
        lines.append(f"Failed step: {result.failed_step}")
    if result.failed_action:
        lines.append(f"Failed action: {result.failed_action}")

    lines.append(
        f"Feedback: {result.feedback_text or 'No extra details provided.'}"
    )
    lines.append("Please return a corrected action sequence.")

    return "\n".join(lines)
