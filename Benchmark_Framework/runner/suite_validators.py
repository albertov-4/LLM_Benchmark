"""Validator construction and task preflight helpers."""

from __future__ import annotations

import shutil
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable


class _UnavailableValidator:
    """Fallback validator used when no real validator is connected yet."""

    def validate(self, domain_file: str, problem_file: str, plan_text: str) -> dict[str, Any]:
        return {
            "valid": False,
            "status": "validator_error",
            "error_type": "validator_unavailable",
            "feedback_text": "No validator is configured for this benchmark run.",
            "failed_step": None,
            "failed_action": None,
            "goal_satisfied": None,
            "plan_length": None,
            "validation_time_ms": None,
            "raw_validator_output": None,
            "details": {
                "domain_file": domain_file,
                "problem_file": problem_file,
                "plan_preview": plan_text[:200],
            },
        }


@lru_cache(maxsize=None)
def _load_framework_module(module_key: str, relative_path: str):
    """Load a framework module without requiring package installation."""
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_validate_command(
    framework_root: Path,
    validator_command: str | Path | None = None,
) -> str | None:
    """Resolve the VAL executable from an explicit command or common local paths."""
    if validator_command is not None:
        explicit_command = str(validator_command)
        if any(sep in explicit_command for sep in ("\\", "/")) or explicit_command.endswith(".exe"):
            explicit_path = Path(explicit_command)
            if explicit_path.exists():
                return str(explicit_path)
            return None

        path_command = shutil.which(explicit_command)
        if path_command:
            return path_command
        return None

    for candidate_name in ("Validate", "validate"):
        path_command = shutil.which(candidate_name)
        if path_command:
            return path_command

    workspace_root = framework_root.parent
    candidate_paths = [
        framework_root / "utils" / "win64" / "bin" / "Validate.exe",
        framework_root / "utils" / "linux64" / "bin" / "Validate",
        workspace_root / "VAL" / "build" / "win64" / "mingw" / "Debug" / "bin" / "Validate.exe",
        workspace_root / "VAL" / "build" / "win64" / "mingw" / "Debug" / "install" / "bin" / "Validate.exe",
        workspace_root / "VAL" / "build" / "win64" / "mingw" / "Release" / "bin" / "Validate.exe",
        workspace_root.parent / "VAL" / "build" / "win64" / "mingw" / "Debug" / "bin" / "Validate.exe",
        workspace_root.parent / "VAL" / "build" / "win64" / "mingw" / "Debug" / "install" / "bin" / "Validate.exe",
        workspace_root.parent / "VAL" / "build" / "win64" / "mingw" / "Release" / "bin" / "Validate.exe",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    return None


def build_real_val_validator(
    framework_root: Path,
    validator_command: str | Path | None = None,
    timeout_seconds: int = 30,
    keep_temp_files: bool = False,
    working_directory: str | Path | None = None,
    extra_args: list[str] | None = None,
):
    """Build a real VAL-backed validator adapter."""
    validator_module = _load_framework_module(
        "benchmark_framework_validator_module",
        "evaluators/validator.py",
    )
    resolved_command = _resolve_validate_command(
        framework_root=framework_root,
        validator_command=validator_command,
    )
    if resolved_command is None:
        raise FileNotFoundError(
            "Unable to resolve the VAL validator executable. "
            "Pass `validator_command=...` or add `Validate` to PATH."
        )

    config = validator_module.VALValidatorConfig(
        validator_command=resolved_command,
        timeout_seconds=timeout_seconds,
        keep_temp_files=keep_temp_files,
        working_directory=str(working_directory) if working_directory is not None else None,
        extra_args=list(extra_args or []),
    )
    return validator_module.VALValidatorAdapter(config)


def build_validator(
    framework_root: Path,
    validator: Any | None = None,
    validator_factory: Callable[[], Any] | None = None,
    use_real_validator: bool = False,
    validator_command: str | Path | None = None,
    validator_timeout_seconds: int = 30,
    validator_keep_temp_files: bool = False,
    validator_working_directory: str | Path | None = None,
    validator_extra_args: list[str] | None = None,
):
    """Return a validator instance for one suite job."""
    if validator_factory is not None:
        return validator_factory()
    if validator is not None:
        return validator
    if use_real_validator:
        return build_real_val_validator(
            framework_root=framework_root,
            validator_command=validator_command,
            timeout_seconds=validator_timeout_seconds,
            keep_temp_files=validator_keep_temp_files,
            working_directory=validator_working_directory,
            extra_args=validator_extra_args,
        )
    return _UnavailableValidator()


def run_task_preflights(
    *,
    framework_root: Path,
    task_cases: list[Any],
    validator_command: str | Path | None = None,
    validator_timeout_seconds: int = 30,
    validator_working_directory: str | Path | None = None,
    validator_extra_args: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run VAL domain/problem checks before model execution."""
    preflight_module = _load_framework_module(
        "benchmark_framework_preflight_module",
        "evaluators/preflight.py",
    )
    resolved_command = _resolve_validate_command(
        framework_root=framework_root,
        validator_command=validator_command,
    )

    if resolved_command is None:
        return [
            {
                "task_family": task_case.task_family,
                "tier": task_case.tier,
                "instance_id": task_case.instance_id,
                "domain_file": str(task_case.domain_file),
                "problem_file": str(task_case.problem_file),
                "ok": False,
                "status": "validator_unavailable",
                "return_code": None,
                "validation_time_ms": None,
                "raw_validator_output": None,
                "error_message": (
                    "Unable to resolve the VAL validator executable. "
                    "Pass `validator_command=...` or add `Validate` to PATH."
                ),
                "details": {
                    "validator_command": str(validator_command) if validator_command is not None else None,
                },
            }
            for task_case in task_cases
        ]

    results: list[dict[str, Any]] = []
    for task_case in task_cases:
        result = preflight_module.run_val_domain_problem_preflight(
            validator_command=resolved_command,
            domain_file=task_case.domain_file,
            problem_file=task_case.problem_file,
            task_family=task_case.task_family,
            tier=task_case.tier,
            instance_id=task_case.instance_id,
            timeout_seconds=validator_timeout_seconds,
            working_directory=validator_working_directory,
            extra_args=validator_extra_args,
        )
        results.append(result.to_dict())
    return results


def _build_preflight_error_payloads(preflight_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Translate failed preflight checks into suite orchestration errors."""
    error_payloads: list[dict[str, Any]] = []
    for result in preflight_results:
        if bool(result.get("ok", False)):
            continue
        error_payloads.append(
            {
                "model_id": None,
                "protocol_id": None,
                "task_family": result.get("task_family"),
                "tier": result.get("tier"),
                "instance_id": result.get("instance_id"),
                "error_type": result.get("status", "preflight_failed"),
                "error_message": result.get("error_message") or "Task preflight failed.",
            }
        )
    return error_payloads
