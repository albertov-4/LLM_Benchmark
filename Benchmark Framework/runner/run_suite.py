"""Suite-level benchmark orchestration based on directory discovery.

This module orchestrates a full benchmark campaign by:
- discovering tasks from the folder hierarchy
- loading protocol and model metadata
- delegating single-run execution to `run_case.py`
- aggregating normalized results
"""

import shutil
from dataclasses import asdict, dataclass, is_dataclass
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable


TIERS = ("easy", "medium", "hard")


@dataclass(slots=True)
class DiscoveredTaskCase:
    task_family: str
    tier: str
    instance_id: str
    domain_file: Path
    problem_file: Path


@dataclass(slots=True)
class SuiteJob:
    model_id: str
    protocol_id: str
    task_case: DiscoveredTaskCase


@dataclass(slots=True)
class LoadedProtocolConfig:
    protocol_id: str
    max_iterations: int
    require_final_plan_only: bool
    raw_config: dict[str, Any]


@dataclass(slots=True)
class LoadedPromptBundle:
    system_prompt: str
    domain_prompt: str
    feedback_prompt: str


class _UnavailableAdapter:
    """Minimal adapter used when a configured adapter cannot be created."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def generate(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "raw_text": "",
            "usage": {},
            "latency_s": None,
            "notes": [
                "No concrete adapter factory was provided.",
                "The configured adapter could not be initialized for this run.",
            ],
            "message_count": len(messages),
        }


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
    """Load a sibling framework module without requiring package installation."""
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_framework_path(path_value: str | Path | None, default_relative: str) -> Path:
    """Resolve benchmark paths relative to the framework root by default."""
    framework_root = Path(__file__).resolve().parents[1]
    if path_value is None:
        return framework_root / default_relative

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return framework_root / candidate


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


def _parse_scalar(raw_value: str) -> Any:
    """Parse a simple scalar from the lightweight YAML-like config files."""
    config_loader = _load_framework_module(
        "benchmark_framework_config_loader",
        "utils/config_loader.py",
    )
    return config_loader.parse_scalar(raw_value)


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
        "metrics": result.get("metrics", {}),
        "raw_output_path": result.get("raw_output_path"),
        "parsed_output_path": result.get("parsed_output_path"),
        "scored_output_path": result.get("scored_output_path"),
    }


def _read_text_if_exists(file_path: Path) -> str:
    """Read a UTF-8 text file when present, otherwise return an empty string."""
    if not file_path.exists():
        return ""
    return file_path.read_text(encoding="utf-8").strip()


def _parse_optional_int(value: Any, field_name: str) -> int | None:
    """Parse optional integer config fields from registry entries."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid integer value for {field_name}: {value!r}") from exc


def _read_required_text(file_path: Path, description: str) -> str:
    """Read a required UTF-8 text file and fail clearly when it is missing or empty."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing {description}: {file_path}. "
            "Create this prompt file or disable the related protocol option."
        )

    content = file_path.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(
            f"Empty {description}: {file_path}. "
            "Fill this prompt file or disable the related protocol option."
        )
    return content


def load_prompt_bundle(
    task_family: str,
    protocol_config: LoadedProtocolConfig,
    prompts_root: str | Path,
) -> LoadedPromptBundle:
    """Load the prompt texts used by one suite job."""
    prompts_root_path = Path(prompts_root)
    prompting_config = protocol_config.raw_config.get("prompting", {})
    if not isinstance(prompting_config, dict):
        prompting_config = {}

    use_system_prompt = bool(prompting_config.get("use_system_prompt", True))
    include_domain_prompt = bool(prompting_config.get("include_domain_prompt", True))
    include_examples = bool(prompting_config.get("include_examples", False))
    include_external_feedback = bool(prompting_config.get("include_external_feedback", False))

    system_prompt = ""
    if use_system_prompt:
        system_prompt = _read_text_if_exists(prompts_root_path / "system.txt")

    domain_prompt = ""
    if include_domain_prompt:
        domain_prompt = _read_required_text(
            prompts_root_path / f"{task_family}.txt",
            f"domain prompt for task family '{task_family}'",
        )

    if include_examples:
        example_prompt = _read_text_if_exists(prompts_root_path / "examples" / f"{task_family}.txt")
        if example_prompt:
            if domain_prompt:
                domain_prompt = f"{domain_prompt}\n\n=== EXAMPLES ===\n{example_prompt}"
            else:
                domain_prompt = f"=== EXAMPLES ===\n{example_prompt}"

    feedback_prompt = ""
    if include_external_feedback:
        feedback_prompt = _read_text_if_exists(prompts_root_path / "feedback.txt")

    return LoadedPromptBundle(
        system_prompt=system_prompt,
        domain_prompt=domain_prompt,
        feedback_prompt=feedback_prompt,
    )


def discover_task_cases(tasks_root: str | Path) -> list[DiscoveredTaskCase]:
    """Discover all benchmark cases from the task folder hierarchy.

    Expected layout:
        tasks/<task_family>/domain/domain.pddl
        tasks/<task_family>/easy/*.pddl
        tasks/<task_family>/medium/*.pddl
        tasks/<task_family>/hard/*.pddl
    """
    root = Path(tasks_root)
    cases: list[DiscoveredTaskCase] = []

    if not root.exists():
        return cases

    for family_dir in sorted(root.iterdir()):
        if not family_dir.is_dir() or family_dir.name.startswith("_") or family_dir.name == "metadata":
            continue

        domain_file = family_dir / "domain" / "domain.pddl"
        for tier in TIERS:
            tier_dir = family_dir / tier
            if not tier_dir.exists():
                continue
            for problem_file in sorted(tier_dir.glob("*.pddl")):
                cases.append(
                    DiscoveredTaskCase(
                        task_family=family_dir.name,
                        tier=tier,
                        instance_id=problem_file.stem,
                        domain_file=domain_file,
                        problem_file=problem_file,
                    )
                )

    return cases


def filter_task_cases(
    task_cases: list[DiscoveredTaskCase],
    task_family: str | None = None,
    tier: str | None = None,
    instance_id: str | None = None,
) -> list[DiscoveredTaskCase]:
    """Filter discovered task cases by optional task selectors."""
    filtered_cases = task_cases
    if task_family is not None:
        filtered_cases = [
            task_case
            for task_case in filtered_cases
            if task_case.task_family == task_family
        ]
    if tier is not None:
        filtered_cases = [
            task_case
            for task_case in filtered_cases
            if task_case.tier == tier
        ]
    if instance_id is not None:
        filtered_cases = [
            task_case
            for task_case in filtered_cases
            if task_case.instance_id == instance_id
        ]
    return filtered_cases


def discover_protocol_ids(protocols_root: str | Path) -> list[str]:
    """Return protocol ids based on yaml filenames.

    Example:
        protocols/direct_plan.yaml -> direct_plan
    """
    root = Path(protocols_root)
    if not root.exists():
        return []
    return sorted(file_path.stem for file_path in root.glob("*.yaml"))


def load_model_registry_entries(model_registry_path: str | Path) -> list[dict[str, Any]]:
    """Extract model entries from the registry."""
    config_loader = _load_framework_module(
        "benchmark_framework_config_loader",
        "utils/config_loader.py",
    )
    return config_loader.load_model_registry_entries(model_registry_path)


def load_protocol_config(
    protocol_id: str,
    protocols_root: str | Path,
) -> LoadedProtocolConfig:
    """Load the protocol metadata needed by the runner with a lightweight parser."""
    protocol_path = Path(protocols_root) / f"{protocol_id}.yaml"
    if not protocol_path.exists():
        raise FileNotFoundError(f"Protocol config not found: {protocol_path}")

    config: dict[str, Any] = {}
    current_section: str | None = None

    for line in protocol_path.read_text(encoding="utf-8").splitlines():
        raw_line = line.rstrip()
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("- "):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0:
            key, _, raw_value = stripped.partition(":")
            key = key.strip()
            raw_value = raw_value.strip()

            if raw_value in {">", "|"}:
                config[key] = ""
                current_section = None
                continue

            if not raw_value:
                config[key] = {}
                current_section = key
                continue

            config[key] = _parse_scalar(raw_value)
            current_section = None
            continue

        if indent == 2 and current_section and ":" in stripped:
            key, _, raw_value = stripped.partition(":")
            key = key.strip()
            raw_value = raw_value.strip()
            if raw_value in {">", "|"}:
                continue
            section = config.setdefault(current_section, {})
            if isinstance(section, dict):
                section[key] = _parse_scalar(raw_value)

    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        evaluation = {}

    return LoadedProtocolConfig(
        protocol_id=str(config.get("protocol_id", protocol_id)),
        max_iterations=int(evaluation.get("max_iterations", 1)),
        require_final_plan_only=bool(evaluation.get("require_final_plan_only", True)),
        raw_config=config,
    )


def build_run_matrix(
    task_cases: list[DiscoveredTaskCase],
    model_ids: list[str],
    protocol_ids: list[str],
) -> list[SuiteJob]:
    """Return the full matrix model x protocol x discovered task case."""
    matrix: list[SuiteJob] = []

    for model_id in model_ids:
        for protocol_id in protocol_ids:
            for task_case in task_cases:
                matrix.append(
                    SuiteJob(
                        model_id=model_id,
                        protocol_id=protocol_id,
                        task_case=task_case,
                    )
                )

    return matrix


def summarize_suite_inputs(
    task_cases: list[DiscoveredTaskCase],
    model_ids: list[str],
    protocol_ids: list[str],
) -> dict[str, int]:
    """Return a small summary useful before launching a full run."""
    return {
        "num_task_cases": len(task_cases),
        "num_models": len(model_ids),
        "num_protocols": len(protocol_ids),
        "num_jobs": len(task_cases) * len(model_ids) * len(protocol_ids),
    }


def build_task_spec_from_case(task_case: DiscoveredTaskCase):
    """Convert a discovered filesystem case into the TaskSpec used by run_case."""
    run_case_module = _load_framework_module(
        "benchmark_framework_run_case",
        "runner/run_case.py",
    )
    return run_case_module.build_task_spec(
        task_family=task_case.task_family,
        tier=task_case.tier,
        instance_id=task_case.instance_id,
        domain_file=str(task_case.domain_file),
        problem_file=str(task_case.problem_file),
    )


def build_protocol_spec(protocol_config: LoadedProtocolConfig):
    """Convert a loaded protocol config into the ProtocolSpec used by run_case."""
    run_case_module = _load_framework_module(
        "benchmark_framework_run_case",
        "runner/run_case.py",
    )
    prompting_config = protocol_config.raw_config.get("prompting", {})
    if not isinstance(prompting_config, dict):
        prompting_config = {}

    return run_case_module.ProtocolSpec(
        protocol_id=protocol_config.protocol_id,
        max_iterations=protocol_config.max_iterations,
        require_final_plan_only=protocol_config.require_final_plan_only,
        include_external_feedback=bool(prompting_config.get("include_external_feedback", False)),
        include_chain_of_thought=bool(prompting_config.get("include_chain_of_thought", False)),
    )


def build_model_adapter(
    model_entry: dict[str, Any],
    protocol_config: LoadedProtocolConfig,
    adapter_factory: Callable[[dict[str, Any], LoadedProtocolConfig], Any] | None = None,
):
    """Build the adapter used by one suite job.

    If no external factory is provided, this tries to instantiate a concrete
    adapter when possible. Otherwise it falls back to an unavailable adapter
    that returns a normalized error-like generation payload.
    """
    if adapter_factory is not None:
        return adapter_factory(model_entry, protocol_config)

    adapter_name = str(model_entry.get("adapter", "")).strip()
    model_id = str(model_entry.get("model_id", "unknown-model"))

    if adapter_name == "hf_local":
        try:
            hf_module = _load_framework_module(
                "benchmark_framework_hf_local_adapter",
                "models/adapters/hf_local.py",
            )
            generation_config = protocol_config.raw_config.get("generation", {})
            if not isinstance(generation_config, dict):
                generation_config = {}

            top_k = generation_config.get("top_k", 10)
            if top_k is None:
                top_k = 10

            hf_config = hf_module.HFLocalConfig(
                model_id=model_id,
                weights_path=str(model_entry.get("weights_path", "")),
                temperature=float(generation_config.get("temperature", 0.0) or 0.0),
                top_k=int(top_k),
                max_tokens=int(generation_config.get("max_tokens", 4096) or 4096),
                device_map=None if model_entry.get("device_map") in {None, "", "none"} else str(model_entry.get("device_map", "auto")),
                torch_dtype=None if model_entry.get("torch_dtype") in {None, ""} else str(model_entry.get("torch_dtype", "auto")),
                trust_remote_code=bool(model_entry.get("trust_remote_code", False)),
                use_chat_template=bool(model_entry.get("use_chat_template", True)),
                add_generation_prompt=bool(model_entry.get("add_generation_prompt", True)),
            )
            return hf_module.HFLocalAdapter(hf_config)
        except Exception:
            return _UnavailableAdapter(model_id)

    if adapter_name == "ollama":
        try:
            ollama_module = _load_framework_module(
                "benchmark_framework_ollama_adapter",
                "models/adapters/ollama.py",
            )
            generation_config = protocol_config.raw_config.get("generation", {})
            if not isinstance(generation_config, dict):
                generation_config = {}

            top_k = generation_config.get("top_k", 10)
            if top_k is None:
                top_k = 10

            ollama_config = ollama_module.OllamaConfig(
                model_id=model_id,
                ollama_model=str(model_entry.get("ollama_model", model_entry.get("api_model_name", ""))),
                base_url=str(model_entry.get("base_url", "http://localhost:11434")),
                temperature=float(generation_config.get("temperature", 0.0) or 0.0),
                top_k=int(top_k),
                max_tokens=int(generation_config.get("max_tokens", 4096) or 4096),
                timeout_seconds=int(model_entry.get("timeout_seconds", 300) or 300),
            )
            return ollama_module.OllamaAdapter(ollama_config)
        except Exception:
            return _UnavailableAdapter(model_id)

    if adapter_name == "nvidia_api":
        try:
            nvidia_module = _load_framework_module(
                "benchmark_framework_nvidia_api_adapter",
                "models/adapters/nvidia_api.py",
            )
            generation_config = protocol_config.raw_config.get("generation", {})
            if not isinstance(generation_config, dict):
                generation_config = {}

            extra_body: dict[str, Any] = {}
            thinking_key = str(model_entry.get("thinking_key", "") or "").strip()
            if thinking_key:
                extra_body["chat_template_kwargs"] = {
                    thinking_key: bool(model_entry.get("thinking_enabled", False)),
                }
            reasoning_budget = _parse_optional_int(
                model_entry.get("reasoning_budget"),
                "reasoning_budget",
            )
            if reasoning_budget is not None:
                extra_body["reasoning_budget"] = reasoning_budget

            nvidia_config = nvidia_module.NvidiaAPIConfig(
                model_id=model_id,
                api_model_name=str(model_entry.get("api_model_name", model_id)),
                base_url=str(model_entry.get("base_url", "https://integrate.api.nvidia.com/v1")),
                api_key_env=str(model_entry.get("api_key_env", "NVIDIA_API_KEY")),
                api_mode=str(model_entry.get("api_mode", "chat_completions")),
                stream=bool(model_entry.get("stream", False)),
                temperature=float(model_entry.get("temperature", generation_config.get("temperature", 0.0)) or 0.0),
                top_p=float(model_entry.get("top_p", generation_config.get("top_p", 0.95)) or 0.95),
                max_tokens=int(model_entry.get("max_tokens", generation_config.get("max_tokens", 4096)) or 4096),
                timeout_seconds=int(model_entry.get("timeout_seconds", 300) or 300),
                job_timeout_seconds=_parse_optional_int(
                    model_entry.get("job_timeout_seconds"),
                    "job_timeout_seconds",
                ),
                extra_body=extra_body,
            )
            return nvidia_module.NvidiaAPIAdapter(nvidia_config)
        except Exception:
            return _UnavailableAdapter(model_id)

    if adapter_name == "llama_cpp_cli":
        try:
            llama_cpp_module = _load_framework_module(
                "benchmark_framework_llama_cpp_cli_adapter",
                "models/adapters/llama_cpp_cli.py",
            )
            generation_config = protocol_config.raw_config.get("generation", {})
            if not isinstance(generation_config, dict):
                generation_config = {}

            top_k = generation_config.get("top_k", 10)
            if top_k is None:
                top_k = 10

            llama_cpp_config = llama_cpp_module.LlamaCppCLIConfig(
                model_id=model_id,
                executable_path=str(model_entry.get("executable_path", "llama-cli")),
                model_path=str(model_entry.get("model_path", model_entry.get("weights_path", ""))),
                temperature=float(generation_config.get("temperature", 0.0) or 0.0),
                top_k=int(top_k),
                top_p=float(generation_config.get("top_p", model_entry.get("top_p", 0.95)) or 0.95),
                max_tokens=int(generation_config.get("max_tokens", 4096) or 4096),
                context_size=_parse_optional_int(model_entry.get("context_size"), "context_size"),
                threads=_parse_optional_int(model_entry.get("threads"), "threads"),
                timeout_seconds=int(model_entry.get("timeout_seconds", 300) or 300),
                extra_args=[],
            )
            return llama_cpp_module.LlamaCppCLIAdapter(llama_cpp_config)
        except Exception:
            return _UnavailableAdapter(model_id)

    return _UnavailableAdapter(model_id)


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
    task_cases: list[DiscoveredTaskCase],
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


def run_suite(
    tasks_root: str | Path | None = None,
    protocols_root: str | Path | None = None,
    prompts_root: str | Path | None = None,
    model_registry_path: str | Path | None = None,
    model_id: str | None = None,
    protocol_id: str | None = None,
    task_family: str | None = None,
    tier: str | None = None,
    instance_id: str | None = None,
    adapter_override: str | None = None,
    output_root: str | Path | None = None,
    run_id: str = "",
    adapter_factory: Callable[[dict[str, Any], LoadedProtocolConfig], Any] | None = None,
    validator: Any | None = None,
    validator_factory: Callable[[], Any] | None = None,
    use_real_validator: bool = False,
    validator_command: str | Path | None = None,
    validator_timeout_seconds: int = 30,
    validator_keep_temp_files: bool = False,
    validator_working_directory: str | Path | None = None,
    validator_extra_args: list[str] | None = None,
    preflight_tasks: bool = False,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    """Run a full benchmark campaign.

    Current behavior:
    - discover task cases from the benchmark folders
    - load protocol metadata and model registry entries
    - optionally select one model, one protocol, task filters, or override adapters for enabled models
    - build adapters and validators for each job
    - call `run_case(...)` for every matrix entry
    - aggregate normalized results

    The orchestration is intentionally lightweight and dependency-injected:
    you can pass real adapter and validator factories later without rewriting
    the suite logic.
    """
    framework_root = Path(__file__).resolve().parents[1]
    resolved_tasks_root = _resolve_framework_path(tasks_root, "tasks")
    resolved_protocols_root = _resolve_framework_path(protocols_root, "protocols")
    resolved_prompts_root = _resolve_framework_path(prompts_root, "prompts")
    resolved_model_registry_path = _resolve_framework_path(model_registry_path, "models/model_registry_nvidia.yaml")
    resolved_output_root = _resolve_framework_path(output_root, "outputs") if output_root is not None else None

    discovered_task_cases = discover_task_cases(resolved_tasks_root)
    task_cases = filter_task_cases(
        discovered_task_cases,
        task_family=task_family,
        tier=tier,
        instance_id=instance_id,
    )
    all_model_entries = load_model_registry_entries(resolved_model_registry_path)
    if model_id is not None:
        model_entries = [
            entry
            for entry in all_model_entries
            if str(entry.get("model_id", "")) == model_id
        ]
    else:
        model_entries = [
            entry
            for entry in all_model_entries
            if bool(entry.get("enabled", True))
        ]
        if adapter_override is not None:
            model_entries = [
                {
                    **entry,
                    "adapter": adapter_override,
                }
                for entry in model_entries
            ]

    model_ids = [str(entry["model_id"]) for entry in model_entries if "model_id" in entry]
    discovered_protocol_ids = discover_protocol_ids(resolved_protocols_root)
    if protocol_id is not None:
        if protocol_id not in discovered_protocol_ids:
            raise ValueError(
                f"Protocol {protocol_id!r} was not found in {resolved_protocols_root}. "
                f"Available protocols: {', '.join(discovered_protocol_ids) or 'none'}"
            )
        protocol_ids = [protocol_id]
    else:
        protocol_ids = discovered_protocol_ids
    run_matrix = build_run_matrix(task_cases, model_ids, protocol_ids)
    preflight_results: list[dict[str, Any]] = []
    orchestration_errors: list[dict[str, Any]] = []

    if preflight_tasks:
        preflight_results = run_task_preflights(
            framework_root=framework_root,
            task_cases=task_cases,
            validator_command=validator_command,
            validator_timeout_seconds=validator_timeout_seconds,
            validator_working_directory=validator_working_directory,
            validator_extra_args=validator_extra_args,
        )
        preflight_errors = _build_preflight_error_payloads(preflight_results)
        if preflight_errors:
            orchestration_errors.extend(preflight_errors)
            return {
                "summary": summarize_suite_inputs(task_cases, model_ids, protocol_ids),
                "task_cases": [asdict(task_case) for task_case in task_cases],
                "protocol_ids": protocol_ids,
                "model_ids": model_ids,
                "run_matrix": [asdict(job) for job in run_matrix],
                "preflight_results": preflight_results,
                "suite_results": [],
                "aggregate_results": aggregate_suite_results([]),
                "orchestration_errors": orchestration_errors,
            }

    model_lookup = {
        str(entry["model_id"]): entry
        for entry in model_entries
        if "model_id" in entry
    }

    run_case_module = _load_framework_module(
        "benchmark_framework_run_case",
        "runner/run_case.py",
    )

    suite_results: list[dict[str, Any]] = []

    total_jobs = len(run_matrix)
    for job_index, job in enumerate(run_matrix, start=1):
        task_label = f"{job.task_case.task_family}/{job.task_case.tier}/{job.task_case.instance_id}"
        print(
            f"[{job_index}/{total_jobs}] START "
            f"model={job.model_id} protocol={job.protocol_id} task={task_label}",
            flush=True,
        )
        try:
            protocol_config = load_protocol_config(job.protocol_id, resolved_protocols_root)
            protocol_spec = build_protocol_spec(protocol_config)
            task_spec = build_task_spec_from_case(job.task_case)
            prompt_bundle = load_prompt_bundle(
                task_family=job.task_case.task_family,
                protocol_config=protocol_config,
                prompts_root=resolved_prompts_root,
            )
            model_entry = model_lookup.get(job.model_id, {"model_id": job.model_id})
            adapter = build_model_adapter(
                model_entry=model_entry,
                protocol_config=protocol_config,
                adapter_factory=adapter_factory,
            )
            current_validator = build_validator(
                framework_root=framework_root,
                validator=validator,
                validator_factory=validator_factory,
                use_real_validator=use_real_validator,
                validator_command=validator_command,
                validator_timeout_seconds=validator_timeout_seconds,
                validator_keep_temp_files=validator_keep_temp_files,
                validator_working_directory=validator_working_directory,
                validator_extra_args=validator_extra_args,
            )

            result = run_case_module.run_case(
                model_id=job.model_id,
                adapter=adapter,
                validator=current_validator,
                task_spec=task_spec,
                protocol_spec=protocol_spec,
                system_prompt=prompt_bundle.system_prompt,
                domain_prompt=prompt_bundle.domain_prompt,
                feedback_prompt=prompt_bundle.feedback_prompt,
                output_root=resolved_output_root,
                run_id=run_id,
            )
            normalized_result = _normalize_record(result)
            suite_results.append(normalized_result)
            solved = normalized_result.get("solved")
            iterations_used = normalized_result.get("iterations_used")
            print(
                f"[{job_index}/{total_jobs}] DONE "
                f"model={job.model_id} protocol={job.protocol_id} task={task_label} "
                f"solved={solved} iterations={iterations_used}",
                flush=True,
            )
        except Exception as exc:
            error_payload = {
                "model_id": job.model_id,
                "protocol_id": job.protocol_id,
                "task_family": job.task_case.task_family,
                "tier": job.task_case.tier,
                "instance_id": job.task_case.instance_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            }
            orchestration_errors.append(error_payload)
            print(
                f"[{job_index}/{total_jobs}] ERROR "
                f"model={job.model_id} protocol={job.protocol_id} task={task_label} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            if stop_on_error:
                raise

    return {
        "summary": summarize_suite_inputs(task_cases, model_ids, protocol_ids),
        "task_cases": [asdict(task_case) for task_case in task_cases],
        "protocol_ids": protocol_ids,
        "model_ids": model_ids,
        "run_matrix": [asdict(job) for job in run_matrix],
        "preflight_results": preflight_results,
        "suite_results": [_build_suite_result_summary(result) for result in suite_results],
        "aggregate_results": aggregate_suite_results(suite_results),
        "orchestration_errors": orchestration_errors,
    }
