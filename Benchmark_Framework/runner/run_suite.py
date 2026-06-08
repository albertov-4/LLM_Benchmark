"""Suite-level benchmark orchestration based on directory discovery."""

from __future__ import annotations

import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Callable, TextIO


@lru_cache(maxsize=None)
def _load_framework_module(module_key: str, relative_path: str):
    """Load a sibling framework module without requiring package installation."""
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = spec_from_file_location(module_key, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")

    module = module_from_spec(spec)
    sys.modules[module_key] = module
    spec.loader.exec_module(module)
    return module


_suite_discovery = _load_framework_module(
    "benchmark_framework_suite_discovery",
    "runner/suite_discovery.py",
)
_suite_config = _load_framework_module(
    "benchmark_framework_suite_config",
    "runner/suite_config.py",
)
_suite_adapters = _load_framework_module(
    "benchmark_framework_suite_adapters",
    "runner/suite_adapters.py",
)
_suite_validators = _load_framework_module(
    "benchmark_framework_suite_validators",
    "runner/suite_validators.py",
)
_suite_aggregation = _load_framework_module(
    "benchmark_framework_suite_aggregation",
    "runner/suite_aggregation.py",
)
_config_loader = _load_framework_module(
    "benchmark_framework_config_loader",
    "utils/config_loader.py",
)


TIERS = _suite_discovery.TIERS
DiscoveredTaskCase = _suite_discovery.DiscoveredTaskCase
SuiteJob = _suite_discovery.SuiteJob
discover_task_cases = _suite_discovery.discover_task_cases
filter_task_cases = _suite_discovery.filter_task_cases
discover_protocol_ids = _suite_discovery.discover_protocol_ids
build_run_matrix = _suite_discovery.build_run_matrix
summarize_suite_inputs = _suite_discovery.summarize_suite_inputs
_task_case_key = _suite_discovery.task_case_key

LoadedProtocolConfig = _suite_config.LoadedProtocolConfig
LoadedPromptBundle = _suite_config.LoadedPromptBundle
load_model_registry_entries = _suite_config.load_model_registry_entries
load_protocol_config = _suite_config.load_protocol_config
load_prompt_bundle = _suite_config.load_prompt_bundle
_read_text_if_exists = _suite_config._read_text_if_exists
_read_required_text = _suite_config._read_required_text

_UnavailableAdapter = _suite_adapters._UnavailableAdapter
_parse_optional_int = _suite_adapters._parse_optional_int
build_model_adapter = _suite_adapters.build_model_adapter

_UnavailableValidator = _suite_validators._UnavailableValidator
_resolve_validate_command = _suite_validators._resolve_validate_command
build_real_val_validator = _suite_validators.build_real_val_validator
build_validator = _suite_validators.build_validator
run_task_preflights = _suite_validators.run_task_preflights
_build_preflight_error_payloads = _suite_validators._build_preflight_error_payloads

_normalize_record = _suite_aggregation._normalize_record
_build_suite_result_summary = _suite_aggregation._build_suite_result_summary
_new_aggregate_bucket = _suite_aggregation._new_aggregate_bucket
_update_aggregate_bucket = _suite_aggregation._update_aggregate_bucket
_finalize_aggregate_bucket = _suite_aggregation._finalize_aggregate_bucket
aggregate_suite_results = _suite_aggregation.aggregate_suite_results

_parse_scalar = _config_loader.parse_scalar


class _ThreadLocalStdout:
    """Route writes to a thread-local stream while preserving normal stdout elsewhere."""

    def __init__(self, fallback: TextIO, local_state: threading.local):
        self._fallback = fallback
        self._local_state = local_state

    def _stream(self) -> TextIO:
        stream = getattr(self._local_state, "stream", None)
        if stream is None:
            return self._fallback
        return stream

    def write(self, text: str) -> int:
        return self._stream().write(text)

    def flush(self) -> None:
        self._stream().flush()

    def isatty(self) -> bool:
        return self._stream().isatty()

    @property
    def encoding(self) -> str | None:
        return getattr(self._stream(), "encoding", None)

    @property
    def errors(self) -> str | None:
        return getattr(self._stream(), "errors", None)


def _resolve_framework_path(path_value: str | Path | None, default_relative: str) -> Path:
    """Resolve benchmark paths relative to the framework root by default."""
    framework_root = Path(__file__).resolve().parents[1]
    if path_value is None:
        return framework_root / default_relative

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return framework_root / candidate


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


def _is_nvidia_api_model(model_entry: dict[str, Any] | None) -> bool:
    """Return whether a model entry uses the NVIDIA API adapter."""
    if not model_entry:
        return False
    return str(model_entry.get("adapter", "")).strip() == "nvidia_api"


def _run_suite_job(
    *,
    job_index: int,
    total_jobs: int,
    job: SuiteJob,
    framework_root: Path,
    resolved_protocols_root: Path,
    resolved_prompts_root: Path,
    resolved_output_root: Path | None,
    model_lookup: dict[str, dict[str, Any]],
    run_case_module: Any,
    adapter_factory: Callable[[dict[str, Any], LoadedProtocolConfig], Any] | None,
    validator: Any | None,
    validator_factory: Callable[[], Any] | None,
    use_real_validator: bool,
    validator_command: str | Path | None,
    validator_timeout_seconds: int,
    validator_keep_temp_files: bool,
    validator_working_directory: str | Path | None,
    validator_extra_args: list[str] | None,
    run_id: str,
) -> tuple[int, dict[str, Any] | None, dict[str, Any] | None, Exception | None]:
    """Run one suite job and return either a normalized result or orchestration error."""
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
        solved = normalized_result.get("solved")
        iterations_used = normalized_result.get("iterations_used")
        print(
            f"[{job_index}/{total_jobs}] DONE "
            f"model={job.model_id} protocol={job.protocol_id} task={task_label} "
            f"solved={solved} iterations={iterations_used}",
            flush=True,
        )
        return job_index, normalized_result, None, None
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
        print(
            f"[{job_index}/{total_jobs}] ERROR "
            f"model={job.model_id} protocol={job.protocol_id} task={task_label} "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return job_index, None, error_payload, exc


def _run_nvidia_model_lanes(
    *,
    task_cases: list[DiscoveredTaskCase],
    model_ids: list[str],
    protocol_ids: list[str],
    run_matrix: list[SuiteJob],
    framework_root: Path,
    resolved_protocols_root: Path,
    resolved_prompts_root: Path,
    resolved_output_root: Path | None,
    model_lookup: dict[str, dict[str, Any]],
    run_case_module: Any,
    adapter_factory: Callable[[dict[str, Any], LoadedProtocolConfig], Any] | None,
    validator: Any | None,
    validator_factory: Callable[[], Any] | None,
    use_real_validator: bool,
    validator_command: str | Path | None,
    validator_timeout_seconds: int,
    validator_keep_temp_files: bool,
    validator_working_directory: str | Path | None,
    validator_extra_args: list[str] | None,
    run_id: str,
    max_concurrent_nvidia_models: int | None,
    stop_on_error: bool,
) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]], dict[str, str]]:
    """Run NVIDIA API models in parallel lanes and keep non-NVIDIA jobs sequential."""
    if max_concurrent_nvidia_models is not None and max_concurrent_nvidia_models < 1:
        raise ValueError("--max-concurrent-nvidia-models must be at least 1.")

    total_jobs = len(run_matrix)
    nvidia_model_ids = [
        model_id
        for model_id in model_ids
        if _is_nvidia_api_model(model_lookup.get(model_id))
    ]
    if nvidia_model_ids and validator is not None and validator_factory is None:
        raise ValueError(
            "parallel_nvidia_models cannot use one shared validator instance. "
            "Pass validator_factory or let the runner create validators per job."
        )
    nvidia_model_id_set = set(nvidia_model_ids)
    lane_log_paths: dict[str, str] = {}
    lane_logs_root: Path | None = None
    if resolved_output_root is not None:
        lane_logs_root = resolved_output_root / "logs" / (run_id or "default")
        lane_logs_root.mkdir(parents=True, exist_ok=True)
        lane_log_paths = {
            model_id: str(lane_logs_root / f"{model_id}.log")
            for model_id in nvidia_model_ids
        }

    job_index_lookup = {
        (job.model_id, job.protocol_id, _task_case_key(job.task_case)): job_index
        for job_index, job in enumerate(run_matrix, start=1)
    }
    result_records: list[tuple[int, dict[str, Any]]] = []
    error_records: list[tuple[int, dict[str, Any]]] = []

    for job_index, job in enumerate(run_matrix, start=1):
        if job.model_id in nvidia_model_id_set:
            continue
        _, result, error, exc = _run_suite_job(
            job_index=job_index,
            total_jobs=total_jobs,
            job=job,
            framework_root=framework_root,
            resolved_protocols_root=resolved_protocols_root,
            resolved_prompts_root=resolved_prompts_root,
            resolved_output_root=resolved_output_root,
            model_lookup=model_lookup,
            run_case_module=run_case_module,
            adapter_factory=adapter_factory,
            validator=validator,
            validator_factory=validator_factory,
            use_real_validator=use_real_validator,
            validator_command=validator_command,
            validator_timeout_seconds=validator_timeout_seconds,
            validator_keep_temp_files=validator_keep_temp_files,
            validator_working_directory=validator_working_directory,
            validator_extra_args=validator_extra_args,
            run_id=run_id,
        )
        if result is not None:
            result_records.append((job_index, result))
        if error is not None:
            error_records.append((job_index, error))
            if stop_on_error and exc is not None:
                raise exc

    if not nvidia_model_ids:
        return result_records, error_records, lane_log_paths

    max_workers = min(
        max_concurrent_nvidia_models or len(nvidia_model_ids),
        len(nvidia_model_ids),
    )
    stop_event = threading.Event()
    stdout_local = threading.local()
    original_stdout = sys.stdout
    stdout_proxy = _ThreadLocalStdout(original_stdout, stdout_local)

    if lane_logs_root is not None:
        print(f"[NVIDIA LANES] Detailed per-model logs: {lane_logs_root}", flush=True)

    def run_model_lane(model_id: str) -> tuple[list[tuple[int, dict[str, Any]]], list[tuple[int, dict[str, Any]]]]:
        lane_results: list[tuple[int, dict[str, Any]]] = []
        lane_errors: list[tuple[int, dict[str, Any]]] = []
        log_path = Path(lane_log_paths[model_id]) if model_id in lane_log_paths else None
        log_stream: TextIO | None = None

        if log_path is not None:
            log_stream = log_path.open("w", encoding="utf-8")
            stdout_local.stream = log_stream

        try:
            print(f"[NVIDIA LANE START] model={model_id}", flush=True)
            for protocol_id in protocol_ids:
                for task_case in task_cases:
                    if stop_event.is_set():
                        return lane_results, lane_errors

                    job = SuiteJob(
                        model_id=model_id,
                        protocol_id=protocol_id,
                        task_case=task_case,
                    )
                    job_index = job_index_lookup[(model_id, protocol_id, _task_case_key(task_case))]
                    _, result, error, exc = _run_suite_job(
                        job_index=job_index,
                        total_jobs=total_jobs,
                        job=job,
                        framework_root=framework_root,
                        resolved_protocols_root=resolved_protocols_root,
                        resolved_prompts_root=resolved_prompts_root,
                        resolved_output_root=resolved_output_root,
                        model_lookup=model_lookup,
                        run_case_module=run_case_module,
                        adapter_factory=adapter_factory,
                        validator=validator,
                        validator_factory=validator_factory,
                        use_real_validator=use_real_validator,
                        validator_command=validator_command,
                        validator_timeout_seconds=validator_timeout_seconds,
                        validator_keep_temp_files=validator_keep_temp_files,
                        validator_working_directory=validator_working_directory,
                        validator_extra_args=validator_extra_args,
                        run_id=run_id,
                    )
                    if result is not None:
                        lane_results.append((job_index, result))
                    if error is not None:
                        lane_errors.append((job_index, error))
                        if stop_on_error:
                            stop_event.set()
                            return lane_results, lane_errors

            print(f"[NVIDIA LANE DONE] model={model_id}", flush=True)
            return lane_results, lane_errors
        finally:
            if log_stream is not None:
                log_stream.flush()
                log_stream.close()
                stdout_local.stream = None

    sys.stdout = stdout_proxy
    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_model_id = {
                executor.submit(run_model_lane, model_id): model_id
                for model_id in nvidia_model_ids
            }
            for future in as_completed(future_to_model_id):
                model_id = future_to_model_id[future]
                lane_results, lane_errors = future.result()
                result_records.extend(lane_results)
                error_records.extend(lane_errors)
                status = "errors" if lane_errors else "ok"
                log_path_text = lane_log_paths.get(model_id)
                if log_path_text:
                    print(
                        f"[NVIDIA LANE DONE] model={model_id} status={status} "
                        f"log={log_path_text}",
                        flush=True,
                    )
                else:
                    print(
                        f"[NVIDIA LANE DONE] model={model_id} status={status}",
                        flush=True,
                    )
    finally:
        sys.stdout = original_stdout

    return result_records, error_records, lane_log_paths


def _select_model_entries(
    all_model_entries: list[dict[str, Any]],
    model_id: str | None,
    adapter_override: str | None,
) -> list[dict[str, Any]]:
    """Apply model filters while preserving the existing override semantics."""
    if model_id is not None:
        return [
            entry
            for entry in all_model_entries
            if str(entry.get("model_id", "")) == model_id
        ]

    model_entries = [
        entry
        for entry in all_model_entries
        if bool(entry.get("enabled", True))
    ]
    if adapter_override is not None:
        return [
            {
                **entry,
                "adapter": adapter_override,
            }
            for entry in model_entries
        ]
    return model_entries


def _select_protocol_ids(protocol_id: str | None, discovered_protocol_ids: list[str], protocols_root: Path) -> list[str]:
    """Apply the optional protocol filter with the same error message as before."""
    if protocol_id is None:
        return discovered_protocol_ids
    if protocol_id not in discovered_protocol_ids:
        raise ValueError(
            f"Protocol {protocol_id!r} was not found in {protocols_root}. "
            f"Available protocols: {', '.join(discovered_protocol_ids) or 'none'}"
        )
    return [protocol_id]


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
    parallel_nvidia_models: bool = False,
    max_concurrent_nvidia_models: int | None = None,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    """Run a full benchmark campaign."""
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
    model_entries = _select_model_entries(
        all_model_entries=load_model_registry_entries(resolved_model_registry_path),
        model_id=model_id,
        adapter_override=adapter_override,
    )
    model_ids = [str(entry["model_id"]) for entry in model_entries if "model_id" in entry]
    protocol_ids = _select_protocol_ids(
        protocol_id=protocol_id,
        discovered_protocol_ids=discover_protocol_ids(resolved_protocols_root),
        protocols_root=resolved_protocols_root,
    )
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

    suite_result_records: list[tuple[int, dict[str, Any]]] = []
    orchestration_error_records: list[tuple[int, dict[str, Any]]] = []
    nvidia_lane_log_paths: dict[str, str] = {}

    if parallel_nvidia_models:
        (
            suite_result_records,
            orchestration_error_records,
            nvidia_lane_log_paths,
        ) = _run_nvidia_model_lanes(
            task_cases=task_cases,
            model_ids=model_ids,
            protocol_ids=protocol_ids,
            run_matrix=run_matrix,
            framework_root=framework_root,
            resolved_protocols_root=resolved_protocols_root,
            resolved_prompts_root=resolved_prompts_root,
            resolved_output_root=resolved_output_root,
            model_lookup=model_lookup,
            run_case_module=run_case_module,
            adapter_factory=adapter_factory,
            validator=validator,
            validator_factory=validator_factory,
            use_real_validator=use_real_validator,
            validator_command=validator_command,
            validator_timeout_seconds=validator_timeout_seconds,
            validator_keep_temp_files=validator_keep_temp_files,
            validator_working_directory=validator_working_directory,
            validator_extra_args=validator_extra_args,
            run_id=run_id,
            max_concurrent_nvidia_models=max_concurrent_nvidia_models,
            stop_on_error=stop_on_error,
        )
    else:
        total_jobs = len(run_matrix)
        for job_index, job in enumerate(run_matrix, start=1):
            _, result, error, exc = _run_suite_job(
                job_index=job_index,
                total_jobs=total_jobs,
                job=job,
                framework_root=framework_root,
                resolved_protocols_root=resolved_protocols_root,
                resolved_prompts_root=resolved_prompts_root,
                resolved_output_root=resolved_output_root,
                model_lookup=model_lookup,
                run_case_module=run_case_module,
                adapter_factory=adapter_factory,
                validator=validator,
                validator_factory=validator_factory,
                use_real_validator=use_real_validator,
                validator_command=validator_command,
                validator_timeout_seconds=validator_timeout_seconds,
                validator_keep_temp_files=validator_keep_temp_files,
                validator_working_directory=validator_working_directory,
                validator_extra_args=validator_extra_args,
                run_id=run_id,
            )
            if result is not None:
                suite_result_records.append((job_index, result))
            if error is not None:
                orchestration_error_records.append((job_index, error))
                if stop_on_error and exc is not None:
                    raise exc

    suite_results = [
        result
        for _, result in sorted(suite_result_records, key=lambda item: item[0])
    ]
    orchestration_errors.extend(
        error
        for _, error in sorted(orchestration_error_records, key=lambda item: item[0])
    )

    return {
        "summary": summarize_suite_inputs(task_cases, model_ids, protocol_ids),
        "task_cases": [asdict(task_case) for task_case in task_cases],
        "protocol_ids": protocol_ids,
        "model_ids": model_ids,
        "run_matrix": [asdict(job) for job in run_matrix],
        "preflight_results": preflight_results,
        "nvidia_lane_log_paths": nvidia_lane_log_paths,
        "suite_results": [_build_suite_result_summary(result) for result in suite_results],
        "aggregate_results": aggregate_suite_results(suite_results),
        "orchestration_errors": orchestration_errors,
    }
