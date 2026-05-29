"""CLI entry point for launching the benchmark suite.

Example:
    python Benchmark_Framework/run_benchmark.py --use-real-validator
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


DEFAULT_MODEL_REGISTRY_PATH = "models/model_registry_nvidia.yaml"

ADAPTER_REGISTRY_PATHS = {
    "nvidia": "models/model_registry_nvidia.yaml",
    "nvidia_api": "models/model_registry_nvidia.yaml",
    "hf": "models/model_registry_hf.yaml",
    "hf_local": "models/model_registry_hf.yaml",
    "huggingface": "models/model_registry_hf.yaml",
    "ollama": "models/model_registry_ollama.yaml",
    "llama_cpp": "models/model_registry_llama_cpp.yaml",
    "llama_cpp_cli": "models/model_registry_llama_cpp.yaml",
}


def _load_module(module_name: str, path: Path):
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json_safe(value: Any) -> Any:
    """Convert framework outputs into JSON-serializable data."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _resolve_model_registry_path(model_registry_path: str | None, adapter: str | None) -> str:
    """Resolve the registry selected by explicit path, adapter shortcut, or default."""
    if model_registry_path:
        return model_registry_path
    if adapter:
        return ADAPTER_REGISTRY_PATHS[adapter]
    return DEFAULT_MODEL_REGISTRY_PATH


def _build_run_id() -> str:
    """Return a filesystem-friendly timestamp for one benchmark launch."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _resolve_output_base(output_root: str | Path, framework_root: Path) -> Path:
    """Resolve the base output directory relative to the framework root."""
    output_base = Path(output_root)
    if output_base.is_absolute():
        return output_base
    return framework_root / output_base


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LLM planning benchmark suite.")
    parser.add_argument("--tasks-root", default="tasks", help="Task root relative to Benchmark_Framework.")
    parser.add_argument("--protocols-root", default="protocols", help="Protocols root relative to Benchmark_Framework.")
    parser.add_argument("--prompts-root", default="prompts", help="Prompts root relative to Benchmark_Framework.")
    parser.add_argument(
        "--model-registry-path",
        default=None,
        help="Manual model registry path relative to Benchmark_Framework. Overrides --adapter.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Run only one model from the registry, using its YAML adapter.",
    )
    parser.add_argument(
        "--protocol-id",
        default=None,
        help="Run only one protocol from the protocols folder.",
    )
    parser.add_argument(
        "--task-family",
        default=None,
        help="Run only one task family, for example <task_family>.",
    )
    parser.add_argument(
        "--tier",
        choices=("easy", "medium", "hard"),
        default=None,
        help="Run only one difficulty tier.",
    )
    parser.add_argument(
        "--instance-id",
        default=None,
        help="Run only one instance id, for example instance-01.",
    )
    parser.add_argument(
        "--adapter",
        choices=sorted(ADAPTER_REGISTRY_PATHS),
        default=None,
        help="Select the matching model registry automatically, for example hf_local, nvidia_api, ollama, or llama_cpp_cli.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Base output directory relative to Benchmark_Framework.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Optional run folder name. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--use-real-validator",
        action="store_true",
        help="Use the real VAL-backed validator instead of the unavailable fallback.",
    )
    parser.add_argument(
        "--validator-command",
        default=None,
        help="Validator command or full executable path. Defaults to auto-resolution.",
    )
    parser.add_argument(
        "--validator-timeout-seconds",
        type=int,
        default=30,
        help="Timeout used by the real validator.",
    )
    parser.add_argument(
        "--preflight-tasks",
        action="store_true",
        help="Check task domain/problem files with VAL before launching model jobs.",
    )
    parser.add_argument(
        "--parallel-nvidia-models",
        action="store_true",
        help=(
            "Run NVIDIA API models in parallel lanes. Each model lane still runs "
            "selected protocols and task instances sequentially."
        ),
    )
    parser.add_argument(
        "--max-concurrent-nvidia-models",
        type=int,
        default=None,
        help=(
            "Maximum number of NVIDIA model lanes to run at once when "
            "--parallel-nvidia-models is enabled. Defaults to all selected "
            "NVIDIA models."
        ),
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the suite immediately when one job raises an orchestration error.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional explicit suite JSON path. Defaults to outputs/scored/<run_id>/suite_result.json.",
    )
    parser.add_argument(
        "--print-full-result",
        action="store_true",
        help="Print the full suite JSON instead of only the summary and aggregate results.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    framework_root = Path(__file__).resolve().parent
    run_suite_module = _load_module(
        "benchmark_framework_run_suite_entrypoint",
        framework_root / "runner" / "run_suite.py",
    )
    resolved_model_registry_path = _resolve_model_registry_path(
        model_registry_path=args.model_registry_path,
        adapter=args.adapter,
    )
    run_id = args.run_id or _build_run_id()
    output_root = _resolve_output_base(args.output_root, framework_root)

    result = run_suite_module.run_suite(
        tasks_root=args.tasks_root,
        protocols_root=args.protocols_root,
        prompts_root=args.prompts_root,
        model_registry_path=resolved_model_registry_path,
        model_id=args.model_id,
        protocol_id=args.protocol_id,
        task_family=args.task_family,
        tier=args.tier,
        instance_id=args.instance_id,
        output_root=output_root,
        run_id=run_id,
        use_real_validator=args.use_real_validator,
        validator_command=args.validator_command,
        validator_timeout_seconds=args.validator_timeout_seconds,
        preflight_tasks=args.preflight_tasks,
        parallel_nvidia_models=args.parallel_nvidia_models,
        max_concurrent_nvidia_models=args.max_concurrent_nvidia_models,
        stop_on_error=args.stop_on_error,
    )

    serializable_result = _json_safe(result)

    output_path = (
        Path(args.output_json)
        if args.output_json
        else output_root / "scored" / run_id / "suite_result.json"
    )
    if not output_path.is_absolute():
        output_path = framework_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(serializable_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.print_full_result:
        print(json.dumps(serializable_result, indent=2, ensure_ascii=False))
    else:
        summary_payload = {
            "summary": serializable_result.get("summary", {}),
            "aggregate_results": serializable_result.get("aggregate_results", {}),
            "orchestration_errors": serializable_result.get("orchestration_errors", []),
            "nvidia_lane_log_paths": serializable_result.get("nvidia_lane_log_paths", {}),
            "run_id": run_id,
            "output_root": str(output_root),
            "saved_to": str(output_path),
        }
        print(json.dumps(summary_payload, indent=2, ensure_ascii=False))

    orchestration_errors = serializable_result.get("orchestration_errors", [])
    if orchestration_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
