"""CLI entry point for launching the benchmark suite.

Example:
    python "Benchmark Framework/run_benchmark.py" --use-real-validator
"""

from __future__ import annotations

import argparse
import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the LLM planning benchmark suite.")
    parser.add_argument("--tasks-root", default="tasks", help="Task root relative to Benchmark Framework.")
    parser.add_argument("--protocols-root", default="protocols", help="Protocols root relative to Benchmark Framework.")
    parser.add_argument("--prompts-root", default="prompts", help="Prompts root relative to Benchmark Framework.")
    parser.add_argument(
        "--model-registry-path",
        default="models/model_registry.yaml",
        help="Model registry path relative to Benchmark Framework.",
    )
    parser.add_argument(
        "--output-root",
        default="outputs",
        help="Per-run artifact root relative to Benchmark Framework.",
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
        "--stop-on-error",
        action="store_true",
        help="Abort the suite immediately when one job raises an orchestration error.",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/scored/suite_result_latest.json",
        help="Where to save the suite result JSON, relative to Benchmark Framework.",
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

    result = run_suite_module.run_suite(
        tasks_root=args.tasks_root,
        protocols_root=args.protocols_root,
        prompts_root=args.prompts_root,
        model_registry_path=args.model_registry_path,
        output_root=args.output_root,
        use_real_validator=args.use_real_validator,
        validator_command=args.validator_command,
        validator_timeout_seconds=args.validator_timeout_seconds,
        stop_on_error=args.stop_on_error,
    )

    serializable_result = _json_safe(result)

    output_path = framework_root / args.output_json
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
            "saved_to": str(output_path),
        }
        print(json.dumps(summary_payload, indent=2, ensure_ascii=False))

    orchestration_errors = serializable_result.get("orchestration_errors", [])
    if orchestration_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
