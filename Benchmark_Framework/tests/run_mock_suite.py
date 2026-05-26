"""Manual entry point for running the benchmark suite with mock components."""

import importlib.util
import json
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    framework_root = Path(__file__).resolve().parents[1]
    fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"

    run_suite_module = _load_module(
        "benchmark_framework_manual_run_suite_module",
        framework_root / "runner" / "run_suite.py",
    )
    mock_adapter_module = _load_module(
        "benchmark_framework_manual_mock_adapter_module",
        framework_root / "tests" / "mocks" / "mock_adapter.py",
    )
    mock_validator_module = _load_module(
        "benchmark_framework_manual_mock_validator_module",
        framework_root / "tests" / "mocks" / "mock_validator.py",
    )

    result = run_suite_module.run_suite(
        tasks_root=fixtures_root / "tasks",
        protocols_root=fixtures_root / "protocols",
        model_registry_path=fixtures_root / "models" / "model_registry.yaml",
        adapter_factory=mock_adapter_module.build_mock_adapter_for_suite,
        validator_factory=mock_validator_module.build_mock_validator_for_suite,
    )

    print("Mock suite completed.")
    print()
    print("Summary:")
    print(json.dumps(result["summary"], indent=2))
    print()
    print("Aggregate results:")
    print(json.dumps(result["aggregate_results"], indent=2))
    print()
    print("Suite results:")
    print(json.dumps(result["suite_results"], indent=2))


if __name__ == "__main__":
    main()
