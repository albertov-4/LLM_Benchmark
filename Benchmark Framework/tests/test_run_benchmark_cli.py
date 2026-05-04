"""Tests for the benchmark CLI argument resolution."""

import importlib.util
import unittest
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RunBenchmarkCLITest(unittest.TestCase):
    def test_default_registry_is_nvidia(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        module = _load_module(
            "benchmark_framework_test_run_benchmark_default",
            framework_root / "run_benchmark.py",
        )

        self.assertEqual(
            module._resolve_model_registry_path(model_registry_path=None, adapter=None),
            "models/model_registry_nvidia.yaml",
        )

    def test_adapter_selects_matching_registry(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        module = _load_module(
            "benchmark_framework_test_run_benchmark_adapter",
            framework_root / "run_benchmark.py",
        )

        self.assertEqual(
            module._resolve_model_registry_path(model_registry_path=None, adapter="hf_local"),
            "models/model_registry_hf.yaml",
        )
        self.assertEqual(
            module._resolve_model_registry_path(model_registry_path=None, adapter="llama_cpp_cli"),
            "models/model_registry_llama_cpp.yaml",
        )

    def test_manual_registry_path_overrides_adapter(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        module = _load_module(
            "benchmark_framework_test_run_benchmark_manual",
            framework_root / "run_benchmark.py",
        )

        self.assertEqual(
            module._resolve_model_registry_path(
                model_registry_path="models/custom_registry.yaml",
                adapter="hf_local",
            ),
            "models/custom_registry.yaml",
        )


if __name__ == "__main__":
    unittest.main()
