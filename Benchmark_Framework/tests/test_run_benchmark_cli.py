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

    def test_relative_output_root_is_resolved_inside_framework(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        module = _load_module(
            "benchmark_framework_test_run_benchmark_output_base",
            framework_root / "run_benchmark.py",
        )

        self.assertEqual(
            module._resolve_output_base("outputs", framework_root),
            framework_root / "outputs",
        )

    def test_preflight_tasks_flag_defaults_to_false_and_can_be_enabled(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        module = _load_module(
            "benchmark_framework_test_run_benchmark_preflight_flag",
            framework_root / "run_benchmark.py",
        )

        parser = module._build_parser()

        self.assertFalse(parser.parse_args([]).preflight_tasks)
        self.assertTrue(parser.parse_args(["--preflight-tasks"]).preflight_tasks)

    def test_parallel_nvidia_flags_default_to_disabled_and_parse_limit(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        module = _load_module(
            "benchmark_framework_test_run_benchmark_parallel_nvidia_flags",
            framework_root / "run_benchmark.py",
        )

        parser = module._build_parser()
        defaults = parser.parse_args([])
        enabled = parser.parse_args(
            ["--parallel-nvidia-models", "--max-concurrent-nvidia-models", "3"]
        )

        self.assertFalse(defaults.parallel_nvidia_models)
        self.assertIsNone(defaults.max_concurrent_nvidia_models)
        self.assertTrue(enabled.parallel_nvidia_models)
        self.assertEqual(enabled.max_concurrent_nvidia_models, 3)


if __name__ == "__main__":
    unittest.main()
