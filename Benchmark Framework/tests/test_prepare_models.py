"""Tests for the model preparation script."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PrepareModelsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        cls.prepare_module = _load_module(
            "benchmark_framework_test_prepare_models_module",
            framework_root / "scripts" / "prepare_models.py",
        )

    def _write_registry(self, directory: Path, content: str) -> Path:
        registry_path = directory / "model_registry.yaml"
        registry_path.write_text(content.strip() + "\n", encoding="utf-8")
        return registry_path

    def test_existing_local_weights_path_is_detected(self) -> None:
        module = self.prepare_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            local_model = root / "local_model"
            local_model.mkdir()
            registry_path = self._write_registry(
                root,
                f"""
models:
  - model_id: local_model
    adapter: hf_local
    provider: huggingface_local
    enabled: true
    weights_path: {local_model}
""",
            )

            results = module.prepare_models_from_registry(
                model_registry_path=registry_path,
                models_dir=root / "models_cache",
            )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].status, "available")

    def test_hf_repo_id_is_recognized_as_downloadable(self) -> None:
        module = self.prepare_module

        self.assertTrue(module.looks_like_hf_repo_id("Qwen/Qwen2.5-1.5B-Instruct"))
        self.assertFalse(module.looks_like_hf_repo_id("tinyllama_1_1b_chat"))
        self.assertFalse(module.looks_like_hf_repo_id("C:/models/qwen"))

    def test_offline_mode_fails_when_repo_model_is_missing(self) -> None:
        module = self.prepare_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            registry_path = self._write_registry(
                root,
                """
models:
  - model_id: qwen
    adapter: hf_local
    provider: huggingface_local
    enabled: true
    weights_path: Qwen/Qwen2.5-1.5B-Instruct
""",
            )

            results = module.prepare_models_from_registry(
                model_registry_path=registry_path,
                models_dir=root / "models_cache",
                offline=True,
            )

        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)
        self.assertEqual(results[0].status, "missing_offline")
        self.assertIn("Expected prepared directory", results[0].message)

    def test_non_hf_local_adapter_is_skipped(self) -> None:
        module = self.prepare_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            registry_path = self._write_registry(
                root,
                """
models:
  - model_id: ollama_qwen
    adapter: ollama
    provider: ollama_local
    enabled: true
    ollama_model: qwen2.5:0.5b
""",
            )

            results = module.prepare_models_from_registry(
                model_registry_path=registry_path,
                models_dir=root / "models_cache",
            )

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].status, "skipped")
        self.assertIn("managed by Ollama", results[0].message)

    def test_dry_run_does_not_download(self) -> None:
        module = self.prepare_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            registry_path = self._write_registry(
                root,
                """
models:
  - model_id: qwen
    adapter: hf_local
    provider: huggingface_local
    enabled: true
    weights_path: Qwen/Qwen2.5-1.5B-Instruct
""",
            )

            original_download = module.download_hf_snapshot

            def fail_download(repo_id, local_dir):
                raise AssertionError("download_hf_snapshot should not be called in dry-run mode")

            module.download_hf_snapshot = fail_download
            try:
                results = module.prepare_models_from_registry(
                    model_registry_path=registry_path,
                    models_dir=root / "models_cache",
                    dry_run=True,
                )
            finally:
                module.download_hf_snapshot = original_download

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].success)
        self.assertEqual(results[0].status, "dry_run")
        self.assertIn("would download", results[0].message)


if __name__ == "__main__":
    unittest.main()
