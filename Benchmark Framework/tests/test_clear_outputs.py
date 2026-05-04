"""Tests for the output cleanup helper."""

import importlib.util
import tempfile
import unittest
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClearOutputsTest(unittest.TestCase):
    def test_collect_and_delete_outputs_preserves_gitkeep(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        module = _load_module(
            "benchmark_framework_test_clear_outputs",
            framework_root / "clear_outputs.py",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            outputs_root = Path(tmp_dir) / "outputs"
            raw_root = outputs_root / "raw"
            parsed_root = outputs_root / "parsed"
            scored_root = outputs_root / "scored"
            raw_root.mkdir(parents=True)
            parsed_root.mkdir(parents=True)
            scored_root.mkdir(parents=True)

            (raw_root / ".gitkeep").write_text("", encoding="utf-8")
            (parsed_root / ".gitkeep").write_text("", encoding="utf-8")
            (scored_root / ".gitkeep").write_text("", encoding="utf-8")
            (raw_root / "model_a").mkdir()
            (raw_root / "model_a" / "artifact.json").write_text("{}", encoding="utf-8")
            (parsed_root / "model_a").mkdir()
            (scored_root / "suite_result_latest.json").write_text("{}", encoding="utf-8")

            targets = module.collect_output_targets(outputs_root)

            self.assertEqual(
                {target.name for target in targets},
                {"model_a", "suite_result_latest.json"},
            )

            module.delete_targets(targets)

            self.assertTrue((raw_root / ".gitkeep").exists())
            self.assertTrue((parsed_root / ".gitkeep").exists())
            self.assertTrue((scored_root / ".gitkeep").exists())
            self.assertFalse((raw_root / "model_a").exists())
            self.assertFalse((parsed_root / "model_a").exists())
            self.assertFalse((scored_root / "suite_result_latest.json").exists())


if __name__ == "__main__":
    unittest.main()
