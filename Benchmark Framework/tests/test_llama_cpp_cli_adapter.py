"""Unit tests for the llama.cpp CLI adapter without running llama.cpp."""

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


class _FakeCompletedProcess:
    def __init__(self, stdout: str, stderr: str = "", returncode: int = 0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class LlamaCppCLIAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        cls.llama_cpp_module = _load_module(
            "benchmark_framework_test_llama_cpp_cli_module",
            framework_root / "models" / "adapters" / "llama_cpp_cli.py",
        )

    def test_generate_invokes_subprocess_and_normalizes_output(self) -> None:
        module = self.llama_cpp_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable_path = root / "llama-cli.exe"
            model_path = root / "model.gguf"
            executable_path.write_text("", encoding="utf-8")
            model_path.write_text("", encoding="utf-8")

            calls = []
            original_run = module.subprocess.run

            def fake_run(command, capture_output, text, timeout, check):
                calls.append(
                    {
                        "command": command,
                        "capture_output": capture_output,
                        "text": text,
                        "timeout": timeout,
                        "check": check,
                    }
                )
                prompt = command[command.index("-p") + 1]
                return _FakeCompletedProcess(stdout=f"{prompt}\n(move a b)")

            module.subprocess.run = fake_run
            try:
                adapter = module.LlamaCppCLIAdapter(
                    module.LlamaCppCLIConfig(
                        model_id="llama_cpp_test",
                        executable_path=str(executable_path),
                        model_path=str(model_path),
                        max_tokens=64,
                        context_size=1024,
                        threads=2,
                    )
                )
                result = adapter.generate([{"role": "user", "content": "Solve."}])
            finally:
                module.subprocess.run = original_run

        self.assertEqual(len(calls), 1)
        command = calls[0]["command"]
        self.assertIn("-m", command)
        self.assertIn("-p", command)
        self.assertIn("-n", command)
        self.assertIn("-c", command)
        self.assertIn("-t", command)
        self.assertEqual(result["model_id"], "llama_cpp_test")
        self.assertEqual(result["raw_text"], "(move a b)")
        self.assertEqual(result["reasoning_text"], "")
        self.assertIsNone(result["usage"]["total_tokens"])

    def test_generate_extracts_inline_thinking_tags(self) -> None:
        module = self.llama_cpp_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable_path = root / "llama-cli.exe"
            model_path = root / "model.gguf"
            executable_path.write_text("", encoding="utf-8")
            model_path.write_text("", encoding="utf-8")

            original_run = module.subprocess.run

            def fake_run(command, capture_output, text, timeout, check):
                prompt = command[command.index("-p") + 1]
                return _FakeCompletedProcess(
                    stdout=f"{prompt}\n<think>choose valid move</think>\n(move a b)"
                )

            module.subprocess.run = fake_run
            try:
                adapter = module.LlamaCppCLIAdapter(
                    module.LlamaCppCLIConfig(
                        model_id="llama_cpp_test",
                        executable_path=str(executable_path),
                        model_path=str(model_path),
                    )
                )
                result = adapter.generate([{"role": "user", "content": "Solve."}])
            finally:
                module.subprocess.run = original_run

        self.assertEqual(result["raw_text"], "(move a b)")
        self.assertEqual(result["reasoning_text"], "choose valid move")
        self.assertIn("inline reasoning extracted", result["notes"][0])

    def test_missing_model_path_fails_clearly(self) -> None:
        module = self.llama_cpp_module
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            executable_path = root / "llama-cli.exe"
            executable_path.write_text("", encoding="utf-8")

            adapter = module.LlamaCppCLIAdapter(
                module.LlamaCppCLIConfig(
                    model_id="llama_cpp_test",
                    executable_path=str(executable_path),
                    model_path=str(root / "missing.gguf"),
                )
            )

            with self.assertRaisesRegex(FileNotFoundError, "model file not found"):
                adapter.generate([{"role": "user", "content": "Solve."}])


if __name__ == "__main__":
    unittest.main()
