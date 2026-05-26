"""Unit tests for the Ollama adapter without requiring a running server."""

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


class OllamaAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        cls.ollama_module = _load_module(
            "benchmark_framework_test_ollama_module",
            framework_root / "models" / "adapters" / "ollama.py",
        )

    def test_generate_returns_normalized_payload(self) -> None:
        module = self.ollama_module

        class TestAdapter(module.OllamaAdapter):
            def _post_chat(self, payload):
                self.last_payload = payload
                return {
                    "model": "qwen2.5:0.5b",
                    "message": {"role": "assistant", "content": "(move car1 j1 j2)"},
                    "done": True,
                    "prompt_eval_count": 12,
                    "eval_count": 5,
                    "total_duration": 100,
                    "load_duration": 10,
                    "prompt_eval_duration": 30,
                    "eval_duration": 60,
                }

        adapter = TestAdapter(
            module.OllamaConfig(
                model_id="ollama_test",
                ollama_model="qwen2.5:0.5b",
                base_url="http://localhost:11434",
                temperature=0.0,
                top_k=10,
                max_tokens=128,
            )
        )

        result = adapter.generate(
            [
                {"role": "system", "content": "You are a planner."},
                {"role": "user", "content": "Solve the problem."},
            ]
        )

        self.assertEqual(adapter.last_payload["model"], "qwen2.5:0.5b")
        self.assertEqual(adapter.last_payload["options"]["num_predict"], 128)
        self.assertFalse(adapter.last_payload["stream"])
        self.assertEqual(result["model_id"], "ollama_test")
        self.assertEqual(result["raw_text"], "(move car1 j1 j2)")
        self.assertEqual(result["reasoning_text"], "")
        self.assertEqual(result["usage"]["prompt_tokens"], 12)
        self.assertEqual(result["usage"]["completion_tokens"], 5)
        self.assertEqual(result["usage"]["total_tokens"], 17)
        self.assertEqual(result["notes"], [])

    def test_generate_captures_provider_thinking_separately(self) -> None:
        module = self.ollama_module

        class TestAdapter(module.OllamaAdapter):
            def _post_chat(self, payload):
                return {
                    "model": "thinking-model",
                    "message": {
                        "role": "assistant",
                        "thinking": "check numeric preconditions",
                        "content": "(move car1 j1 j2)",
                    },
                    "done": True,
                }

        adapter = TestAdapter(module.OllamaConfig(model_id="ollama_thinking_test"))

        result = adapter.generate([{"role": "user", "content": "Solve the problem."}])

        self.assertEqual(result["raw_text"], "(move car1 j1 j2)")
        self.assertEqual(result["reasoning_text"], "check numeric preconditions")
        self.assertIn("provider reasoning captured", result["notes"][0])

    def test_model_id_is_fallback_model_name(self) -> None:
        module = self.ollama_module
        adapter = module.OllamaAdapter(
            module.OllamaConfig(
                model_id="fallback-model",
                ollama_model="",
            )
        )

        self.assertEqual(adapter._resolve_ollama_model(), "fallback-model")


if __name__ == "__main__":
    unittest.main()
