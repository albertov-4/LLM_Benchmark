"""Unit tests for the local Hugging Face adapter."""

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


class _FakeInferenceMode:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorchCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class _FakeTorch:
    cuda = _FakeTorchCuda()
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    @staticmethod
    def inference_mode():
        return _FakeInferenceMode()


class _FakeTensor:
    def __init__(self, data):
        self.data = data
        self.device = "cpu"

    @property
    def shape(self):
        if self.data and isinstance(self.data[0], list):
            return (len(self.data), len(self.data[0]))
        return (len(self.data),)

    def to(self, device):
        self.device = device
        return self

    def __getitem__(self, item):
        return self.data[item]


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    eos_token = "</s>"

    def __init__(self):
        self.last_prompt = ""
        self.last_chat_template_kwargs = {}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        self.last_chat_template_kwargs = dict(kwargs)
        self.last_prompt = "\n".join(f"{message['role']}::{message['content']}" for message in messages)
        if add_generation_prompt:
            self.last_prompt += "\nassistant::"
        return self.last_prompt

    def __call__(self, prompt_text, return_tensors="pt"):
        self.last_prompt = prompt_text
        return {
            "input_ids": _FakeTensor([[10, 11, 12]]),
            "attention_mask": _FakeTensor([[1, 1, 1]]),
        }

    def decode(self, tokens, skip_special_tokens=True):
        self.last_decoded = list(tokens)
        return "(move car1 j1 j2)\n(move car1 j2 j3)"


class _RejectingKwargsTokenizer(_FakeTokenizer):
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        if kwargs:
            raise TypeError("unsupported chat template kwargs")
        return super().apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
        )


class _FakeModel:
    def __init__(self):
        self.device = "cpu"
        self.last_generate_kwargs = {}

    def eval(self):
        return self

    def generate(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return [[10, 11, 12, 20, 21]]


class HFLocalAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        cls.hf_module = _load_module(
            "benchmark_framework_test_hf_local_module",
            framework_root / "models" / "adapters" / "hf_local.py",
        )

    def test_generate_returns_normalized_payload(self) -> None:
        module = self.hf_module

        class TestAdapter(module.HFLocalAdapter):
            def _load_backend(self):
                return _FakeTorch, object, object

            def load_model(self):
                self.model = _FakeModel()
                self.tokenizer = _FakeTokenizer()

        adapter = TestAdapter(
            module.HFLocalConfig(
                model_id="test-model",
                weights_path="test-source",
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

        self.assertEqual(result["model_id"], "test-model")
        self.assertEqual(result["raw_text"], "(move car1 j1 j2)\n(move car1 j2 j3)")
        self.assertEqual(result["reasoning_text"], "")
        self.assertEqual(result["usage"]["prompt_tokens"], 3)
        self.assertEqual(result["usage"]["completion_tokens"], 2)
        self.assertEqual(result["usage"]["total_tokens"], 5)
        self.assertEqual(result["notes"], [])
        self.assertIn("max_new_tokens", adapter.model.last_generate_kwargs)
        self.assertFalse(adapter.model.last_generate_kwargs["do_sample"])

    def test_generate_uses_top_p_and_thinking_chat_template_kwargs(self) -> None:
        module = self.hf_module

        class TestAdapter(module.HFLocalAdapter):
            def _load_backend(self):
                return _FakeTorch, object, object

            def load_model(self):
                self.model = _FakeModel()
                self.tokenizer = _FakeTokenizer()

        adapter = TestAdapter(
            module.HFLocalConfig(
                model_id="test-model",
                weights_path="test-source",
                temperature=0.1,
                top_k=10,
                top_p=0.95,
                max_tokens=128,
                thinking_key="enable_thinking",
                thinking_enabled=False,
            )
        )

        adapter.generate([{"role": "user", "content": "Solve the problem."}])

        self.assertTrue(adapter.model.last_generate_kwargs["do_sample"])
        self.assertEqual(adapter.model.last_generate_kwargs["temperature"], 0.1)
        self.assertEqual(adapter.model.last_generate_kwargs["top_p"], 0.95)
        self.assertEqual(
            adapter.tokenizer.last_chat_template_kwargs,
            {"enable_thinking": False},
        )

    def test_chat_template_retries_without_thinking_kwargs_when_unsupported(self) -> None:
        module = self.hf_module

        class TestAdapter(module.HFLocalAdapter):
            def _load_backend(self):
                return _FakeTorch, object, object

            def load_model(self):
                self.model = _FakeModel()
                self.tokenizer = _RejectingKwargsTokenizer()

        adapter = TestAdapter(
            module.HFLocalConfig(
                model_id="test-model",
                weights_path="test-source",
                thinking_key="enable_thinking",
                thinking_enabled=False,
            )
        )

        result = adapter.generate([{"role": "user", "content": "Solve the problem."}])

        self.assertEqual(result["raw_text"], "(move car1 j1 j2)\n(move car1 j2 j3)")
        self.assertEqual(result["reasoning_text"], "")
        self.assertEqual(adapter.tokenizer.last_chat_template_kwargs, {})

    def test_generate_extracts_inline_thinking_tags(self) -> None:
        module = self.hf_module

        class ThinkingTokenizer(_FakeTokenizer):
            def decode(self, tokens, skip_special_tokens=True):
                return "<think>verify adjacency</think>\n(move car1 j1 j2)"

        class TestAdapter(module.HFLocalAdapter):
            def _load_backend(self):
                return _FakeTorch, object, object

            def load_model(self):
                self.model = _FakeModel()
                self.tokenizer = ThinkingTokenizer()

        adapter = TestAdapter(module.HFLocalConfig(model_id="test-model"))

        result = adapter.generate([{"role": "user", "content": "Solve the problem."}])

        self.assertEqual(result["raw_text"], "(move car1 j1 j2)")
        self.assertEqual(result["reasoning_text"], "verify adjacency")
        self.assertIn("inline reasoning extracted", result["notes"][0])

    def test_resolve_model_source_prefers_weights_path(self) -> None:
        module = self.hf_module
        adapter = module.HFLocalAdapter(
            module.HFLocalConfig(
                model_id="fallback-model-id",
                weights_path="C:/models/local-weights",
            )
        )

        self.assertEqual(adapter._resolve_model_source(), "C:/models/local-weights")

    def test_resolve_model_source_prefers_prepared_models_cache_for_hf_repo(self) -> None:
        module = self.hf_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "models_cache"
            prepared_model = cache_dir / "nvidia__Llama-3.1-Nemotron-Nano-4B-v1.1"
            prepared_model.mkdir(parents=True)
            (prepared_model / "config.json").write_text("{}", encoding="utf-8")

            class TestAdapter(module.HFLocalAdapter):
                def _models_cache_dir(self):
                    return cache_dir

            adapter = TestAdapter(
                module.HFLocalConfig(
                    model_id="hf_nemotron",
                    weights_path="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
                )
            )

            self.assertEqual(adapter._resolve_model_source(), str(prepared_model.resolve()))

    def test_resolve_model_source_falls_back_to_hf_repo_when_cache_is_missing(self) -> None:
        module = self.hf_module

        with tempfile.TemporaryDirectory() as tmp_dir:
            cache_dir = Path(tmp_dir) / "models_cache"
            cache_dir.mkdir()

            class TestAdapter(module.HFLocalAdapter):
                def _models_cache_dir(self):
                    return cache_dir

            adapter = TestAdapter(
                module.HFLocalConfig(
                    model_id="hf_nemotron",
                    weights_path="nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
                )
            )

            self.assertEqual(
                adapter._resolve_model_source(),
                "nvidia/Llama-3.1-Nemotron-Nano-4B-v1.1",
            )


if __name__ == "__main__":
    unittest.main()
