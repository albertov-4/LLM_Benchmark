"""Unit tests for the NVIDIA API adapter without remote calls."""

import importlib.util
import os
import sys
import tempfile
import types
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


class _FakeUsage:
    prompt_tokens = 4
    completion_tokens = 2
    total_tokens = 6


class _FakeMessage:
    content = "(move a b)"


class _FakeChoice:
    message = _FakeMessage()


class _FakeCompletion:
    choices = [_FakeChoice()]
    usage = _FakeUsage()


class _FakeDelta:
    def __init__(self, content=None, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeStreamChoice:
    def __init__(self, delta):
        self.delta = delta


class _FakeStreamChunk:
    def __init__(self, content=None, reasoning_content=None):
        self.choices = [_FakeStreamChoice(_FakeDelta(content, reasoning_content))]


class _FakeResponse:
    output_text = "(move response a b)"
    usage = _FakeUsage()


class _FakeCompletions:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("stream"):
            return [
                _FakeStreamChunk(reasoning_content="think "),
                _FakeStreamChunk(content="(move "),
                _FakeStreamChunk(content="a b)"),
            ]
        return _FakeCompletion()


class _FailingStream:
    def __iter__(self):
        yield _FakeStreamChunk(content="(move ")
        yield _FakeStreamChunk(reasoning_content="partial reasoning ")
        raise RuntimeError("incomplete chunked read")


class _TimeoutStream:
    def __iter__(self):
        yield _FakeStreamChunk(content="(move ")
        yield _FakeStreamChunk(content="a b)")


class _FailingStreamCompletions:
    def __init__(self, stream):
        self.stream = stream
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self.stream


class _FakeChat:
    def __init__(self):
        self.completions = _FakeCompletions()


class _CustomChat:
    def __init__(self, stream):
        self.completions = _FailingStreamCompletions(stream)


class _FakeResponses:
    def __init__(self):
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse()


class _FakeOpenAI:
    last_instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.chat = _FakeChat()
        self.responses = _FakeResponses()
        _FakeOpenAI.last_instance = self


class _CustomOpenAI:
    last_instance = None

    def __init__(self, stream, **kwargs):
        self.kwargs = kwargs
        self.chat = _CustomChat(stream)
        self.responses = _FakeResponses()
        _CustomOpenAI.last_instance = self


class NvidiaAPIAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        cls.nvidia_module = _load_module(
            "benchmark_framework_test_nvidia_api_module",
            framework_root / "models" / "adapters" / "nvidia_api.py",
        )

    def test_generate_uses_openai_compatible_client(self) -> None:
        module = self.nvidia_module
        old_api_key = os.environ.get("NVIDIA_API_KEY")
        os.environ["NVIDIA_API_KEY"] = "test-key"

        fake_openai_module = types.SimpleNamespace(OpenAI=_FakeOpenAI)

        class TestAdapter(module.NvidiaAPIAdapter):
            def _load_openai_client_class(self):
                return fake_openai_module.OpenAI

        try:
            adapter = TestAdapter(
                module.NvidiaAPIConfig(
                    model_id="nvidia_test",
                    api_model_name="deepseek-ai/deepseek-v4-pro",
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=128,
                    extra_body={"chat_template_kwargs": {"thinking": False}},
                )
            )
            result = adapter.generate([{"role": "user", "content": "Solve."}])
        finally:
            if old_api_key is None:
                os.environ.pop("NVIDIA_API_KEY", None)
            else:
                os.environ["NVIDIA_API_KEY"] = old_api_key

        fake_client = _FakeOpenAI.last_instance
        self.assertEqual(fake_client.kwargs["base_url"], "https://integrate.api.nvidia.com/v1")
        self.assertEqual(fake_client.kwargs["api_key"], "test-key")
        create_kwargs = fake_client.chat.completions.last_kwargs
        self.assertEqual(create_kwargs["model"], "deepseek-ai/deepseek-v4-pro")
        self.assertEqual(create_kwargs["max_tokens"], 128)
        self.assertFalse(create_kwargs["stream"])
        self.assertEqual(result["model_id"], "nvidia_test")
        self.assertEqual(result["raw_text"], "(move a b)")
        self.assertEqual(result["usage"]["total_tokens"], 6)

    def test_streaming_chat_collects_content_and_reasoning_separately(self) -> None:
        module = self.nvidia_module
        old_api_key = os.environ.get("NVIDIA_API_KEY")
        os.environ["NVIDIA_API_KEY"] = "test-key"

        class TestAdapter(module.NvidiaAPIAdapter):
            def _load_openai_client_class(self):
                return _FakeOpenAI

        try:
            adapter = TestAdapter(
                module.NvidiaAPIConfig(
                    model_id="nvidia_stream_test",
                    api_model_name="moonshotai/kimi-k2-thinking",
                    stream=True,
                )
            )
            result = adapter.generate([{"role": "user", "content": "Solve."}])
        finally:
            if old_api_key is None:
                os.environ.pop("NVIDIA_API_KEY", None)
            else:
                os.environ["NVIDIA_API_KEY"] = old_api_key

        self.assertEqual(result["raw_text"], "(move a b)")
        self.assertEqual(result["reasoning_text"], "think")
        self.assertTrue(result["stream_complete"])
        self.assertIsNone(result["stream_error"])
        self.assertFalse(result["partial_output"])
        self.assertFalse(result["timed_out_by_job_limit"])
        self.assertIn("reasoning_content captured", result["notes"][0])

    def test_streaming_chat_returns_partial_output_when_stream_fails(self) -> None:
        module = self.nvidia_module
        old_api_key = os.environ.get("NVIDIA_API_KEY")
        os.environ["NVIDIA_API_KEY"] = "test-key"

        class TestAdapter(module.NvidiaAPIAdapter):
            def _load_openai_client_class(self):
                return lambda **kwargs: _CustomOpenAI(_FailingStream(), **kwargs)

        try:
            adapter = TestAdapter(
                module.NvidiaAPIConfig(
                    model_id="nvidia_partial_stream_test",
                    api_model_name="vendor/model",
                    stream=True,
                )
            )
            result = adapter.generate([{"role": "user", "content": "Solve."}])
        finally:
            if old_api_key is None:
                os.environ.pop("NVIDIA_API_KEY", None)
            else:
                os.environ["NVIDIA_API_KEY"] = old_api_key

        self.assertEqual(result["raw_text"], "(move")
        self.assertEqual(result["reasoning_text"], "partial reasoning")
        self.assertFalse(result["stream_complete"])
        self.assertIn("RuntimeError: incomplete chunked read", result["stream_error"])
        self.assertTrue(result["partial_output"])
        self.assertFalse(result["timed_out_by_job_limit"])
        self.assertIn("stream interrupted", " ".join(result["notes"]))

    def test_streaming_chat_stops_on_job_timeout_with_partial_output(self) -> None:
        module = self.nvidia_module
        old_api_key = os.environ.get("NVIDIA_API_KEY")
        old_perf_counter = module.perf_counter
        os.environ["NVIDIA_API_KEY"] = "test-key"

        timings = iter([0.0, 0.0, 2.0, 2.1])
        module.perf_counter = lambda: next(timings)

        class TestAdapter(module.NvidiaAPIAdapter):
            def _load_openai_client_class(self):
                return lambda **kwargs: _CustomOpenAI(_TimeoutStream(), **kwargs)

        try:
            adapter = TestAdapter(
                module.NvidiaAPIConfig(
                    model_id="nvidia_job_timeout_test",
                    api_model_name="vendor/model",
                    stream=True,
                    job_timeout_seconds=1,
                )
            )
            result = adapter.generate([{"role": "user", "content": "Solve."}])
        finally:
            module.perf_counter = old_perf_counter
            if old_api_key is None:
                os.environ.pop("NVIDIA_API_KEY", None)
            else:
                os.environ["NVIDIA_API_KEY"] = old_api_key

        self.assertEqual(result["raw_text"], "(move")
        self.assertFalse(result["stream_complete"])
        self.assertEqual(result["stream_error"], "JobTimeout: generation exceeded 1 seconds")
        self.assertTrue(result["partial_output"])
        self.assertTrue(result["timed_out_by_job_limit"])
        self.assertIn("generation stopped by job timeout.", result["notes"])

    def test_responses_api_mode_uses_responses_client(self) -> None:
        module = self.nvidia_module
        old_api_key = os.environ.get("NVIDIA_API_KEY")
        os.environ["NVIDIA_API_KEY"] = "test-key"

        class TestAdapter(module.NvidiaAPIAdapter):
            def _load_openai_client_class(self):
                return _FakeOpenAI

        try:
            adapter = TestAdapter(
                module.NvidiaAPIConfig(
                    model_id="nvidia_responses_test",
                    api_model_name="openai/gpt-oss-120b",
                    api_mode="responses",
                    max_tokens=4096,
                )
            )
            result = adapter.generate([{"role": "user", "content": "Solve."}])
        finally:
            if old_api_key is None:
                os.environ.pop("NVIDIA_API_KEY", None)
            else:
                os.environ["NVIDIA_API_KEY"] = old_api_key

        fake_client = _FakeOpenAI.last_instance
        self.assertEqual(fake_client.responses.last_kwargs["model"], "openai/gpt-oss-120b")
        self.assertEqual(fake_client.responses.last_kwargs["max_output_tokens"], 4096)
        self.assertEqual(result["raw_text"], "(move response a b)")

    def test_missing_api_key_fails_clearly(self) -> None:
        module = self.nvidia_module
        old_api_key = os.environ.pop("NVIDIA_API_KEY", None)
        try:
            adapter = module.NvidiaAPIAdapter(
                module.NvidiaAPIConfig(
                    model_id="nvidia_test",
                    api_model_name="deepseek-ai/deepseek-v4-pro",
                    secrets_path="missing-secrets.local.json",
                )
            )
            with self.assertRaisesRegex(RuntimeError, "NVIDIA API key not found"):
                adapter._get_api_key()
        finally:
            if old_api_key is not None:
                os.environ["NVIDIA_API_KEY"] = old_api_key

    def test_local_secrets_file_is_used_when_env_is_missing(self) -> None:
        module = self.nvidia_module
        old_custom_key = os.environ.pop("NVIDIA_LOCAL_SECRET_KEY", None)

        with tempfile.TemporaryDirectory() as tmp_dir:
            secrets_path = Path(tmp_dir) / "secrets.local.json"
            secrets_path.write_text(
                '{"NVIDIA_LOCAL_SECRET_KEY": "local-secret-value"}',
                encoding="utf-8",
            )
            try:
                adapter = module.NvidiaAPIAdapter(
                    module.NvidiaAPIConfig(
                        model_id="nvidia_local_secret_test",
                        api_model_name="vendor/model",
                        api_key_env="NVIDIA_LOCAL_SECRET_KEY",
                        secrets_path=str(secrets_path),
                    )
                )
                self.assertEqual(adapter._get_api_key(), "local-secret-value")
            finally:
                if old_custom_key is not None:
                    os.environ["NVIDIA_LOCAL_SECRET_KEY"] = old_custom_key

    def test_custom_api_key_env_is_used(self) -> None:
        module = self.nvidia_module
        old_custom_key = os.environ.get("NVIDIA_CUSTOM_MODEL_KEY")
        os.environ["NVIDIA_CUSTOM_MODEL_KEY"] = "custom-test-key"

        try:
            adapter = module.NvidiaAPIAdapter(
                module.NvidiaAPIConfig(
                    model_id="nvidia_custom_key_test",
                    api_model_name="vendor/model",
                    api_key_env="NVIDIA_CUSTOM_MODEL_KEY",
                )
            )
            self.assertEqual(adapter._get_api_key(), "custom-test-key")
        finally:
            if old_custom_key is None:
                os.environ.pop("NVIDIA_CUSTOM_MODEL_KEY", None)
            else:
                os.environ["NVIDIA_CUSTOM_MODEL_KEY"] = old_custom_key


if __name__ == "__main__":
    unittest.main()
