# Tests

This folder contains lightweight tests for the benchmark framework.

Structure:
- `mocks/`: controllable mock adapters and validators
- `fixtures/`: small tasks and configuration files used by tests
- `test_hf_local.py`: local Hugging Face adapter tests without loading real models
- `test_llama_cpp_cli_adapter.py`: llama.cpp CLI adapter tests without executing llama.cpp
- `test_nvidia_api_adapter.py`: NVIDIA API adapter tests without remote API calls
- `test_ollama_adapter.py`: Ollama adapter tests without requiring a running Ollama server
- `test_prepare_models.py`: model preparation tests without real downloads
- `test_parser.py`: focused tests for the shared parser
- `test_run_case.py`: single-run pipeline tests
- `test_run_suite.py`: suite orchestration tests
- `test_real_validator.py`: optional integration test for the real `VAL` validator on the toy fixture
- `test_clear_outputs.py`: output cleanup tests
- `run_mock_suite.py`: manual entry point for running a small mock suite

Goals:
- verify the `generate -> parse -> validate -> metrics` flow
- verify that `run_suite.py` can orchestrate multiple components together
- keep tests independent from GPUs, external APIs and large model downloads

Useful commands:
- `python -m unittest discover -s "Benchmark Framework/tests" -p "test_*.py"`
- `python "Benchmark Framework/tests/run_mock_suite.py"`
- `python "Benchmark Framework/tests/test_real_validator.py"`

Artifact tests:
- `test_run_case.py` checks that raw, parsed and scored artifacts are persisted in a temporary output directory
- tests should avoid relying on the real benchmark task families unless a task-specific behavior is being tested explicitly
