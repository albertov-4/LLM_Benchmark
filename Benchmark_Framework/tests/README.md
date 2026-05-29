# Tests

The test suite checks the benchmark pipeline without requiring GPUs, external
APIs, or large model downloads.

## Structure

- `mocks/`: controllable model adapter and validator doubles.
- `fixtures/`: small benchmark fixtures for runner and validator tests.
- `test_parser.py`: parser behavior and output normalization.
- `test_real_validator.py`: optional VAL integration test on the toy fixture.
- `test_run_case.py`: single-case generation, parsing, validation, metrics, and
  artifact persistence.
- `test_run_suite.py`: task discovery, matrix orchestration, preflight behavior,
  adapter selection, and aggregation.
- `test_run_benchmark_cli.py`: CLI argument behavior.
- `test_hf_local.py`: Hugging Face adapter behavior without loading real models.
- `test_nvidia_api_adapter.py`: NVIDIA adapter behavior without remote API calls.
- `test_ollama_adapter.py`: Ollama adapter behavior without a running server.
- `test_llama_cpp_cli_adapter.py`: llama.cpp adapter behavior without executing
  llama.cpp.
- `test_prepare_models.py`: model preparation behavior without real downloads.
- `test_clear_outputs.py`: output cleanup behavior.
- `run_mock_suite.py`: manual smoke run with mock components.

## Commands

Run the unit tests:

```powershell
python -m unittest discover -s Benchmark_Framework/tests -p "test_*.py"
```

Run the mock suite manually:

```powershell
python Benchmark_Framework/tests/run_mock_suite.py
```

Run the optional real-validator check:

```powershell
python Benchmark_Framework/tests/test_real_validator.py
```

The real-validator test skips itself when a VAL executable cannot be resolved.

## Testing Goals

The tests focus on the `generate -> parse -> validate -> metrics` flow,
artifact writing, suite orchestration, and adapter normalization. Tests should
prefer fixtures and mocks unless a behavior specifically requires the real
benchmark task families or a real VAL executable.
