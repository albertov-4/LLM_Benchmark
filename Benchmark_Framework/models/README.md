# Models

This folder contains model registries and backend adapters. Registries describe
what to run; adapters implement how to call each backend.

## Registry Files

- `model_registry_nvidia.yaml`: NVIDIA API models through an OpenAI-compatible
  client.
- `model_registry_hf.yaml`: local Hugging Face Transformers models.
- `model_registry_ollama.yaml`: models served by a local Ollama server.
- `model_registry_llama_cpp.yaml`: GGUF models executed through llama.cpp CLI.
- `adapters/`: adapter implementations used by the runner.

Each registry contains a `models` list:

```yaml
models:
  - model_id: example_model
    family: example_family
    adapter: nvidia_api
    provider: nvidia_api
    enabled: true
```

`--adapter` selects a matching registry automatically. `--model-registry-path`
loads an explicit registry and takes precedence over `--adapter`.

## Common Fields

- `model_id`: stable id used in filters and output paths.
- `family`: human-readable model family.
- `adapter`: backend implementation. Supported values are `nvidia_api`,
  `hf_local`, `ollama`, and `llama_cpp_cli`.
- `provider`: descriptive provider/runtime label.
- `enabled`: included in full runs when no `--model-id` is supplied.
- `reasoning_notes`: setup notes, expected limitations, or model behavior.
  Provider-side reasoning is diagnostic only; benchmark scoring uses the final
  plan extracted from adapter `raw_text`.
- `temperature`, `top_p`, `max_tokens`, `timeout_seconds`: generation/runtime
  settings used where the adapter supports them.

Some registries include `registry_rules`. These are documentation metadata for
expected fields; execution is controlled by entries under `models`.

## Hugging Face Local

Important fields:

- `weights_path`: Hugging Face repo id or local directory.
- `device_map`: `auto`, `none`, or another Transformers-supported placement.
- `torch_dtype`: `auto`, `float16`, `bfloat16`, or similar.
- `trust_remote_code`: allow custom model code.
- `use_chat_template`: use tokenizer chat templates when available.
- `add_generation_prompt`: add the assistant-generation marker.
- `thinking_key` and `thinking_enabled`: optional chat-template kwargs for
  models that expose reasoning controls.

The adapter resolves model sources in this order: explicit local path,
`models_cache/<namespace>__<repo>` for prepared Hub models, then the original
repo id or configured value. Runtime loading and `scripts/prepare_models.py`
use the same repo-id detection and cache-directory naming rules.

Prepare local models with:

```powershell
python Benchmark_Framework/scripts/prepare_models.py --model-registry-path models/model_registry_hf.yaml --models-dir models_cache
```

## NVIDIA API

Important fields:

- `api_model_name`: remote model name sent to the provider.
- `api_mode`: endpoint style, such as `chat_completions` or `responses`.
- `base_url`: OpenAI-compatible base URL.
- `api_key_env`: environment variable or local secrets key.
- `stream`: enables streamed responses.
- `timeout_seconds`: API/client timeout.
- `job_timeout_seconds`: optional total timeout for one streamed attempt.
- `debug_stream`: print streaming diagnostics.
- `thinking_key`, `thinking_enabled`, `reasoning_budget`: optional
  provider-specific reasoning controls.

For streaming runs, interrupted streams with partial text are preserved and
marked in the generation payload with stream status fields.

## Ollama

Important fields:

- `ollama_model`: local Ollama model name.
- `base_url`: usually `http://localhost:11434`.
- `temperature`, `top_p`, `max_tokens`, `timeout_seconds`.

Install Ollama separately, start the service, pull the model, and enable the
matching registry entry.

## llama.cpp CLI

Important fields:

- `executable_path`: command or path for the llama.cpp executable.
- `model_path`: local `.gguf` file.
- `weights_path`: optional alternative source for file-based setups.
- `gguf_source`: informational source for the GGUF file.
- `context_size`, `threads`, `timeout_seconds`.

The runner does not download GGUF files. Replace placeholder paths with real
local files before enabling llama.cpp entries.

## Sampling Practice

For planning benchmarks, keep generation settings stable when comparing models.
Low temperatures are usually preferable because output must be valid PDDL, not
creative prose. Change one sampling field at a time when testing sensitivity and
record the chosen values in the registry.

## Examples

```powershell
python Benchmark_Framework/run_benchmark.py --adapter nvidia_api --model-id <model_id> --use-real-validator
python Benchmark_Framework/run_benchmark.py --adapter hf_local --model-id <model_id> --use-real-validator
python Benchmark_Framework/run_benchmark.py --adapter ollama --model-id <model_id> --use-real-validator
python Benchmark_Framework/run_benchmark.py --adapter llama_cpp_cli --model-id <model_id> --use-real-validator
```
