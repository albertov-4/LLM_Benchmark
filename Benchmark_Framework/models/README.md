# Models

This folder contains model registries and adapter implementations.

Components:
- `model_registry_nvidia.yaml`: NVIDIA API registry
- `model_registry_hf.yaml`: Hugging Face local registry
- `model_registry_ollama.yaml`: Ollama registry
- `model_registry_llama_cpp.yaml`: llama.cpp / GGUF registry
- `adapters/`: implementations exposing `generate(messages)`

## Registry Files

Each registry file contains a `models` list. Every item in that list describes one model configuration.

Example:

```yaml
models:
  - model_id: example_model
    family: example_family
    adapter: nvidia_api
    provider: nvidia_api
    enabled: true
```

The launcher can select a registry automatically with `--adapter`, or load an explicit registry with `--model-registry-path`.

Some registry files also include `registry_rules`. This section documents expected fields for that registry:
- `required_fields`: fields that every model entry in that registry should define.
- `optional_fields`: fields accepted by that registry but not required for every model.

`registry_rules` is documentation metadata. Model execution is controlled by the entries under `models`.

## Complete Field Reference

Common fields:
- `model_id`: unique identifier used by the benchmark outputs, filters and result files.
- `family`: model family label used for organization and human readability.
- `adapter`: backend implementation to use. Supported values include `hf_local`, `nvidia_api`, `ollama` and `llama_cpp_cli`.
- `provider`: provider or runtime label. This is mainly descriptive and helps distinguish local, API and server-backed models.
- `enabled`: includes or excludes the model when no `--model-id` filter is used.
- `reasoning_notes`: free-text notes about access, model behavior, reasoning mode, expected limitations or setup requirements.

Hugging Face local fields:
- `weights_path`: Hugging Face repo id or local directory passed to `from_pretrained(...)`.
- `device_map`: device placement for Transformers, for example `auto` or `none`.
- `torch_dtype`: dtype used when loading the model, for example `auto`, `float16` or `bfloat16`.
- `trust_remote_code`: allows model repositories with custom Python model code.
- `use_chat_template`: uses the tokenizer chat template when available.
- `add_generation_prompt`: adds the assistant-generation marker when applying the chat template.

NVIDIA API fields:
- `api_model_name`: remote model name sent to the NVIDIA API.
- `api_mode`: API endpoint style used by the adapter. Current values are `chat_completions` and `responses`.
- `base_url`: OpenAI-compatible API base URL.
- `api_key_env`: environment variable name or local secrets key used to load the API key.
- `stream`: enables streamed responses when supported by the selected API mode.
- `temperature`: sampling temperature.
- `top_p`: nucleus sampling parameter.
- `max_tokens`: maximum generated tokens requested from the provider.
- `timeout_seconds`: API/client timeout.
- `job_timeout_seconds`: optional total generation-attempt timeout for streaming runs.
- `debug_stream`: enables NVIDIA streaming progress logs for this model.
- `debug_stream_interval_seconds`: minimum interval between NVIDIA streaming progress logs.
- `thinking_key`: provider-specific key inserted under `chat_template_kwargs` to enable or disable thinking/reasoning modes.
- `thinking_enabled`: boolean value used with `thinking_key`.
- `reasoning_budget`: provider-specific reasoning budget passed in the request body.

Ollama fields:
- `ollama_model`: local Ollama model name to call.
- `base_url`: Ollama server URL, usually `http://localhost:11434`.
- `temperature`: sampling temperature.
- `top_p`: nucleus sampling parameter.
- `max_tokens`: maximum generated tokens requested from Ollama.
- `timeout_seconds`: HTTP request timeout.

llama.cpp CLI fields:
- `executable_path`: path or command name for the llama.cpp executable.
- `model_path`: local `.gguf` file used by llama.cpp.
- `weights_path`: optional alternative source for `model_path` in adapters that support it.
- `gguf_source`: informational source or repository for the GGUF file. This is not downloaded automatically by the runner.
- `temperature`: sampling temperature.
- `top_p`: nucleus sampling parameter.
- `max_tokens`: maximum generated tokens requested from llama.cpp.
- `context_size`: context window passed to llama.cpp when configured.
- `threads`: number of CPU threads passed to llama.cpp when configured.
- `timeout_seconds`: process timeout.

Additional metadata fields:
- `context_window`: optional descriptive context-window metadata. It is useful for documentation, but the current runner does not use it directly.
- `executable_path`: command or executable location for CLI-based adapters.
- `model_path`: local model file path for file-based adapters.

## Useful Fields

Most runs only require these fields:
- `model_id`
- `adapter`
- `enabled`
- `weights_path` for Hugging Face local models
- `api_model_name` and `api_key_env` for NVIDIA API models
- `ollama_model` for Ollama models
- `model_path` and `executable_path` for llama.cpp models
- `max_tokens`
- `temperature`
- `top_p`
- `timeout_seconds`
- `job_timeout_seconds` for NVIDIA streaming runs
- `debug_stream` when diagnosing NVIDIA streaming stalls
- `stream` for APIs that support streaming

## Generation Sampling Fields

`temperature` and `top_p` control how deterministic or varied the model output is.

`temperature`:
- controls how strongly the model favors the most likely next token
- lower values make output more deterministic
- higher values make output more varied and less predictable
- for planning benchmarks, low values are usually preferable because the goal is valid action generation, not creative variation

Practical `temperature` values:
- `0.0`: most deterministic setting when supported by the backend
- `0.1` to `0.3`: still conservative, but may avoid some repetitive failures
- `0.7` to `1.0`: more exploratory, usually less stable for strict PDDL output

`top_p`:
- controls nucleus sampling
- the model samples only from the smallest token set whose cumulative probability reaches `top_p`
- lower values restrict the candidate token set
- higher values allow more alternatives

Practical `top_p` values:
- `0.7`: restrictive and more focused
- `0.9` to `0.95`: common balanced setting
- `1.0`: least restrictive setting

How to modify them:

```yaml
models:
  - model_id: example_model
    adapter: nvidia_api
    temperature: 0.1
    top_p: 0.9
```

Recommended benchmark practice:
- keep `temperature` and `top_p` fixed when comparing models
- change one field at a time when testing sensitivity
- record the chosen values in the registry so results remain reproducible
- start with `temperature: 0.0` or `0.1` for strict plan-only protocols

## Operational Notes

- `--adapter` selects the matching registry automatically.
- `--model-registry-path` overrides the adapter shortcut and loads a specific YAML file.
- Alternative registries can keep entries disabled by default to avoid accidental heavy runs.
- For local/HPC Hugging Face runs, `weights_path` can be replaced with a prepared local model directory.
- For NVIDIA API runs, set the variable or local secret referenced by `api_key_env`.
- For NVIDIA streaming runs, partial output is preserved when the stream is interrupted after text has already been received.
- For llama.cpp, replace `model_path: REPLACE_WITH_LOCAL_GGUF_FILE` with a real local `.gguf` path.

## Examples

- NVIDIA: `python "Benchmark Framework/run_benchmark.py" --adapter nvidia_api --model-id <model_id>`
- Hugging Face: `python "Benchmark Framework/run_benchmark.py" --adapter hf_local --model-id <model_id>`
- Ollama: `python "Benchmark Framework/run_benchmark.py" --adapter ollama --model-id <model_id>`
- llama.cpp: `python "Benchmark Framework/run_benchmark.py" --adapter llama_cpp_cli --model-id <model_id>`
