# Models

This folder contains model registries and adapter implementations.

Components:
- `model_registry_nvidia.yaml`: NVIDIA API registry
- `model_registry_hf.yaml`: Hugging Face local registry
- `model_registry_ollama.yaml`: Ollama registry
- `model_registry_llama_cpp.yaml`: llama.cpp / GGUF registry
- `adapters/`: implementations exposing `generate(messages)`

Useful registry fields:
- `enabled`: includes or excludes a model without deleting the entry
- `adapter`: selects the backend, such as `hf_local`, `nvidia_api`, `ollama` or `llama_cpp_cli`
- `weights_path`: local path or Hugging Face repo id for `from_pretrained(...)`
- `ollama_model`: local Ollama model name
- `base_url`: HTTP/API endpoint for adapters that use a server
- `api_model_name`: remote model name used by API adapters
- `api_key_env`: environment variable or local secret key name for the API key
- `api_mode`: API mode used by the adapter
- `stream`: enables streaming where supported
- `thinking_key`, `thinking_enabled` and `reasoning_budget`: optional reasoning-related provider settings
- `executable_path`: llama.cpp executable path
- `model_path`: local `.gguf` file for llama.cpp
- `device_map`, `torch_dtype`, `trust_remote_code`, `use_chat_template`, `add_generation_prompt`: Hugging Face loading and prompting options
- `timeout_seconds`: adapter call timeout

Operational notes:
- `--adapter` selects the matching registry automatically.
- `--model-registry-path` overrides the adapter shortcut and loads a specific YAML file.
- Alternative registries can keep entries disabled by default to avoid accidental heavy runs.
- For local/HPC Hugging Face runs, `weights_path` can be replaced with a prepared local model directory.
- For NVIDIA API runs, set the variable or local secret referenced by `api_key_env`.
- For llama.cpp, replace `model_path: REPLACE_WITH_LOCAL_GGUF_FILE` with a real local `.gguf` path.

Examples:
- NVIDIA: `python "Benchmark Framework/run_benchmark.py" --adapter nvidia_api --model-id <model_id>`
- Hugging Face: `python "Benchmark Framework/run_benchmark.py" --adapter hf_local --model-id <model_id>`
- Ollama: `python "Benchmark Framework/run_benchmark.py" --adapter ollama --model-id <model_id>`
- llama.cpp: `python "Benchmark Framework/run_benchmark.py" --adapter llama_cpp_cli --model-id <model_id>`
