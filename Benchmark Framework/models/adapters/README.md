# Model Adapters

Adapters give every model backend the same benchmark interface.

Expected interface:
- input: `generate(messages: list[dict[str, str]])`
- output: a normalized dictionary with `model_id`, `raw_text`, `usage`, `latency_s` and `notes`
- optional metadata can include provider response details, separate reasoning text, token usage and generation parameters

Supported adapters:
- `hf_local.py`: local Hugging Face models
- `ollama.py`: local models served through the Ollama HTTP API
- `llama_cpp_cli.py`: local GGUF models executed through the llama.cpp command line interface
- `nvidia_api.py`: remote NVIDIA models through an OpenAI-compatible client

Rules:
- the runner should not depend on provider-specific response formats
- model comparisons should always go through the common adapter interface
- adapter-specific setup belongs in the model registry and adapter configuration, not in the runner
