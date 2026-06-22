# Model Adapters

Adapters give every backend the same benchmark-facing interface. The runner
does not depend on provider-specific response objects.

## Contract

Expected input:

```python
generate(messages: list[dict[str, str]])
```

Expected normalized output:

```python
{
    "model_id": "...",
    "raw_text": "...",
    "usage": {...},
    "latency_s": ...,
    "notes": [...],
}
```

Adapters may include additional metadata such as provider payloads,
`reasoning_text`, token counts, sampling settings, streaming status, or error
details. The runner stores these fields in raw artifacts; parser artifacts keep
only a `source_ref` to reasoning text.

## Implementations

- `hf_local.py`: local Hugging Face Transformers models.
- `nvidia_api.py`: NVIDIA API models through an OpenAI-compatible client.
- `ollama.py`: local models served by Ollama.
- `llama_cpp_cli.py`: local GGUF models through llama.cpp CLI.

## Shared Behavior

Adapters normalize model output before the runner parses it. Hugging Face,
Ollama, and llama.cpp adapters share one cleanup helper that strips common
reasoning tags such as `think`, `thinking`, `reasoning`, and `analysis` when
they appear as explicit tagged blocks. Reasoning text can still be preserved
separately when the backend returns it in a structured way.

The Hugging Face adapter also shares repo-id detection and
`models_cache/<namespace>__<repo>` path resolution with
`scripts/prepare_models.py`.

## NVIDIA Streaming

The NVIDIA adapter supports both non-streaming and streaming responses.

For streaming runs:

- streamed content and streamed reasoning are accumulated separately;
- interrupted streams can return partial text instead of failing the whole job;
- partial results include `partial_output`, `stream_complete`, and
  `stream_error`;
- `job_timeout_seconds` can stop a long attempt and return text accumulated so
  far;
- `debug_stream: true` prints request, progress, timeout, error, and completion
  diagnostics.

This behavior lets long API calls produce auditable artifacts even when a stream
does not finish cleanly.

NVIDIA model parallelism is handled by the suite runner, not by this adapter.
When `run_benchmark.py` is launched with `--parallel-nvidia-models`, the runner
starts separate per-model lanes for `nvidia_api` entries. Each lane still calls
this adapter synchronously for one benchmark case at a time. In that mode, the
runner can route each lane's stdout to `outputs/logs/<run_id>/<model_id>.log`.

## Adapter Boundaries

Adapter-specific setup belongs in the registry entry and adapter configuration.
Known adapter construction fails fast when configuration or imports are invalid;
only unknown or empty adapter names use the unavailable-adapter placeholder.
Parsing, validation, repair, metrics, and artifact layout belong in the shared
runner and evaluator layers.
