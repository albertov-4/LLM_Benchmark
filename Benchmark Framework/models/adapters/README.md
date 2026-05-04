# Model Adapters

Gli adapter servono a dare a tutti i modelli la stessa interfaccia logica.

Interfaccia minima attesa:
- input: `generate(messages: list[dict[str, str]])`
- output: dizionario normalizzato con `model_id`, `raw_text`, `usage`, `latency_s` e `notes`
- metadati opzionali: risposta provider, reasoning separato, token usage e parametri di generazione

Adapter previsti:
- `hf_local.py`: modelli locali Hugging Face
- `ollama.py`: modelli locali serviti da Ollama via HTTP API
- `llama_cpp_cli.py`: modelli GGUF locali eseguiti tramite binario llama.cpp
- `nvidia_api.py`: modelli remoti NVIDIA via client OpenAI-compatible
- `openai_api.py`: modelli serviti via API

Nota:
- il flow reale del framework supporta `hf_local.py`, `ollama.py`, `llama_cpp_cli.py` e `nvidia_api.py`
- `openai_api.py` resta uno scaffold per integrazioni future via API

Regola:
- il runner non deve conoscere i dettagli del provider
- il confronto fra modelli deve passare sempre dalla stessa interfaccia comune
