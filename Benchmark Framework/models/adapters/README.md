# Model Adapters

Gli adapter servono a dare a tutti i modelli la stessa interfaccia logica.

Interfaccia minima attesa:
- input: prompt o lista di messaggi
- output: testo grezzo del modello
- metadati: id modello, token usage, latency, seed, parametri di generazione

Adapter previsti:
- `hf_local.py`: modelli locali Hugging Face
- `openai_api.py`: modelli serviti via API
- `vllm.py`: modelli esposti via server vLLM

Regola:
- il runner non deve conoscere i dettagli del provider
- il confronto fra modelli deve passare sempre dalla stessa interfaccia comune
