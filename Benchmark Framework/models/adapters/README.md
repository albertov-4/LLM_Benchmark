# Model Adapters

Gli adapter servono a dare a tutti i modelli la stessa interfaccia logica.

Interfaccia minima attesa:
- input: prompt o lista di messaggi
- output: testo grezzo del modello
- metadati: id modello, token usage, latency, seed, parametri di generazione

Adapter previsti:
- `hf_local.py`: modelli locali Hugging Face
- `openai_api.py`: modelli serviti via API

Nota:
- al momento il flow reale del framework usa soprattutto `hf_local.py`
- `openai_api.py` resta uno scaffold per integrazioni future via API

Regola:
- il runner non deve conoscere i dettagli del provider
- il confronto fra modelli deve passare sempre dalla stessa interfaccia comune
