# Models

Questa cartella contiene il registry dei modelli e gli adapter usati dal benchmark.

Componenti:
- `model_registry.yaml`: elenco dei modelli disponibili per la suite
- `adapters/`: implementazioni compatibili con l'interfaccia `generate(messages)`

Campi runtime utili nel registry:
- `enabled`: permette di escludere un modello senza cancellarlo dal registry
- `weights_path`: path locale oppure identificatore `from_pretrained(...)`
- `device_map`: per esempio `auto` oppure `none`
- `torch_dtype`: per esempio `auto`, `float16`, `bfloat16`, `float32`
- `trust_remote_code`: flag passato al backend HF
- `use_chat_template` e `add_generation_prompt`: controllano il rendering dei messaggi per modelli chat

Nota pratica:
- i modelli attivi nel registry sono pensati come starter set locale
- se hai gia pesi locali, puoi sostituire `weights_path` con un path assoluto o relativo
