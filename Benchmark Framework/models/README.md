# Models

Questa cartella contiene il registry dei modelli e gli adapter usati dal benchmark.

Componenti:
- `model_registry_nvidia.yaml`: registry principale per NVIDIA API
- `model_registry_hf.yaml`: equivalenti Hugging Face dei modelli NVIDIA
- `model_registry_ollama.yaml`: equivalenti Ollama disponibili o plausibili
- `model_registry_llama_cpp.yaml`: equivalenti GGUF per llama.cpp
- `adapters/`: implementazioni compatibili con l'interfaccia `generate(messages)`

Campi runtime utili nel registry:
- `enabled`: permette di escludere un modello senza cancellarlo dal registry
- `adapter`: seleziona il backend, per esempio `hf_local` oppure `ollama`
- `weights_path`: path locale oppure identificatore `from_pretrained(...)`
- `ollama_model`: nome del modello installato in Ollama, per esempio `qwen2.5:0.5b`
- `base_url`: endpoint locale di Ollama, di default `http://localhost:11434`
- `api_model_name`: nome modello remoto usato dagli adapter API
- `api_key_env`: nome della variabile d'ambiente che contiene la API key; puo essere diverso per ogni modello
- `api_mode`: modalita API, per NVIDIA `chat_completions` oppure `responses`
- `stream`: abilita la lettura streaming dove supportata
- `thinking_key` e `thinking_enabled`: configurano `chat_template_kwargs`
- `reasoning_budget`: parametro NVIDIA opzionale per modelli reasoning
- `executable_path`: path del binario llama.cpp, per esempio `llama-cli.exe`
- `model_path`: path del file `.gguf` per llama.cpp
- `device_map`: per esempio `auto` oppure `none`
- `torch_dtype`: per esempio `auto`, `float16`, `bfloat16`, `float32`
- `timeout_seconds`: timeout di chiamata per adapter HTTP/API
- `trust_remote_code`: flag passato al backend HF
- `use_chat_template` e `add_generation_prompt`: controllano il rendering dei messaggi per modelli chat

Nota pratica:
- `model_registry_nvidia.yaml` resta il registry principale per NVIDIA API
- i registry alternativi hanno i modelli disabilitati di default per evitare download o esecuzioni pesanti accidentali
- se hai gia pesi locali, puoi sostituire `weights_path` con un path assoluto o relativo
- per usare Ollama, avvia il servizio e abilita un modello con `adapter: ollama`
- per NVIDIA, imposta la variabile indicata da `api_key_env` oppure mettila in `Benchmark Framework/secrets.local.json`
- per llama.cpp, sostituisci `model_path: REPLACE_WITH_LOCAL_GGUF_FILE` con il path reale del file `.gguf`

Esempi:
- NVIDIA: `python "Benchmark Framework/run_benchmark.py" --adapter nvidia_api --model-id nvidia_phi_4_mini_instruct`
- Hugging Face: `python "Benchmark Framework/run_benchmark.py" --adapter hf_local --model-id hf_qwen2_5_1_5b_instruct_awq`
- Ollama: `python "Benchmark Framework/run_benchmark.py" --adapter ollama --model-id ollama_phi_4_mini_instruct`
- llama.cpp: `python "Benchmark Framework/run_benchmark.py" --adapter llama_cpp_cli --model-id llamacpp_phi_4_mini_instruct`

`--adapter` seleziona automaticamente il registry coerente. Se vuoi usare un file YAML custom, passa `--model-registry-path`.
