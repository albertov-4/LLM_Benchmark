# Setup

Questo file descrive il setup minimo per far partire il benchmark su un altro PC.

## 1. Ambiente Python

Dalla root di `LLM_Benchmark`:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r "Benchmark Framework/requirements.txt"
```

Verifica rapida:

```powershell
python -c "import torch, transformers, accelerate, openai; print(torch.__version__); print(transformers.__version__)"
```

## 2. Validator esterno `VAL`

`VAL` non e una dipendenza Python del progetto. Va installato a parte.

Due opzioni:

- aggiungere al `PATH` la cartella che contiene `Validate.exe`
- oppure passare il path completo al launcher con `--validator-command`

Verifica:

```powershell
Validate -h
```

oppure:

```powershell
& "C:\percorso\completo\Validate.exe" -h
```

## 3. Hugging Face

Il benchmark puo scaricare modelli dalla Hub di Hugging Face.

Opzionale ma consigliato:

```powershell
$env:HF_TOKEN="il_tuo_token"
```

Senza token il benchmark puo comunque partire, ma con rate limit piu bassi.

Prima di un run reale e consigliato preparare i modelli:

```powershell
python "Benchmark Framework/scripts/prepare_models.py" --models-dir models_cache
```

Su HPC/Leonardo prepara i modelli in una fase separata e usa path locali nel registry.

## 4. Registry dei modelli

Prima del run:

- il default del launcher usa `models/model_registry_nvidia.yaml`
- lascia `enabled: true` solo sui modelli che vuoi davvero lanciare
- su macchine deboli preferisci modelli piccoli
- se non hai `accelerate` o non vuoi usare la GPU, imposta `device_map: none`
- per cambiare backend usa `--adapter`, per esempio `hf_local`, `nvidia_api`, `ollama` o `llama_cpp_cli`
- per usare un file custom usa `--model-registry-path`; se lo passi insieme a `--adapter`, vince il path manuale

Per Ollama:

- installa Ollama separatamente
- avvia il servizio Ollama
- scarica un modello, per esempio `ollama pull qwen2.5:0.5b`
- abilita nel registry una voce con `adapter: ollama`

Per NVIDIA API puoi usare variabili d'ambiente oppure un file locale ignorato da Git.

Opzione variabile d'ambiente:

```powershell
$env:NVIDIA_PHI_API_KEY="la_tua_chiave"
```

Opzione file locale:

```powershell
Copy-Item "Benchmark Framework/secrets.local.example.json" "Benchmark Framework/secrets.local.json"
```

Poi inserisci le chiavi in `Benchmark Framework/secrets.local.json`. Il file reale e ignorato da Git.

Esempi ragionevoli per test locali:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

## 5. Primo run

Dalla root della repo:

```powershell
python "Benchmark Framework/run_benchmark.py" --use-real-validator --validator-command "Validate"
```

Per limitare il test a un protocollo:

```powershell
python "Benchmark Framework/run_benchmark.py" --protocol-id direct_plan --use-real-validator
```

Per usare Hugging Face locale:

```powershell
python "Benchmark Framework/run_benchmark.py" --adapter hf_local --protocol-id direct_plan --use-real-validator
```

Per limitare il test a un solo modello:

```powershell
python "Benchmark Framework/run_benchmark.py" --model-id nvidia_gemma_4_31b_it --use-real-validator
```

Se `Validate` non e nel `PATH`:

```powershell
python "Benchmark Framework/run_benchmark.py" --use-real-validator --validator-command "C:\percorso\completo\Validate.exe"
```

## 6. Output salvati

Il benchmark salva:

- JSON finale della suite in `Benchmark Framework/outputs/scored/suite_result_latest.json`
- output raw per job in `Benchmark Framework/outputs/raw/...`
- output parsed per job in `Benchmark Framework/outputs/parsed/...`
- output scored per job in `Benchmark Framework/outputs/scored/...`

Per pulire gli output generati mantenendo cartelle e `.gitkeep`:

```powershell
python "Benchmark Framework/clear_outputs.py"
```

## 7. Problemi comuni

### Errore su `device_map: auto`

Se vedi un errore che dice che serve `accelerate`:

- installa `accelerate`
- oppure imposta `device_map: none` nel registry del modello che stai usando

### Warning su symlink di Hugging Face su Windows

Non blocca il benchmark. Significa solo che la cache usera piu spazio disco.

### Benchmark troppo pesante

Riduci il numero di modelli attivi, usa `--model-id` oppure usa `--protocol-id` per ridurre la matrice dei job.
