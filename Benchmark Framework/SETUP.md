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
python -c "import torch, transformers, accelerate; print(torch.__version__); print(transformers.__version__)"
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

## 4. Registry dei modelli

Prima del run:

- lascia `enabled: true` solo sui modelli che vuoi davvero lanciare
- su macchine deboli preferisci modelli piccoli
- se non hai `accelerate` o non vuoi usare la GPU, imposta `device_map: none`

Esempi ragionevoli per test locali:

- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

## 5. Primo run

Dalla root della repo:

```powershell
python "Benchmark Framework/run_benchmark.py" --use-real-validator --validator-command "Validate"
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

## 7. Problemi comuni

### Errore su `device_map: auto`

Se vedi un errore che dice che serve `accelerate`:

- installa `accelerate`
- oppure imposta `device_map: none` nel `model_registry.yaml`

### Warning su symlink di Hugging Face su Windows

Non blocca il benchmark. Significa solo che la cache usera piu spazio disco.

### Benchmark troppo pesante

Riduci il numero di modelli attivi e inizia con uno solo.
