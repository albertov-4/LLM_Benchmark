# Setup

This file describes the minimum setup required to run the benchmark on another machine.

## 1. Python Environment

From the `LLM_Benchmark` repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r "Benchmark Framework/requirements.txt"
```

Quick check:

```powershell
python -c "import torch, transformers, accelerate, openai; print(torch.__version__); print(transformers.__version__)"
```

## 2. External `VAL` Validator

`VAL` is not a Python dependency. It must be installed separately.

Two options are supported:
- add the folder containing `Validate.exe` to `PATH`
- pass the full executable path to the launcher with `--validator-command`

Check whether `VAL` is available:

```powershell
Validate -h
```

Or use the full executable path:

```powershell
& "C:\full\path\to\Validate.exe" -h
```

## 3. Hugging Face

The benchmark can download models from the Hugging Face Hub.

Optional but recommended:

```powershell
$env:HF_TOKEN="your_token"
```

Without a token, the benchmark can still run, but Hub rate limits may be lower.

Before a real run, prepare the selected models:

```powershell
python "Benchmark Framework/scripts/prepare_models.py" --models-dir models_cache
```

`models_cache` is a local preparation/cache directory and should not be versioned.

On HPC systems such as Leonardo, prepare models in a separate stage and use local paths in the registry during benchmark jobs.

## 4. Model Registry

Before running the benchmark:
- the launcher default uses `models/model_registry_nvidia.yaml`
- keep `enabled: true` only on models that should actually run
- prefer small models on limited hardware
- if `accelerate` is unavailable or GPU placement is not needed, set `device_map: none`
- use `--adapter` to select a backend such as `hf_local`, `nvidia_api`, `ollama` or `llama_cpp_cli`
- use `--model-registry-path` for a custom registry file; when both `--adapter` and `--model-registry-path` are provided, the explicit path takes precedence

For Ollama:
- install Ollama separately
- start the Ollama service
- download a model, for example `ollama pull qwen2.5:0.5b`
- enable a registry entry with `adapter: ollama`

For NVIDIA API models, use environment variables or a local secrets file ignored by Git.

Environment variable option:

```powershell
$env:NVIDIA_PHI_API_KEY="your_key"
```

Local secrets file option:

```powershell
Copy-Item "Benchmark Framework/secrets.local.example.json" "Benchmark Framework/secrets.local.json"
```

Then add the keys to `Benchmark Framework/secrets.local.json`. The real local secrets file is ignored by Git.

Reasonable local test models:
- `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- `Qwen/Qwen2.5-0.5B-Instruct`
- `HuggingFaceTB/SmolLM2-360M-Instruct`

## 5. First Run

From the repository root:

```powershell
python "Benchmark Framework/run_benchmark.py" --use-real-validator --validator-command "Validate"
```

Run only one protocol:

```powershell
python "Benchmark Framework/run_benchmark.py" --protocol-id direct_plan --use-real-validator
```

Use local Hugging Face models:

```powershell
python "Benchmark Framework/run_benchmark.py" --adapter hf_local --protocol-id direct_plan --use-real-validator
```

Run only one model:

```powershell
python "Benchmark Framework/run_benchmark.py" --model-id nvidia_gemma_4_31b_it --use-real-validator
```

Run only one task family, tier or instance:

```powershell
python "Benchmark Framework/run_benchmark.py" --task-family <task_family> --tier <tier> --instance-id <instance_id> --use-real-validator
```

Check that task PDDL files are readable by `VAL` before launching model jobs:

```powershell
python "Benchmark Framework/run_benchmark.py" --preflight-tasks --use-real-validator
```

If `Validate` is not in `PATH`:

```powershell
python "Benchmark Framework/run_benchmark.py" --use-real-validator --validator-command "C:\full\path\to\Validate.exe"
```

## 6. Saved Outputs

The benchmark saves:
- one timestamped folder under each output area: `raw/`, `parsed/` and `scored/`
- final suite JSON in `Benchmark Framework/outputs/scored/<timestamp>/suite_result.json`
- raw model output per job in `Benchmark Framework/outputs/raw/<timestamp>/...`
- parsed plan output per job in `Benchmark Framework/outputs/parsed/<timestamp>/...`
- scored validation output per job in `Benchmark Framework/outputs/scored/<timestamp>/...`
- iteration details in `attempts`, including prompts/messages and feedback

To clear generated outputs while preserving folders and `.gitkeep` files:

```powershell
python "Benchmark Framework/clear_outputs.py"
```

## 7. Common Issues

### `device_map: auto` Error

If an error says that `accelerate` is required:
- install `accelerate`
- or set `device_map: none` in the registry entry being used

### Hugging Face Symlink Warning on Windows

This does not block the benchmark. It only means the local cache may use more disk space.

### Benchmark Too Heavy

Reduce the number of enabled models, use `--model-id`, or use `--protocol-id` to reduce the job matrix.
