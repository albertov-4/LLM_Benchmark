# Leonardo SLURM Scripts

This folder contains SLURM wrappers for running the benchmark workflow on
Leonardo or a similar HPC environment. The scripts locate `Benchmark_Framework`
from the script location or `SLURM_SUBMIT_DIR`, activate an existing Python
environment, and then call the same Python entry points used locally.

## Scripts

- `prepare_models.sh`: prepares Hugging Face models into `models_cache` or a
  configured model directory.
- `test_models_cache.sh`: checks prepared model directories in offline mode.
- `test_benchmark.sh`: runs a narrow benchmark job for one model, protocol, task
  family, tier, and instance.
- `run_benchmark.sh`: runs a broader Hugging Face benchmark job.
- `clear_outputs.sh`: lists or clears generated outputs through
  `clear_outputs.py`.

## Environment

The scripts load `python/3.11.7` and then try to activate a virtual environment
from common locations. You can make this explicit:

```bash
export PYTHON_VENV=/absolute/path/to/venv
```

Common variables:

- `HF_HOME`: Hugging Face cache location.
- `HF_TOKEN`: token for gated or rate-limited Hub downloads.
- `VALIDATOR_COMMAND`: `Validate` or an absolute path to the VAL executable.
- `MODEL_ID`: model id from the selected registry.
- `PROTOCOL_ID`: protocol id such as `direct_plan` or `iterative_repair`.
- `TASK_FAMILY`, `TIER`, `INSTANCE_ID`: task filters.
- `RUN_ID`: output run folder name.

## Model Preparation

Prepare models before GPU benchmark jobs:

```bash
sbatch Benchmark_Framework/Leonardo_script/prepare_models.sh
```

Useful overrides:

```bash
MODEL_ID=hf_gemma_4_31b_it MODELS_DIR=/leonardo_work/YOUR_ACCOUNT/models sbatch Benchmark_Framework/Leonardo_script/prepare_models.sh
DRY_RUN=1 sbatch Benchmark_Framework/Leonardo_script/prepare_models.sh
OFFLINE=1 sbatch Benchmark_Framework/Leonardo_script/prepare_models.sh
```

After preparation, point the selected Hugging Face registry entries to local
model directories when running offline jobs.

## Cache Check

Check prepared model directories without network access:

```bash
MODELS_DIR=/leonardo_work/YOUR_ACCOUNT/models sbatch Benchmark_Framework/Leonardo_script/test_models_cache.sh
```

This script verifies that configs, tokenizers, weights, and shard indexes look
complete.

## Benchmark Jobs

Run a small validation job:

```bash
MODEL_ID=hf_gemma_4_31b_it \
PROTOCOL_ID=direct_plan \
TASK_FAMILY=fo-sailing \
TIER=easy \
INSTANCE_ID=pfile1 \
sbatch Benchmark_Framework/Leonardo_script/test_benchmark.sh
```

Run the broader benchmark wrapper:

```bash
PROTOCOL_ID=iterative_repair sbatch Benchmark_Framework/Leonardo_script/run_benchmark.sh
```

Both benchmark scripts call `run_benchmark.py` with `--adapter hf_local`,
`--use-real-validator`, `--validator-command`, and `--preflight-tasks`.

## Output Cleanup

By default the cleanup script lists generated outputs without deleting them:

```bash
sbatch Benchmark_Framework/Leonardo_script/clear_outputs.sh
```

To delete generated outputs:

```bash
CONFIRM_CLEAR_OUTPUTS=1 sbatch Benchmark_Framework/Leonardo_script/clear_outputs.sh
```

The underlying Python script preserves output folders and `.gitkeep` files.
