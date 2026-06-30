# Leonardo SLURM Scripts

This folder contains SLURM wrappers for running the benchmark workflow on
Leonardo or a similar HPC environment. The full from-zero setup is documented
in [../docs/leonardo_setup_from_zero.md](../docs/leonardo_setup_from_zero.md).

## Scripts

- `setup_leonardo_env.sh`: legacy/profile-aware environment repair helper. The
  default documented setup is the manual procedure in the Leonardo guide.
- `prepare_models.sh`: wrapper around `scripts/prepare_models.py`; manual
  `hf download` into `models_cache` is the default documented download path.
- `test_models_cache.sh`: offline model-cache check; it does not need GPU.
- `test_benchmark.sh`: small GPU benchmark job for one model/task/protocol.
- `run_benchmark.sh`: submits split GPU benchmark jobs through SLURM.
- `run_benchmark_single.sh`: shared worker used by split benchmark jobs.
- `clear_outputs.sh`: lists or clears generated outputs through
  `clear_outputs.py`.

## Required Environment Variables

Set these before submitting benchmark jobs:

```bash
export SLURM_ACCOUNT="<CINECA_PROJECT_ACCOUNT>"
export PYTHON_VENV=$CINECA_SCRATCH/our_env
export GPTOSS_PYTHON_VENV=$CINECA_SCRATCH/gptoss_env
```

Common optional variables:

- `HF_HOME`: Hugging Face cache location.
- `HF_TOKEN`: token for gated or rate-limited Hub downloads.
- `VALIDATOR_COMMAND`: `Validate` or an absolute path to the VAL executable.
- `MODEL_ID`, `PROTOCOL_ID`, `TASK_FAMILY`, `TIER`, `INSTANCE_ID`: run filters.
- `RUN_ID`: output run folder name.
- `OUTPUT_JSON`: optional suite summary path passed to `run_benchmark.py`.

## Model Downloads And Cache Checks

Do not request GPU/Booster for simple Hugging Face downloads or offline cache
checks. The working cache location is:

```bash
$CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework/models_cache
```

Use the Leonardo setup guide for the manual `hf download` commands and offline
cache check. Use GPU/Booster only for actual local inference and benchmark jobs.

## Benchmark Jobs

Run a small validation job:

```bash
cd $CINECA_SCRATCH/LLM_Benchmark
export SLURM_ACCOUNT="<CINECA_PROJECT_ACCOUNT>"
MODEL_ID=hf_gemma_4_31b_it \
PROTOCOL_ID=direct_plan \
TASK_FAMILY=fo-sailing \
TIER=easy \
INSTANCE_ID=pfile1 \
sbatch --account="$SLURM_ACCOUNT" Benchmark_Framework/Leonardo_script/test_benchmark.sh
```

Run the split model-task launcher:

```bash
cd $CINECA_SCRATCH/LLM_Benchmark
SLURM_ACCOUNT="<CINECA_PROJECT_ACCOUNT>" \
bash Benchmark_Framework/Leonardo_script/run_benchmark.sh
```

Logs are written under:

```text
Benchmark_Framework/slurm_logs/<run_id>/
```

Per-job suite summaries are saved under:

```text
Benchmark_Framework/outputs/scored/<run_id>/suite_results/<model_id>__<task_family>.json
```

## Output Cleanup

By default the cleanup script lists generated outputs without deleting them:

```bash
sbatch Benchmark_Framework/Leonardo_script/clear_outputs.sh
```

To delete generated outputs:

```bash
CONFIRM_CLEAR_OUTPUTS=1 sbatch Benchmark_Framework/Leonardo_script/clear_outputs.sh
```
