# Leonardo SLURM Scripts

This folder contains SLURM wrappers for running the benchmark workflow on
Leonardo or a similar HPC environment. The scripts locate `Benchmark_Framework`
from the script location or `SLURM_SUBMIT_DIR`, activate an existing Python
environment, and then call the same Python entry points used locally.

## Scripts

- `prepare_models.sh`: prepares Hugging Face models into `models_cache` or a
  configured model directory.
- `setup_leonardo_env.sh`: installs the selected Leonardo Python environment.
  The default `our_env` profile pins PyTorch CUDA 12.1 and installs
  `mamba-ssm==2.2.4` without upgrading Torch; the `gptoss_env` profile uses the
  GPT-OSS requirements and skips those `our_env`-specific pins.
- `test_models_cache.sh`: checks prepared model directories in offline mode.
- `test_benchmark.sh`: runs a narrow benchmark job for one model, protocol, task
  family, tier, and instance.
- `run_benchmark.sh`: launcher that creates one `RUN_ID` and submits the
  model-task benchmark jobs in `jobs/`.
- `run_benchmark_single.sh`: shared worker used by the model-task jobs. It
  activates the environment, checks CUDA and VAL, then calls `run_benchmark.py`.
- `jobs/*.sh`: static SLURM jobs for one Hugging Face model and one task family.
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
- `OUTPUT_JSON`: optional suite summary path passed to `run_benchmark.py`.

GPU benchmark jobs also load `gcc/12.2.0`, use CUDA 12.1 from
`/leonardo/prod/opt/compilers/cuda/12.1/none`, set `MAX_JOBS=1`, and target
A100 GPUs with `TORCH_CUDA_ARCH_LIST=8.0`.

## Leonardo Python Environment

Use the setup script when creating or repairing the Leonardo venv:

```bash
PYTHON_VENV=/leonardo_scratch/large/userexternal/avarini0/our_env \
sbatch Benchmark_Framework/Leonardo_script/setup_leonardo_env.sh
```

The script defaults to `our_env`. It switches to the GPT-OSS profile when
`PYTHON_VENV` ends in `gptoss_env`, or when you set it explicitly:

```bash
LEONARDO_ENV_PROFILE=gptoss_env \
PYTHON_VENV=/leonardo_scratch/large/userexternal/avarini0/gptoss_env \
sbatch Benchmark_Framework/Leonardo_script/setup_leonardo_env.sh
```

It runs the equivalent of:

```bash
module load gcc/12.2.0
source /leonardo_scratch/large/userexternal/avarini0/our_env/bin/activate

export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.1/none
export CUDACXX=$CUDA_HOME/bin/nvcc
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export CC=$(which gcc)
export CXX=$(which g++)
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.0"

pip install -r Benchmark_Framework/requirements/leonardo-our-env.txt

pip install --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir --no-deps --no-build-isolation mamba-ssm==2.2.4 -v
```

Do not use plain `pip install mamba-ssm`; it can upgrade Torch and Triton to
versions that are incompatible with this CUDA 12.1 environment.

The setup script finishes with:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torchvision, torchaudio; print(torchvision.__version__, torchaudio.__version__)"
python -c "import mamba_ssm; print('mamba ok')"
python -c "import triton; print(triton.__version__)"
pip check
```

Expected versions are `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`,
`torchaudio==2.5.1+cu121`, `triton==3.1.0`, `mamba-ssm==2.2.4`, and
`pip check` reporting no broken requirements.

`hf_gpt_oss_120b` uses a separate environment:

```bash
LEONARDO_ENV_PROFILE=gptoss_env \
PYTHON_VENV=/leonardo_scratch/large/userexternal/avarini0/gptoss_env \
sbatch Benchmark_Framework/Leonardo_script/setup_leonardo_env.sh
```

The launcher routes that model through `GPTOSS_PYTHON_VENV` or the default
`/leonardo_scratch/large/userexternal/avarini0/gptoss_env` path.

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

For model-specific debugging notes on `hf_qwen_3_6_27b` and
`hf_nemotron_3_nano_30b_a3b`, including the Mamba/Transformers shim,
`libstdc++` preload, Qwen cache naming, and Nemotron generation issues, see
`QWEN_NEMOTRON_NOTES.md`.

Run a small validation job:

```bash
MODEL_ID=hf_gemma_4_31b_it \
PROTOCOL_ID=direct_plan \
TASK_FAMILY=fo-sailing \
TIER=easy \
INSTANCE_ID=pfile1 \
sbatch Benchmark_Framework/Leonardo_script/test_benchmark.sh
```

Run the split model-task benchmark launcher:

```bash
bash Benchmark_Framework/Leonardo_script/run_benchmark.sh
```

The launcher submits every script under `Leonardo_script/jobs/` with the same
`RUN_ID`. Each job runs one model and one task family with a one day SLURM time
limit. Override the protocol for all submitted jobs from the launcher
environment if needed:

```bash
PROTOCOL_ID=direct_plan bash Benchmark_Framework/Leonardo_script/run_benchmark.sh
```

Logs are written under:

```text
Benchmark_Framework/slurm_logs/<run_id>/
```

Per-case artifacts share the same run folder. Per-job suite summaries are saved
under:

```text
Benchmark_Framework/outputs/scored/<run_id>/suite_results/<model_id>__<task_family>.json
```

`test_benchmark.sh` and `run_benchmark_single.sh` call `run_benchmark.py` with
`--adapter hf_local`, `--use-real-validator`, `--validator-command`, and
`--preflight-tasks`.

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
