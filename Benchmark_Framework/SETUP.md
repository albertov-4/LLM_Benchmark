# Setup

This guide covers the minimum setup needed to run `Benchmark_Framework` on a
local machine or on an HPC system such as Leonardo.

## Python Environment

From the `LLM_Benchmark` repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r Benchmark_Framework/requirements.txt
```

Quick dependency check:

```powershell
python -c "import torch, transformers, accelerate, openai; print(torch.__version__); print(transformers.__version__)"
```

## Leonardo CUDA 12.1 Environment

On Leonardo, keep PyTorch pinned to the CUDA 12.1 wheel set used by the
benchmark jobs:

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
```

Restore the supported PyTorch matrix with:

```bash
pip install --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

Install Mamba without allowing pip to upgrade Torch or Triton:

```bash
pip install --no-cache-dir --no-deps --no-build-isolation mamba-ssm==2.2.4 -v
```

Do not run plain `pip install mamba-ssm`; newer releases can pull incompatible
Torch, Triton, or CUDA wheels. The repository also provides:

```bash
PYTHON_VENV=/leonardo_scratch/large/userexternal/avarini0/our_env \
sbatch Benchmark_Framework/Leonardo_script/setup_leonardo_env.sh
```

Final checks:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import torchvision, torchaudio; print(torchvision.__version__, torchaudio.__version__)"
python -c "import mamba_ssm; print('mamba ok')"
python -c "import triton; print(triton.__version__)"
pip check
```

Expected: `torch 2.5.1+cu121`, CUDA `12.1`, `torchvision 0.20.1+cu121`,
`torchaudio 2.5.1+cu121`, `triton 3.1.0`, `mamba_ssm` import OK, and
`No broken requirements found`.

## VAL Validator

`VAL` is an external PDDL validator, not a Python dependency. The benchmark can
use it in three ways:

- Put `Validate` or `Validate.exe` on `PATH`.
- Pass a full executable path with `--validator-command`.
- Pass one of the bundled platform utilities under `utils/linux64/bin/` or
  `utils/win64/bin/` with `--validator-command` when it works on the target
  machine.

Check whether VAL is available:

```powershell
Validate -h
```

Or use a full path:

```powershell
& "C:\full\path\to\Validate.exe" -h
```

## Model Registries

The launcher defaults to `models/model_registry_nvidia.yaml`. You can select a
backend registry with `--adapter` or provide a registry path explicitly:

```powershell
python Benchmark_Framework/run_benchmark.py --adapter hf_local --protocol-id direct_plan --use-real-validator
python Benchmark_Framework/run_benchmark.py --model-registry-path models/model_registry_ollama.yaml --model-id <model_id>
```

Supported registry families:

- `nvidia_api`: remote NVIDIA API models.
- `hf_local`: local Hugging Face Transformers models.
- `ollama`: models served by a local Ollama process.
- `llama_cpp_cli`: local GGUF models executed through llama.cpp.

Before a run, keep `enabled: true` only on models that should participate when
no `--model-id` filter is provided.

## Hugging Face Models

Local Hugging Face runs can load a model from a Hub repo id, a prepared
`models_cache` entry, or an explicit local path in `weights_path`.

Optional token:

```powershell
$env:HF_TOKEN="your_token"
```

Prepare enabled Hugging Face models:

```powershell
python Benchmark_Framework/scripts/prepare_models.py --models-dir models_cache
```

Useful preparation modes:

```powershell
python Benchmark_Framework/scripts/prepare_models.py --model-id hf_gemma_4_31b_it
python Benchmark_Framework/scripts/prepare_models.py --dry-run
python Benchmark_Framework/scripts/prepare_models.py --offline
```

`models_cache` is a local preparation/cache directory and should not be
versioned. On HPC systems, prepare models before GPU benchmark jobs and point
registry `weights_path` values to the prepared local directories.

## NVIDIA API Models

For NVIDIA API models, set the environment variable referenced by `api_key_env`
or use a local secrets file ignored by Git.

Environment variable option:

```powershell
$env:NVIDIA_PHI_API_KEY="your_key"
```

Local secrets file option:

```json
{
  "NVIDIA_GEMMA_API_KEY": "your_key",
  "NVIDIA_GPT_OSS_API_KEY": "your_key"
}
```

Store that JSON object in `Benchmark_Framework/secrets.local.json`. The file is
ignored by Git. Keys must match the `api_key_env` value used by each enabled
entry in `models/model_registry_nvidia.yaml`.

Streaming entries can use:

- `stream`: enables streamed responses.
- `timeout_seconds`: API/client timeout.
- `job_timeout_seconds`: optional total attempt timeout for streamed runs.
- `debug_stream`: prints streaming diagnostics.

If a stream is interrupted after text has been received, the adapter returns the
partial text and records `partial_output`, `stream_complete`, `stream_error`,
and `timed_out_by_job_limit` in the generation payload.

## First Runs

Run the default registry:

```powershell
python Benchmark_Framework/run_benchmark.py --use-real-validator --validator-command Validate
```

Run one protocol:

```powershell
python Benchmark_Framework/run_benchmark.py --protocol-id direct_plan --use-real-validator
```

Run one model:

```powershell
python Benchmark_Framework/run_benchmark.py --model-id nvidia_gemma_4_31b_it --use-real-validator
```

Run one task family, tier, and instance:

```powershell
python Benchmark_Framework/run_benchmark.py --task-family fo-sailing --tier easy --instance-id pfile1 --use-real-validator
```

Check all selected task domain/problem files with VAL before launching model
jobs:

```powershell
python Benchmark_Framework/run_benchmark.py --preflight-tasks --use-real-validator
```

If `Validate` is not on `PATH`:

```powershell
python Benchmark_Framework/run_benchmark.py --use-real-validator --validator-command "C:\full\path\to\Validate.exe"
```

## Saved Outputs

Each run writes one folder per `run_id` under each output area:

```text
Benchmark_Framework/outputs/raw/<run_id>/...
Benchmark_Framework/outputs/parsed/<run_id>/...
Benchmark_Framework/outputs/scored/<run_id>/...
Benchmark_Framework/outputs/scored/<run_id>/suite_result.json
```

Split Leonardo runs can write one suite summary per submitted model-task job:

```text
Benchmark_Framework/outputs/scored/<run_id>/suite_results/<model_id>__<task_family>.json
```

`raw` contains messages and generation payloads, `parsed` contains extracted
plans and parser issues, and `scored` contains validation results, repair
feedback, metrics, and artifact paths.

Clear generated outputs while preserving the folder structure:

```powershell
python Benchmark_Framework/clear_outputs.py
```

## Common Issues

`device_map: auto` requires `accelerate`. Install `accelerate` or set
`device_map: none` for the selected registry entry.

Hugging Face symlink warnings on Windows do not block the benchmark. They only
mean the local cache may use more disk space.

If a benchmark run is too heavy, reduce enabled models, use `--model-id`, filter
the task matrix, or run only one protocol.
