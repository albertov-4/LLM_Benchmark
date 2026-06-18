# Leonardo Requirements

This folder contains curated requirements for the Python environments used on
Leonardo. They are not full `pip freeze` lock files; they keep the direct
runtime dependencies and the pins that mattered in the working environments.

## Environments

| File | Environment | Models |
| --- | --- | --- |
| `leonardo-our-env.txt` | `our_env` | `hf_gemma_4_31b_it`, `hf_nemotron_3_nano_30b_a3b`, `hf_phi_4`, `hf_qwen_3_6_27b` |
| `leonardo-gptoss-env.txt` | `gptoss_env` | `hf_gpt_oss_120b` |

## Install our_env

```bash
module load python/3.11.7
module load gcc/12.2.0
module load cuda/12.1
source /leonardo_scratch/large/userexternal/avarini0/our_env/bin/activate

pip install -r Benchmark_Framework/requirements/leonardo-our-env.txt

pip install --force-reinstall \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

pip install --no-cache-dir --no-deps --no-build-isolation mamba-ssm==2.2.4 -v
```

Do not install `mamba-ssm` with a plain `pip install mamba-ssm`; it can upgrade
Torch, Triton, or CUDA wheels away from the CUDA 12.1 stack used by Leonardo
jobs.

## Install gptoss_env

```bash
module load python/3.11.7
source /leonardo_scratch/large/userexternal/avarini0/gptoss_env/bin/activate

pip install -r Benchmark_Framework/requirements/leonardo-gptoss-env.txt
```

Or use the profile-aware setup script:

```bash
LEONARDO_ENV_PROFILE=gptoss_env \
PYTHON_VENV=/leonardo_scratch/large/userexternal/avarini0/gptoss_env \
sbatch Benchmark_Framework/Leonardo_script/setup_leonardo_env.sh
```

`hf_gpt_oss_120b` is routed separately by the SLURM launcher through
`GPTOSS_PYTHON_VENV` or the default
`/leonardo_scratch/large/userexternal/avarini0/gptoss_env` path.

## Checks

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import transformers; print(transformers.__version__)"
pip check
```
