# Model Preparation

This benchmark can run on a local PC or on an HPC system such as Leonardo.
The important rule is simple: local runs may download models, but HPC benchmark
jobs should use model files that were prepared beforehand.

## Hugging Face repo id vs local path

When Hugging Face receives a repo id such as `Qwen/Qwen2.5-1.5B-Instruct`, it
may contact the Hugging Face Hub and download files into a cache or local
directory.

When Hugging Face receives a local path such as
`/leonardo_work/YOUR_ACCOUNT/models/qwen2_5_1_5b_instruct`, it loads files from
disk and does not need internet access.

For HPC jobs, use local paths in the selected `models/model_registry_*.yaml`.
Compute nodes may not have internet access, and large downloads should not
happen during GPU jobs.

## Local PC workflow

From the repository root:

```powershell
python "Benchmark Framework/scripts/prepare_models.py" --models-dir models_cache
python "Benchmark Framework/run_benchmark.py" --use-real-validator
```

If `weights_path` in the registry is a Hugging Face repo id, the preparation
script downloads it into `Benchmark Framework/models_cache`.
This directory is a local cache/preparation directory and should not be
committed to the repository.

## Leonardo/HPC preparation stage

Run this in a stage where internet access and storage writes are allowed:

```bash
python "Benchmark Framework/scripts/prepare_models.py" \
  --models-dir /leonardo_work/YOUR_ACCOUNT/models
```

After preparation, update the HPC registry entries so `weights_path` points to
the prepared local model directory.

## Leonardo/HPC benchmark stage

Run benchmark jobs offline from local paths:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python "Benchmark Framework/run_benchmark.py" \
  --use-real-validator \
  --model-registry-path models/model_registry_hf.yaml
```

The benchmark job should not download models. If a model is missing, fix the
preparation stage or the local `weights_path` before launching GPU jobs.

## Useful preparation options

Prepare one model:

```powershell
python "Benchmark Framework/scripts/prepare_models.py" --model-registry-path models/model_registry_hf.yaml --model-id hf_phi_4_mini_instruct
```

Check what would happen without downloading:

```powershell
python "Benchmark Framework/scripts/prepare_models.py" --dry-run
```

Verify offline readiness:

```powershell
python "Benchmark Framework/scripts/prepare_models.py" --offline
```
