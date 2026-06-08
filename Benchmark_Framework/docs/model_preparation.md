# Model Preparation

The benchmark can run on a local PC or on an HPC system such as Leonardo. The
important operational rule is that GPU benchmark jobs should not download large
model weights. Prepare Hugging Face models first, then run benchmarks from local
files.

## Hugging Face Repo Ids And Local Paths

A `weights_path` such as `Qwen/Qwen2.5-1.5B-Instruct` is a Hugging Face repo id.
If the files are not already prepared locally, Transformers or
`huggingface_hub` may contact the Hub.

A `weights_path` such as `/leonardo_work/YOUR_ACCOUNT/models/qwen2_5_1_5b` is a
local path. The benchmark loads files from disk and does not need internet
access.

For HPC runs, prefer local paths in `models/model_registry_hf.yaml`. Compute
nodes may not have internet access, and large downloads should happen in a
separate preparation stage.

## Local Workflow

From the repository root:

```powershell
python Benchmark_Framework/scripts/prepare_models.py --models-dir models_cache
python Benchmark_Framework/run_benchmark.py --adapter hf_local --use-real-validator
```

If a registry entry uses a Hub repo id, the preparation script stores it under:

```text
Benchmark_Framework/models_cache/<namespace>__<repo>
```

`models_cache` is a local cache/preparation directory and should not be
committed.

## Leonardo Preparation Stage

Run this where internet access and storage writes are allowed:

```bash
python Benchmark_Framework/scripts/prepare_models.py \
  --model-registry-path models/model_registry_hf.yaml \
  --models-dir /leonardo_work/YOUR_ACCOUNT/models
```

Then update the selected Hugging Face registry entries so `weights_path` points
to the prepared local model directory.

The repository also provides a SLURM wrapper:

```bash
sbatch Benchmark_Framework/Leonardo_script/prepare_models.sh
```

Environment variables such as `MODEL_ID`, `MODELS_DIR`, `MODEL_REGISTRY_PATH`,
`DRY_RUN`, and `OFFLINE` can be used to narrow or check the preparation job.

## Offline Benchmark Stage

Run benchmark jobs from prepared local paths:

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python Benchmark_Framework/run_benchmark.py \
  --adapter hf_local \
  --use-real-validator \
  --model-registry-path models/model_registry_hf.yaml
```

If a model is missing, fix the preparation stage or the registry path before
launching a GPU job.

## Useful Checks

Prepare one enabled model:

```powershell
python Benchmark_Framework/scripts/prepare_models.py --model-id hf_gemma_4_31b_it
```

Print what would happen without downloading:

```powershell
python Benchmark_Framework/scripts/prepare_models.py --dry-run
```

Verify that required files are already available:

```powershell
python Benchmark_Framework/scripts/prepare_models.py --offline
```

On Leonardo, check prepared model directories with:

```bash
sbatch Benchmark_Framework/Leonardo_script/test_models_cache.sh
```
