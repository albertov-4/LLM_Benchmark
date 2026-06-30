# Model Preparation

The benchmark can run on a local PC or on Leonardo. On Leonardo, the working
model cache is the framework cache:

```bash
$CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework/models_cache
```

Do not create unrelated model folders outside the framework for the default
workflow. See [leonardo_setup_from_zero.md](leonardo_setup_from_zero.md) for the
complete from-zero download procedure.

## Hugging Face Repo Ids And Local Paths

A `weights_path` such as `Qwen/Qwen3.6-27B` is a Hugging Face repo id. If files
are not already prepared locally, Transformers or `huggingface_hub` may contact
the Hub.

A prepared cache directory such as:

```bash
$CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework/models_cache/Qwen__Qwen3.6-27B
```

is local data. The benchmark can load it without internet access.

## Leonardo Download Rule

Simple Hugging Face downloads and offline cache checks do not require GPU or
Booster. Run them on the login node and write into `Benchmark_Framework/models_cache`.
Use GPU/Booster for actual local model inference and benchmark jobs.

## Token Handling

Do not write Hugging Face tokens into repository files. Use a shell variable:

```bash
read -s HF_TOKEN
export HF_TOKEN
hf auth whoami --token "$HF_TOKEN"
```

Pass the token explicitly to download commands with `--token "$HF_TOKEN"`.

## Offline Cache Check

From the framework directory:

```bash
cd $CINECA_SCRATCH/LLM_Benchmark/Benchmark_Framework
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
bash Leonardo_script/test_models_cache.sh
```

If a download stops, rerun the same `hf download` command. Do not delete the
whole `models_cache` directory.
