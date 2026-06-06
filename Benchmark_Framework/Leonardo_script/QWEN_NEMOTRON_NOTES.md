# Qwen and Nemotron Leonardo Notes

These notes record the fixes and diagnostics applied on Leonardo for
`hf_qwen_3_6_27b` and `hf_nemotron_3_nano_30b_a3b`.

Context used during debugging:

- Python venv: `/leonardo_scratch/large/userexternal/avarini0/our_env`
- Python: `3.11.7`
- Torch stack: `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`,
  `torchaudio==2.5.1+cu121`
- CUDA used by Torch: `12.1`
- GPUs: A100, compute capability `8.0`
- `transformers==5.9.0`
- `mamba-ssm==2.2.4`
- `triton==3.1.0`

## Shared Leonardo Runtime Fixes

The benchmark jobs load `gcc/12.2.0` and CUDA 12.1. For native CUDA extensions
such as `causal_conv1d_cuda`, `LD_LIBRARY_PATH` alone was not enough because the
Python process could still load Leonardo's older GCC runtime first.

The worker scripts now force GCC 12's `libstdc++.so.6` with `LD_PRELOAD`:

```bash
GCC_LIBSTDCXX="$("${CXX}" -print-file-name=libstdc++.so.6)"
GCC_LIB_DIR="$(dirname "${GCC_LIBSTDCXX}")"
export LD_LIBRARY_PATH="${GCC_LIB_DIR}:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${GCC_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}"
```

This is needed for errors like:

```text
ImportError: ... libstdc++.so.6: version `GLIBCXX_3.4.29' not found
required by ... causal_conv1d_cuda...
```

The same runtime block is present in:

- `run_benchmark_single.sh`
- `test_benchmark.sh`

## Nemotron

Registry entry:

```yaml
model_id: hf_nemotron_3_nano_30b_a3b
weights_path: unsloth/Nemotron-3-Nano-30B-A3B
model_loader: auto_model
torch_dtype: bfloat16
trust_remote_code: true
```

### Problem 1: Mamba Import with Transformers 5.9

Observed error:

```text
ImportError: cannot import name 'GreedySearchDecoderOnlyOutput'
from 'transformers.generation'
```

Cause:

`mamba-ssm==2.2.4` imports legacy generation output classes that are no longer
exported by `transformers==5.9.0`.

Fix:

`models/adapters/hf_local.py` installs a small compatibility shim before loading
HF models:

```python
GreedySearchDecoderOnlyOutput -> GenerateDecoderOnlyOutput
SampleDecoderOnlyOutput -> GenerateDecoderOnlyOutput
```

This mirrors the upstream Mamba direction for newer Transformers versions.

### Problem 2: `causal_conv1d_cuda` and `GLIBCXX_3.4.29`

Observed error:

```text
ImportError: ... gcc-runtime-8.5.0 ... libstdc++.so.6:
version `GLIBCXX_3.4.29' not found
```

Cause:

The compute job was seeing Leonardo's older GCC runtime before the GCC 12
runtime needed by the compiled extension.

Fix:

Load `gcc/12.2.0` and force its `libstdc++.so.6` with `LD_PRELOAD` in the
SLURM worker scripts.

Expected diagnostic lines in a successful test:

```text
mamba_ssm import ok
causal_conv1d_cuda import ok
GreedySearchDecoderOnlyOutput import ok
```

### Problem 3: `cache_position=None`

After Mamba and `causal_conv1d_cuda` imports were fixed, Nemotron loaded weights
but failed inside the remote model code:

```text
TypeError: 'NoneType' object is not subscriptable
```

Traceback location:

```text
modeling_nemotron_h.py", line 1633, in prepare_inputs_for_generation
    or cache_position[-1] >= input_ids.shape[1]
```

An attempted workaround passed a manual `cache_position`, which moved the error
forward but caused a mask length mismatch:

```text
RuntimeError: The size of tensor a (1828) must match the size of tensor b (1829)
```

The first workaround used `use_cache=False`, which avoided the crash but made
generation extremely slow when Nemotron generated long outputs. In the
`block-grouping` run, many generations reached roughly `raw_chars=14340` and
took around `3700-3900s`.

The current adapter workaround keeps caching enabled and patches Nemotron's
`prepare_inputs_for_generation` at runtime. For Nemotron models only,
`hf_local.py` now initializes `NemotronHHybridDynamicCache` and fills
`cache_position` when Transformers does not pass them correctly.

This should be validated with a single `test_benchmark.sh` run before trusting a
full Nemotron matrix. If it works, expected behavior is:

- no `cache_position None` `TypeError`;
- no mask mismatch such as `1828` vs `1829`;
- much lower generation elapsed time than the `use_cache=False` workaround.

If it still fails, keep the traceback from `run_case.py`; the next likely fix
would be a dedicated Nemotron generation path or a dedicated Nemotron
environment with a Transformers version known to match the remote model
implementation.

## Qwen

Registry entry:

```yaml
model_id: hf_qwen_3_6_27b
weights_path: Qwen/Qwen3.6-27B
model_loader: causal_lm
torch_dtype: bfloat16
trust_remote_code: true
```

### Problem 1: Wrong Loader

Original registry configuration used:

```yaml
model_loader: image_text_to_text
```

That routed Qwen through:

```python
AutoProcessor
AutoModelForImageTextToText
```

For the current Qwen text model, the intended path is text-only causal LM:

```python
AutoTokenizer
AutoModelForCausalLM
```

Fix:

```yaml
model_loader: causal_lm
```

### Problem 2: Missing Local Cache / Case Mismatch

Observed error:

```text
OSError: Can't load the configuration of 'Qwen/Qwen3.6-27B'
```

Cause:

Boost compute nodes do not have internet access, so the model must already be
available in the repository-managed cache. The adapter derives the prepared
cache path from `weights_path`:

```text
Benchmark_Framework/models_cache/Qwen__Qwen3.6-27B
```

The local cache was:

```text
Benchmark_Framework/models_cache/Qwen__Qwen3.6-27b
```

Linux paths are case-sensitive, so `27b` and `27B` are different directories.

Fix:

```bash
mv Benchmark_Framework/models_cache/Qwen__Qwen3.6-27b \
   Benchmark_Framework/models_cache/Qwen__Qwen3.6-27B
```

Verification:

```bash
ls Benchmark_Framework/models_cache/Qwen__Qwen3.6-27B/config.json
```

## Useful Diagnostics

Find which test logs belong to Qwen:

```bash
grep -l "hf_qwen_3_6_27b" Benchmark_Framework/slurm_logs/benchmark_test_*.out
```

Find generation status in a run folder:

```bash
grep -H "\[GEN ERROR\]\|\[GEN DONE\]\|\[.*DONE model=" \
  Benchmark_Framework/slurm_logs/<RUN_ID>/*.out
```

Find non-empty error logs:

```bash
find Benchmark_Framework/slurm_logs/<RUN_ID> -name "*.err" -size +0 -print
```

Run a single diagnostic job:

```bash
MODEL_ID=hf_qwen_3_6_27b \
PROTOCOL_ID=direct_plan \
TASK_FAMILY=fo-sailing \
TIER=easy \
INSTANCE_ID=pfile1 \
sbatch Benchmark_Framework/Leonardo_script/test_benchmark.sh
```

```bash
MODEL_ID=hf_nemotron_3_nano_30b_a3b \
PROTOCOL_ID=direct_plan \
TASK_FAMILY=fo-sailing \
TIER=easy \
INSTANCE_ID=pfile1 \
sbatch Benchmark_Framework/Leonardo_script/test_benchmark.sh
```

For either model, require at least one `[GEN DONE]` before launching the full
24-job benchmark matrix.
