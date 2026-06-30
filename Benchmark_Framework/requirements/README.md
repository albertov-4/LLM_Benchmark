# Leonardo Requirements

This folder contains curated requirements for the Python environments used on
Leonardo. They are not full `pip freeze` lock files; they keep direct runtime
pins that mattered in the working environments.

Use [../docs/leonardo_setup_from_zero.md](../docs/leonardo_setup_from_zero.md)
for the complete install procedure. In particular, install the CUDA 12.1 PyTorch
stack before the rest of `our_env`, and install `causal-conv1d` separately.

## Environments

| File | Environment | Active models |
| --- | --- | --- |
| `leonardo-our-env.txt` | `our_env` | `hf_gemma_4_31b_it`, `hf_phi_4`, `hf_qwen_3_6_27b` |
| `leonardo-gptoss-env.txt` | `gptoss_env` | `hf_gpt_oss_120b` |

Nemotron local inference is currently disabled in the Hugging Face registry and
is not part of the default active setup.

## Activation

Use the activation helpers created by the Leonardo setup guide:

```bash
source $CINECA_SCRATCH/activate_our_env.sh
source $CINECA_SCRATCH/activate_gptoss_env.sh
```

The `our_env` helper also loads Leonardo modules and applies the `libstdc++`
fix needed by binary imports such as `causal-conv1d`.

## Checks

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import transformers; print(transformers.__version__)"
pip check
```
