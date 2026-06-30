#!/bin/bash
#SBATCH --job-name=benchmark_test
#SBATCH --partition=boost_usr_prod
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=Benchmark_Framework/slurm_logs/benchmark_test_%j.out
#SBATCH --error=Benchmark_Framework/slurm_logs/benchmark_test_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR:-${PWD}}" && pwd)"

FRAMEWORK_DIR=""
for CANDIDATE_FRAMEWORK_DIR in "${SCRIPT_DIR}/.." "${SUBMIT_DIR}" "${SUBMIT_DIR}/.." "${SUBMIT_DIR}/Benchmark_Framework" "${SUBMIT_DIR}/../Benchmark_Framework"; do
    if [ -f "${CANDIDATE_FRAMEWORK_DIR}/run_benchmark.py" ]; then
        FRAMEWORK_DIR="$(cd "${CANDIDATE_FRAMEWORK_DIR}" && pwd)"
        break
    fi
done

if [ -z "${FRAMEWORK_DIR}" ]; then
    echo "ERROR: cannot locate Benchmark_Framework from script dir or SLURM_SUBMIT_DIR."
    echo "Submit from the repository root or set SLURM_SUBMIT_DIR to the repository path."
    exit 1
fi

REPO_ROOT="$(cd "${FRAMEWORK_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

module purge
module load python/3.11.7
module load gcc/12.2.0
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

export CUDA_HOME="${CUDA_HOME:-/leonardo/prod/opt/compilers/cuda/12.1/none}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
GCC_LIBSTDCXX="$("${CXX}" -print-file-name=libstdc++.so.6)"
GCC_LIB_DIR="$(dirname "${GCC_LIBSTDCXX}")"
export LD_LIBRARY_PATH="${GCC_LIB_DIR}:${LD_LIBRARY_PATH}"
export LD_PRELOAD="${GCC_LIBSTDCXX}${LD_PRELOAD:+:${LD_PRELOAD}}"
export MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

MODEL_ID="${MODEL_ID:-hf_gemma_4_31b_it}"

if [ "${MODEL_ID}" = "hf_gpt_oss_120b" ] && [ -z "${PYTHON_VENV:-}" ]; then
    PYTHON_VENV="${GPTOSS_PYTHON_VENV:-${CINECA_SCRATCH:?Set CINECA_SCRATCH}/gptoss_env}"
fi

VENV_ACTIVATED=0
for CANDIDATE_VENV_DIR in "${PYTHON_VENV:-}" "${VIRTUAL_ENV:-}" "${REPO_ROOT}/../our_env" "${REPO_ROOT}/our_env" "${FRAMEWORK_DIR}/our_env" "${REPO_ROOT}/project_venv" "${REPO_ROOT}/venv" "${REPO_ROOT}/.venv" "${REPO_ROOT}/.venv-new" "${FRAMEWORK_DIR}/project_venv" "${FRAMEWORK_DIR}/venv"; do
    if [ -z "${CANDIDATE_VENV_DIR}" ]; then
        continue
    fi
    VENV_DIR="${CANDIDATE_VENV_DIR}"
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        echo "Activated venv: ${VENV_DIR}"
        VENV_ACTIVATED=1
        break
    fi
done

if [ "${VENV_ACTIVATED}" != "1" ]; then
    echo "ERROR: no Python venv found."
    echo "Set PYTHON_VENV=/absolute/path/to/your/venv, activate a venv before sbatch, or create one of: ../our_env, our_env, project_venv, venv, .venv, .venv-new."
    exit 1
fi

export HF_HOME="${HF_HOME:-${SCRATCH:-${REPO_ROOT}}/hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}"

PROTOCOL_ID="${PROTOCOL_ID:-direct_plan}"
TASK_FAMILY="${TASK_FAMILY:-fo-sailing}"
TIER="${TIER:-easy}"
INSTANCE_ID="${INSTANCE_ID:-pfile1}"
DEFAULT_VALIDATOR_COMMAND="${FRAMEWORK_DIR}/utils/linux64/bin/Validate"
VALIDATOR_COMMAND="${VALIDATOR_COMMAND:-${DEFAULT_VALIDATOR_COMMAND}}"
RUN_ID="${RUN_ID:-}"

if [ -f "${DEFAULT_VALIDATOR_COMMAND}" ] && [ ! -x "${DEFAULT_VALIDATOR_COMMAND}" ]; then
    chmod +x "${DEFAULT_VALIDATOR_COMMAND}"
fi

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Repo root: ${REPO_ROOT}"
echo "Framework dir: ${FRAMEWORK_DIR}"
echo "HF_HOME: ${HF_HOME}"
echo "GCC libstdc++: ${GCC_LIBSTDCXX}"
echo "LD_PRELOAD: ${LD_PRELOAD}"
python --version

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "ERROR: nvidia-smi not found."
    exit 1
fi

if [[ "${VALIDATOR_COMMAND}" == */* ]]; then
    if [ ! -x "${VALIDATOR_COMMAND}" ]; then
        echo "ERROR: VAL validator not found or not executable: ${VALIDATOR_COMMAND}"
        echo "Set VALIDATOR_COMMAND=/path/to/Validate or make ${DEFAULT_VALIDATOR_COMMAND} executable."
        exit 1
    fi
elif ! command -v "${VALIDATOR_COMMAND}" >/dev/null 2>&1; then
    echo "ERROR: VAL validator not found in PATH: ${VALIDATOR_COMMAND}"
    echo "Set VALIDATOR_COMMAND=/path/to/Validate or add Validate to PATH."
    exit 1
fi

python - <<'PY'
import os
import sys

import torch
import transformers

print("torch", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available(), "gpus", torch.cuda.device_count(), flush=True)
print("transformers", transformers.__version__, flush=True)
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"), flush=True)
print("LD_LIBRARY_PATH", os.environ.get("LD_LIBRARY_PATH", "<unset>"), flush=True)
print("LD_PRELOAD", os.environ.get("LD_PRELOAD", "<unset>"), flush=True)

if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot initialize CUDA on this node.", file=sys.stderr)
    print("Your torch build must match Leonardo's CUDA/driver stack. Reinstall torch in our_env with a CUDA 12.1 wheel, for example:", file=sys.stderr)
    print("  pip uninstall -y torch torchvision torchaudio", file=sys.stderr)
    print("  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio", file=sys.stderr)
    sys.exit(1)
PY

if [ "${MODEL_ID}" = "hf_nemotron_3_nano_30b_a3b" ]; then
    python - <<'PY'
import sys
import traceback

import transformers

try:
    print("Installing Transformers generation compatibility shim...", flush=True)
    import transformers.generation as generation_module

    generate_decoder_only_output = getattr(generation_module, "GenerateDecoderOnlyOutput", None)
    if generate_decoder_only_output is None:
        raise RuntimeError("transformers.generation.GenerateDecoderOnlyOutput is not available.")

    for legacy_name in ("GreedySearchDecoderOnlyOutput", "SampleDecoderOnlyOutput"):
        if not hasattr(generation_module, legacy_name):
            setattr(generation_module, legacy_name, generate_decoder_only_output)
            print(f"Patched transformers.generation.{legacy_name} -> GenerateDecoderOnlyOutput", flush=True)

    print("Checking mamba_ssm import...", flush=True)
    import mamba_ssm
    print("mamba_ssm import ok", flush=True)
except Exception as exc:
    print(f"ERROR: mamba_ssm import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

try:
    print("Checking causal_conv1d_cuda import...", flush=True)
    import causal_conv1d_cuda
    print("causal_conv1d_cuda import ok", flush=True)
except Exception as exc:
    print(f"ERROR: causal_conv1d_cuda import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
    traceback.print_exc()
    sys.exit(1)

try:
    print("Checking GreedySearchDecoderOnlyOutput import...", flush=True)
    from transformers.generation import GreedySearchDecoderOnlyOutput
    print("GreedySearchDecoderOnlyOutput import ok from transformers.generation", flush=True)
except Exception as exc:
    print(f"WARNING: GreedySearchDecoderOnlyOutput import failed from transformers.generation: {type(exc).__name__}: {exc}", flush=True)
    try:
        from transformers.generation.utils import GreedySearchDecoderOnlyOutput
        print("GreedySearchDecoderOnlyOutput import ok from transformers.generation.utils", flush=True)
    except Exception as nested_exc:
        print(
            "ERROR: GreedySearchDecoderOnlyOutput import failed from both transformers.generation "
            f"and transformers.generation.utils: {type(nested_exc).__name__}: {nested_exc}",
            file=sys.stderr,
        )
        traceback.print_exc()
        sys.exit(1)
PY
else
    echo "Skipping Nemotron-specific dependency checks for MODEL_ID=${MODEL_ID}"
fi

CMD=(
    python "${FRAMEWORK_DIR}/run_benchmark.py"
    --adapter hf_local
    --model-id "${MODEL_ID}"
    --protocol-id "${PROTOCOL_ID}"
    --task-family "${TASK_FAMILY}"
    --tier "${TIER}"
    --instance-id "${INSTANCE_ID}"
    --use-real-validator
    --validator-command "${VALIDATOR_COMMAND}"
    --preflight-tasks
    --stop-on-error
)

if [ -n "${RUN_ID}" ]; then
    CMD+=(--run-id "${RUN_ID}")
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
