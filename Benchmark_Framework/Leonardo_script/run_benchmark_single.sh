#!/bin/bash
#SBATCH --job-name=benchmark_single
#SBATCH --account=try26_spezia
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

# Default offline/cache behaviour for Leonardo compute nodes.
# Models are expected to be already present in Benchmark_Framework/models_cache.
export HF_HOME="${HF_HOME:-${FRAMEWORK_DIR}/models_cache}"
export USE_HUB_KERNELS="${USE_HUB_KERNELS:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}"

MODEL_ID="${MODEL_ID:-}"
PROTOCOL_ID="${PROTOCOL_ID:-iterative_repair}"
TASK_FAMILY="${TASK_FAMILY:-}"
TIER="${TIER:-}"
INSTANCE_ID="${INSTANCE_ID:-}"
DEFAULT_VALIDATOR_COMMAND="${FRAMEWORK_DIR}/utils/linux64/bin/Validate"
VALIDATOR_COMMAND="${VALIDATOR_COMMAND:-${DEFAULT_VALIDATOR_COMMAND}}"
RUN_ID="${RUN_ID:-}"
OUTPUT_JSON="${OUTPUT_JSON:-}"

if [ -f "${DEFAULT_VALIDATOR_COMMAND}" ] && [ ! -x "${DEFAULT_VALIDATOR_COMMAND}" ]; then
    chmod +x "${DEFAULT_VALIDATOR_COMMAND}"
fi

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Repo root: ${REPO_ROOT}"
echo "Framework dir: ${FRAMEWORK_DIR}"
echo "HF_HOME: ${HF_HOME}"
echo "USE_HUB_KERNELS: ${USE_HUB_KERNELS}"
echo "HF_HUB_OFFLINE: ${HF_HUB_OFFLINE}"
echo "TRANSFORMERS_OFFLINE: ${TRANSFORMERS_OFFLINE}"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "CUDACXX: ${CUDACXX}"
echo "CC: ${CC}"
echo "CXX: ${CXX}"
echo "GCC libstdc++: ${GCC_LIBSTDCXX}"
echo "LD_PRELOAD: ${LD_PRELOAD}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
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

print("torch", torch.__version__, "torch_cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
print("transformers", transformers.__version__)
print("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
print("HF_HOME", os.environ.get("HF_HOME", "<unset>"))
print("USE_HUB_KERNELS", os.environ.get("USE_HUB_KERNELS", "<unset>"))
print("HF_HUB_OFFLINE", os.environ.get("HF_HUB_OFFLINE", "<unset>"))
print("TRANSFORMERS_OFFLINE", os.environ.get("TRANSFORMERS_OFFLINE", "<unset>"))

if not torch.cuda.is_available():
    print("ERROR: PyTorch cannot initialize CUDA on this node.", file=sys.stderr)
    print("Your torch build must match Leonardo's CUDA/driver stack. Reinstall torch in our_env with a CUDA 12.1 wheel, for example:", file=sys.stderr)
    print("  pip uninstall -y torch torchvision torchaudio", file=sys.stderr)
    print("  pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio", file=sys.stderr)
    sys.exit(1)
PY

CMD=(
    python "${FRAMEWORK_DIR}/run_benchmark.py"
    --adapter hf_local
    --use-real-validator
    --validator-command "${VALIDATOR_COMMAND}"
    --preflight-tasks
)

if [ -n "${RUN_ID}" ]; then
    CMD+=(--run-id "${RUN_ID}")
fi

if [ -n "${MODEL_ID}" ]; then
    CMD+=(--model-id "${MODEL_ID}")
fi

if [ -n "${PROTOCOL_ID}" ]; then
    CMD+=(--protocol-id "${PROTOCOL_ID}")
fi

if [ -n "${TASK_FAMILY}" ]; then
    CMD+=(--task-family "${TASK_FAMILY}")
fi

if [ -n "${TIER}" ]; then
    CMD+=(--tier "${TIER}")
fi

if [ -n "${INSTANCE_ID}" ]; then
    CMD+=(--instance-id "${INSTANCE_ID}")
fi

if [ -n "${OUTPUT_JSON}" ]; then
    CMD+=(--output-json "${OUTPUT_JSON}")
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"