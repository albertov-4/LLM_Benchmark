#!/bin/bash
#SBATCH --job-name=benchmark_test
#SBATCH --account=IscrC_VisLLMs
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=benchmark_test_%j.out
#SBATCH --error=benchmark_test_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FRAMEWORK_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

module purge
module load python/3.11.7

for VENV_DIR in "${REPO_ROOT}/project_venv" "${REPO_ROOT}/venv" "${REPO_ROOT}/.venv" "${REPO_ROOT}/.venv-new" "${FRAMEWORK_DIR}/project_venv" "${FRAMEWORK_DIR}/venv"; do
    if [ -f "${VENV_DIR}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        echo "Activated venv: ${VENV_DIR}"
        break
    fi
done

export HF_HOME="${HF_HOME:-${SCRATCH:-${REPO_ROOT}}/hf_cache}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p "${HF_HOME}"

MODEL_ID="${MODEL_ID:-hf_llama_3_1_nemotron_nano_4b_v1_1}"
PROTOCOL_ID="${PROTOCOL_ID:-direct_plan}"
TASK_FAMILY="${TASK_FAMILY:-farmland}"
TIER="${TIER:-easy}"
INSTANCE_ID="${INSTANCE_ID:-pfile1}"
VALIDATOR_COMMAND="${VALIDATOR_COMMAND:-Validate}"
RUN_ID="${RUN_ID:-leonardo_test_${SLURM_JOB_ID:-manual}}"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Repo root: ${REPO_ROOT}"
echo "Framework dir: ${FRAMEWORK_DIR}"
echo "HF_HOME: ${HF_HOME}"
python --version

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    echo "ERROR: nvidia-smi not found."
    exit 1
fi

if ! command -v "${VALIDATOR_COMMAND}" >/dev/null 2>&1; then
    echo "ERROR: VAL validator not found: ${VALIDATOR_COMMAND}"
    echo "Set VALIDATOR_COMMAND=/path/to/Validate or add Validate to PATH."
    exit 1
fi

python -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count()); print('transformers', transformers.__version__)"

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
    --run-id "${RUN_ID}"
)

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"

