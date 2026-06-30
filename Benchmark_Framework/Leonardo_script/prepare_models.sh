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

cd "${FRAMEWORK_DIR}"

module purge
module load python/3.11.7

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

export HF_HOME="${HF_HOME:-${FRAMEWORK_DIR}/models_cache/.hf_home}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
mkdir -p "${HF_HOME}"
mkdir -p "${FRAMEWORK_DIR}/models_cache"

MODEL_REGISTRY_PATH="${MODEL_REGISTRY_PATH:-models/model_registry_hf.yaml}"
MODELS_DIR="${MODELS_DIR:-models_cache}"
MODEL_ID="${MODEL_ID:-hf_gpt_oss_120b}"
DRY_RUN="${DRY_RUN:-0}"
OFFLINE="${OFFLINE:-0}"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Repo root: ${REPO_ROOT}"
echo "Framework dir: ${FRAMEWORK_DIR}"
echo "HF_HOME: ${HF_HOME}"
echo "Model registry: ${MODEL_REGISTRY_PATH}"
echo "Models dir: ${MODELS_DIR}"
python --version

if [ -z "${HF_TOKEN:-}" ]; then
    echo "WARNING: HF_TOKEN is not set. Public models may still download, but rate limits can be lower and gated models will fail."
else
    echo "HF_TOKEN is set."
fi

python -c "import huggingface_hub; print('huggingface_hub', huggingface_hub.__version__)"

CMD=(
    python scripts/prepare_models.py
    --model-registry-path "${MODEL_REGISTRY_PATH}"
    --models-dir "${MODELS_DIR}"
)

if [ -n "${MODEL_ID}" ]; then
    CMD+=(--model-id "${MODEL_ID}")
fi

if [ "${DRY_RUN}" = "1" ]; then
    CMD+=(--dry-run)
fi

if [ "${OFFLINE}" = "1" ]; then
    CMD+=(--offline)
fi

printf 'Running command:'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}"
