#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --account=try26_spezia
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --output=Benchmark_Framework/slurm_logs/benchmark_test_%j.out
#SBATCH --error=Benchmark_Framework/slurm_logs/benchmark_test_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FRAMEWORK_DIR}/.." && pwd)"
WORKER_SCRIPT="${SCRIPT_DIR}/run_benchmark_single.sh"

DEFAULT_MODEL_IDS=(
    hf_gemma_4_31b_it
    hf_gpt_oss_120b
    hf_phi_4
    hf_qwen_3_6_27b
)
DEFAULT_TASK_FAMILIES=(
    block-grouping
    expedition
    fo-counters
    fo-sailing
    rover
    settlersnumeric
)

if [ -n "${MODEL_IDS:-}" ]; then
    read -r -a SELECTED_MODEL_IDS <<< "${MODEL_IDS}"
else
    SELECTED_MODEL_IDS=("${DEFAULT_MODEL_IDS[@]}")
fi

if [ -n "${TASK_FAMILIES:-}" ]; then
    read -r -a SELECTED_TASK_FAMILIES <<< "${TASK_FAMILIES}"
else
    SELECTED_TASK_FAMILIES=("${DEFAULT_TASK_FAMILIES[@]}")
fi

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
PROTOCOL_ID="${PROTOCOL_ID:-iterative_repair}"
LOG_DIR="${FRAMEWORK_DIR}/slurm_logs/${RUN_ID}"

SLURM_ACCOUNT="${SLURM_ACCOUNT:?Set SLURM_ACCOUNT=<CINECA_PROJECT_ACCOUNT>}"
SLURM_PARTITION="${SLURM_PARTITION:-boost_usr_prod}"
SLURM_TIME="${SLURM_TIME:-1-00:00:00}"
SLURM_GPUS="${SLURM_GPUS:-4}"
SLURM_NODES="${SLURM_NODES:-1}"
SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-1}"
SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-32}"
SLURM_MEM="${SLURM_MEM:-0}"

python_venv_for_model() {
    case "$1" in
        hf_gpt_oss_120b)
            echo "${GPTOSS_PYTHON_VENV:-${CINECA_SCRATCH:?Set CINECA_SCRATCH}/gptoss_env}"
            ;;
        *)
            echo ""
            ;;
    esac
}

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Run this launcher on Leonardo or a SLURM login node."
    exit 1
fi

if [ ! -f "${WORKER_SCRIPT}" ]; then
    echo "ERROR: worker script not found: ${WORKER_SCRIPT}"
    exit 1
fi

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

submitted_jobs=0

echo "Submitting benchmark jobs"
echo "RUN_ID: ${RUN_ID}"
echo "Protocol: ${PROTOCOL_ID}"
echo "Models: ${SELECTED_MODEL_IDS[*]}"
echo "Task families: ${SELECTED_TASK_FAMILIES[*]}"
echo "Logs: ${LOG_DIR}"

for MODEL_ID in "${SELECTED_MODEL_IDS[@]}"; do
    for TASK_FAMILY in "${SELECTED_TASK_FAMILIES[@]}"; do
        MODEL_SLUG="${MODEL_ID#hf_}"
        MODEL_SLUG="${MODEL_SLUG//[^A-Za-z0-9]/_}"
        TASK_SLUG="${TASK_FAMILY//[^A-Za-z0-9]/_}"
        JOB_NAME="bench_${MODEL_SLUG}_${TASK_SLUG}"
        OUTPUT_JSON="outputs/scored/${RUN_ID}/suite_results/${MODEL_ID}__${TASK_FAMILY}.json"
        MODEL_PYTHON_VENV="$(python_venv_for_model "${MODEL_ID}")"
        EXPORTS="ALL,RUN_ID=${RUN_ID},BENCHMARK_REPO_ROOT=${REPO_ROOT},MODEL_ID=${MODEL_ID},TASK_FAMILY=${TASK_FAMILY},PROTOCOL_ID=${PROTOCOL_ID},OUTPUT_JSON=${OUTPUT_JSON}"

        if [ -n "${MODEL_PYTHON_VENV}" ]; then
            EXPORTS="${EXPORTS},PYTHON_VENV=${MODEL_PYTHON_VENV}"
        fi

        sbatch \
            --job-name="${JOB_NAME}" \
            --account="${SLURM_ACCOUNT}" \
            --partition="${SLURM_PARTITION}" \
            --time="${SLURM_TIME}" \
            --gres="gpu:${SLURM_GPUS}" \
            --nodes="${SLURM_NODES}" \
            --ntasks-per-node="${SLURM_NTASKS_PER_NODE}" \
            --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
            --mem="${SLURM_MEM}" \
            --chdir="${REPO_ROOT}" \
            --export="${EXPORTS}" \
            --output="${LOG_DIR}/%x_%j.out" \
            --error="${LOG_DIR}/%x_%j.err" \
            "${WORKER_SCRIPT}"

        submitted_jobs=$((submitted_jobs + 1))
    done
done

echo "Submitted ${submitted_jobs} benchmark jobs with RUN_ID=${RUN_ID}"
