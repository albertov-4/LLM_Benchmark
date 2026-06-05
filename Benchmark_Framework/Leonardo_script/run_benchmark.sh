#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FRAMEWORK_DIR}/.." && pwd)"
JOB_DIR="${SCRIPT_DIR}/jobs"

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
LOG_DIR="${FRAMEWORK_DIR}/slurm_logs/${RUN_ID}"

if ! command -v sbatch >/dev/null 2>&1; then
    echo "ERROR: sbatch not found. Run this launcher on Leonardo or a SLURM login node."
    exit 1
fi

if [ ! -d "${JOB_DIR}" ]; then
    echo "ERROR: job script directory not found: ${JOB_DIR}"
    exit 1
fi

mkdir -p "${LOG_DIR}"

cd "${REPO_ROOT}"

shopt -s nullglob
JOB_SCRIPTS=("${JOB_DIR}"/*.sh)
shopt -u nullglob

if [ "${#JOB_SCRIPTS[@]}" -eq 0 ]; then
    echo "ERROR: no job scripts found in ${JOB_DIR}"
    exit 1
fi

echo "Submitting ${#JOB_SCRIPTS[@]} benchmark jobs"
echo "RUN_ID: ${RUN_ID}"
echo "Logs: ${LOG_DIR}"

for JOB_SCRIPT in "${JOB_SCRIPTS[@]}"; do
    sbatch \
        --export=ALL,RUN_ID="${RUN_ID}" \
        --output="${LOG_DIR}/%x_%j.out" \
        --error="${LOG_DIR}/%x_%j.err" \
        "${JOB_SCRIPT}"
done

echo "Submitted benchmark jobs with RUN_ID=${RUN_ID}"
