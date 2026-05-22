#!/bin/bash
#SBATCH --job-name=clear_outputs
#SBATCH --account=try26_varini
#SBATCH --partition=boost_usr_prod
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=clear_outputs_%j.out
#SBATCH --error=clear_outputs_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FRAMEWORK_DIR}/.." && pwd)"

cd "${FRAMEWORK_DIR}"

module purge
module load python/3.11.7

VENV_ACTIVATED=0
for VENV_DIR in "${REPO_ROOT}/project_venv" "${REPO_ROOT}/venv" "${REPO_ROOT}/.venv" "${REPO_ROOT}/.venv-new" "${FRAMEWORK_DIR}/project_venv" "${FRAMEWORK_DIR}/venv"; do
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
    echo "Expected one of: project_venv, venv, .venv, .venv-new."
    exit 1
fi

CONFIRM_CLEAR_OUTPUTS="${CONFIRM_CLEAR_OUTPUTS:-0}"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Repo root: ${REPO_ROOT}"
echo "Framework dir: ${FRAMEWORK_DIR}"
python --version

if [ "${CONFIRM_CLEAR_OUTPUTS}" = "1" ]; then
    echo "CONFIRM_CLEAR_OUTPUTS=1: generated outputs will be deleted after clear_outputs.py lists them."
    printf 'y\n' | python clear_outputs.py
else
    echo "Safe mode: listing generated outputs only. No files will be removed."
    echo "To delete, submit with: CONFIRM_CLEAR_OUTPUTS=1 sbatch \"Benchmark Framework/Leonardo script/clear_outputs.sh\""
    python clear_outputs.py < /dev/null
fi

