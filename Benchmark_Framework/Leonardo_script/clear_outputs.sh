#!/bin/bash
#SBATCH --job-name=benchmark_test
#SBATCH --account=try26_varini
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
    echo "To delete, submit with: CONFIRM_CLEAR_OUTPUTS=1 sbatch Benchmark_Framework/Leonardo_script/clear_outputs.sh"
    python clear_outputs.py < /dev/null
fi
