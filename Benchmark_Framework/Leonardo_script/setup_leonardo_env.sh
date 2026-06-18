#!/bin/bash
#SBATCH --job-name=setup_leonardo_env
#SBATCH --account=try26_varini
#SBATCH --partition=boost_usr_prod
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --output=Benchmark_Framework/slurm_logs/setup_leonardo_env_%j.out
#SBATCH --error=Benchmark_Framework/slurm_logs/setup_leonardo_env_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRAMEWORK_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${FRAMEWORK_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

module purge
module load python/3.11.7
module load gcc/12.2.0
module load cuda/12.1 2>/dev/null || module load cuda 2>/dev/null || true

VENV_DIR="${PYTHON_VENV:-${REPO_ROOT}/../our_env}"
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "ERROR: venv not found: ${VENV_DIR}"
    echo "Set PYTHON_VENV=/absolute/path/to/venv or create ${REPO_ROOT}/../our_env."
    exit 1
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

VENV_BASENAME="$(basename "${VENV_DIR}")"
if [ -n "${LEONARDO_ENV_PROFILE:-}" ]; then
    ENV_PROFILE="${LEONARDO_ENV_PROFILE}"
elif [ "${VENV_BASENAME}" = "gptoss_env" ]; then
    ENV_PROFILE="gptoss_env"
else
    ENV_PROFILE="our_env"
fi

export CUDA_HOME="${CUDA_HOME:-/leonardo/prod/opt/compilers/cuda/12.1/none}"
export CUDACXX="${CUDACXX:-${CUDA_HOME}/bin/nvcc}"
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CC="${CC:-$(command -v gcc)}"
export CXX="${CXX:-$(command -v g++)}"
export MAX_JOBS="${MAX_JOBS:-1}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"

echo "Venv: ${VENV_DIR}"
echo "Environment profile: ${ENV_PROFILE}"
echo "CUDA_HOME: ${CUDA_HOME}"
echo "CUDACXX: ${CUDACXX}"
echo "CC: ${CC}"
echo "CXX: ${CXX}"
echo "MAX_JOBS: ${MAX_JOBS}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"

python --version
nvcc --version
gcc --version | head -1

case "${ENV_PROFILE}" in
    our_env)
        python -m pip install -r "${FRAMEWORK_DIR}/requirements/leonardo-our-env.txt"

        python -m pip install --force-reinstall \
            torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
            --index-url https://download.pytorch.org/whl/cu121

        python -m pip install --no-cache-dir --no-deps --no-build-isolation mamba-ssm==2.2.4 -v

        python -c "import torchvision, torchaudio; print(torchvision.__version__, torchaudio.__version__)"
        python -c "import mamba_ssm; print('mamba ok')"
        ;;
    gptoss_env)
        python -m pip install -r "${FRAMEWORK_DIR}/requirements/leonardo-gptoss-env.txt"
        ;;
    *)
        echo "ERROR: unknown LEONARDO_ENV_PROFILE=${ENV_PROFILE}"
        echo "Use LEONARDO_ENV_PROFILE=our_env or LEONARDO_ENV_PROFILE=gptoss_env."
        exit 1
        ;;
esac

python -c "import torch; print(torch.__version__, torch.version.cuda)"
python -c "import transformers; print(transformers.__version__)"
python -c "import triton; print(triton.__version__)"
python -m pip check
