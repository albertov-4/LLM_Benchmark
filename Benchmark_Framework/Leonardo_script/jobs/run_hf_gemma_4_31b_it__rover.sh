#!/bin/bash
#SBATCH --job-name=bench_gemma_rover
#SBATCH --account=try26_varini
#SBATCH --partition=boost_usr_prod
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
export RUN_ID
export MODEL_ID="hf_gemma_4_31b_it"
export TASK_FAMILY="rover"
export PROTOCOL_ID="${PROTOCOL_ID:-iterative_repair}"
export OUTPUT_JSON="outputs/scored/${RUN_ID}/suite_results/hf_gemma_4_31b_it__rover.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../run_benchmark_single.sh"
