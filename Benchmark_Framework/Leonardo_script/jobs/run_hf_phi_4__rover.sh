#!/bin/bash
#SBATCH --job-name=bench_phi_rover
#SBATCH --account=try26_varini
#SBATCH --partition=boost_usr_prod
#SBATCH --time=40:00:00
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0

set -euo pipefail

RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
export RUN_ID
export MODEL_ID="hf_phi_4"
export TASK_FAMILY="rover"
export PROTOCOL_ID="${PROTOCOL_ID:-iterative_repair}"
export OUTPUT_JSON="outputs/scored/${RUN_ID}/suite_results/hf_phi_4__rover.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../run_benchmark_single.sh"
