#!/bin/bash
#SBATCH --job-name=bench_llama_fo_counters
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
export MODEL_ID="hf_llama_3_3_70b_instruct"
export TASK_FAMILY="fo-counters"
export PROTOCOL_ID="${PROTOCOL_ID:-iterative_repair}"
export OUTPUT_JSON="outputs/scored/${RUN_ID}/suite_results/hf_llama_3_3_70b_instruct__fo-counters.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../run_benchmark_single.sh"
