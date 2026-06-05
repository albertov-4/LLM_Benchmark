#!/bin/bash
#SBATCH --job-name=bench_qwen_settlersnumeric
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
export MODEL_ID="hf_qwen_3_6_27b"
export TASK_FAMILY="settlersnumeric"
export PROTOCOL_ID="${PROTOCOL_ID:-iterative_repair}"
export OUTPUT_JSON="outputs/scored/${RUN_ID}/suite_results/hf_qwen_3_6_27b__settlersnumeric.json"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${SCRIPT_DIR}/../run_benchmark_single.sh"
