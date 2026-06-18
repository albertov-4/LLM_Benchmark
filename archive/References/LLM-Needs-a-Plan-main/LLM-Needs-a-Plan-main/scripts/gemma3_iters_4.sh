#!/bin/bash
#SBATCH --account=IscrC_VisLLMs
#SBATCH --partition=boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --job-name=gemma3_iters_4
#SBATCH --output=gemma3_iters_4.out
#SBATCH --error=gemma3_iters_4.err

# ====================================================================
# LLM-Needs-a-Plan Production Experiment
# ====================================================================

echo "=========================================="
echo "LLM PDDL Planning Experiment"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Account: $SLURM_JOB_ACCOUNT"
echo "Partition: $SLURM_JOB_PARTITION"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPUs: $SLURM_GPUS"
echo "Start Time: $(date)"
echo "=========================================="

# Load required modules
echo "Loading Python module..."
module load python/3.11.7

# Navigate to project directory and verify structure
if [ ! -d "src" ] || [ ! -f "config.yml" ]; then
    echo "ERROR: Not in LLM-Needs-a-Plan project directory!"
    echo "Current directory: $(pwd)"
    echo "Available files and directories:"
    ls -la
    exit 1
fi

echo "Working directory: $(pwd)"

# Activate virtual environment
if [ -d "project_venv" ]; then
    VENV_DIR="project_venv"
elif [ -d "venv" ]; then
    VENV_DIR="venv"
else
    echo "ERROR: Virtual environment not found!"
    echo "Looking for 'venv' or 'project_venv' directory"
    exit 1
fi

echo "Activating virtual environment: $VENV_DIR"
source $VENV_DIR/bin/activate

# Verify Python environment
echo "Python executable: $(which python)"
echo "Python version: $(python --version)"

# Set up Python path for imports
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"
export LLM_PROJECT_ROOT="$(pwd)"

# Display GPU information
echo "=========================================="
echo "GPU Information"
echo "=========================================="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
    nvidia-smi
else
    echo "nvidia-smi not available"
fi

# Verify dependencies
echo "=========================================="
echo "Dependency Check"
echo "=========================================="
python -c "
import torch
import transformers
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ====================================================================
# EXPERIMENT CONFIGURATION
# You can modify these parameters for different experiments
# ====================================================================

# Model configuration
MODEL_NAME="gemma3"
WEIGHTS_PATH="src/models/Gemma3"  # weights directory for the chosen model
OUTPUT_DIR="src/results"

# Domain configuration  
PROBLEMS_PATH="src/data"

# Generation parameters
MAX_ITERATIONS=4
MAX_TOKENS=15000
TEMPERATURE=0.1
TOP_K=10

# Logging configuration
LOG_LEVEL="INFO"
LOG_FILE=""  # set to e.g. "logs/gemma3_iters_1.log" to persist logs

# Features to enable
ENABLE_SYSTEM_PROMPT=true
ENABLE_COT=true
ENABLE_SAMPLING=true
INCLUDE_PROMPT=true
SKIP_SPECIAL_TOKENS=true
VERBOSE=true

# Create results directory
mkdir -p src/results/gemma3
echo "Results will be saved to: src/results/gemma3"

echo "=========================================="
echo "Experiment Configuration"
echo "=========================================="
echo "Weights Path: $WEIGHTS_PATH"
echo "Problems Path: $PROBLEMS_PATH"
echo "Max Iterations: $MAX_ITERATIONS"
echo "Max Tokens: $MAX_TOKENS"
echo "Temperature: $TEMPERATURE"
echo "Top-k: $TOP_K"
echo "Chain of Thought: $ENABLE_COT"
echo "Sampling: $ENABLE_SAMPLING"
echo "System Prompt: $ENABLE_SYSTEM_PROMPT"
echo "Include Prompt: $INCLUDE_PROMPT"
echo "Skip Special Tokens: $SKIP_SPECIAL_TOKENS"
echo "Verbose Output: $VERBOSE"
echo "Log Level: $LOG_LEVEL"
echo "Log File: ${LOG_FILE:-none}"
echo "=========================================="

# Build command arguments
MAIN_ARGS="--problems_path $PROBLEMS_PATH --weights_path $WEIGHTS_PATH"
MAIN_ARGS="$MAIN_ARGS --output_dir $OUTPUT_DIR --model $MODEL_NAME"
MAIN_ARGS="$MAIN_ARGS --max_iterations $MAX_ITERATIONS --max_tokens $MAX_TOKENS"
MAIN_ARGS="$MAIN_ARGS --log-level $LOG_LEVEL"

if [ -n "$LOG_FILE" ]; then
    mkdir -p "$(dirname "$LOG_FILE")"
    MAIN_ARGS="$MAIN_ARGS --log-file $LOG_FILE"
fi

if [ "$ENABLE_SYSTEM_PROMPT" = true ]; then
    MAIN_ARGS="$MAIN_ARGS --add_system_prompt"
else
    MAIN_ARGS="$MAIN_ARGS --no-add_system_prompt"
fi

if [ "$ENABLE_COT" = true ]; then
    MAIN_ARGS="$MAIN_ARGS --cot"
else
    MAIN_ARGS="$MAIN_ARGS --no-cot"
fi

if [ "$INCLUDE_PROMPT" = true ]; then
    MAIN_ARGS="$MAIN_ARGS --include_prompt"
else
    MAIN_ARGS="$MAIN_ARGS --no-include_prompt"
fi

if [ "$SKIP_SPECIAL_TOKENS" = true ]; then
    MAIN_ARGS="$MAIN_ARGS --skip_special_tokens"
else
    MAIN_ARGS="$MAIN_ARGS --no-skip_special_tokens"
fi

if [ "$ENABLE_SAMPLING" = true ]; then
    MAIN_ARGS="$MAIN_ARGS --sampling --temperature $TEMPERATURE --top_k $TOP_K"
else
    MAIN_ARGS="$MAIN_ARGS --temperature $TEMPERATURE"
fi

if [ "$VERBOSE" = true ]; then
    MAIN_ARGS="$MAIN_ARGS --verbose"
fi

echo "Full command: python src/main.py $MAIN_ARGS"
echo ""

# ====================================================================
# RUN EXPERIMENT
# ====================================================================

echo "=========================================="
echo "Starting PDDL Planning Experiment"
echo "=========================================="

# Record start time
EXPERIMENT_START=$(date +%s)

# Run the main experiment
python src/main.py $MAIN_ARGS

# Capture exit code
EXPERIMENT_EXIT_CODE=$?

# Record end time
EXPERIMENT_END=$(date +%s)
EXPERIMENT_DURATION=$((EXPERIMENT_END - EXPERIMENT_START))

echo "=========================================="
echo "Experiment Results"
echo "=========================================="
echo "Exit code: $EXPERIMENT_EXIT_CODE"
echo "Duration: ${EXPERIMENT_DURATION}s ($(($EXPERIMENT_DURATION / 60))m $(($EXPERIMENT_DURATION % 60))s)"
echo "End time: $(date)"

# Show generated results
if [ -d "src/results" ]; then
    echo ""
    echo "Generated files:"
    ls -la src/results/ | head -20
    
    # Count generated files
    RESULT_COUNT=$(find src/results -name "*.txt" -o -name "*.json" -o -name "*.pddl" | wc -l)
    echo "Total result files: $RESULT_COUNT"
fi

# Show GPU memory usage at end
echo ""
echo "Final GPU memory usage:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader

echo "=========================================="
echo "Job Summary"
echo "=========================================="

if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
    echo "✓ EXPERIMENT COMPLETED SUCCESSFULLY"
    echo "Check src/results/ for generated plans and analysis"
else
    echo "✗ EXPERIMENT FAILED"
    echo "Check error logs for details"
fi

echo ""
echo "Output files:"
echo "  Main output: experiment_${SLURM_JOB_ID}.out"
echo "  Error log: experiment_${SLURM_JOB_ID}.err"
echo "  Results directory: src/results/"

# Final cleanup and exit
exit $EXPERIMENT_EXIT_CODE