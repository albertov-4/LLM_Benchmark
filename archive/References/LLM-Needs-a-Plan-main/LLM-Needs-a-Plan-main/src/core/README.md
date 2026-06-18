# Core Module

The `core` module contains the main components of the LLM-Needs-a-Plan framework, providing a complete pipeline for PDDL planning with large language models.

## Architecture Overview

```
core/
├── __init__.py           # Module exports
├── file_manager.py       # PDDL file discovery and organization
├── model_manager.py      # LLM loading and inference management
├── pddl_processor.py     # PDDL processing and plan generation
└── pddl_planner.py       # Main orchestrator class
```

## Classes

### FileManager (`file_manager.py`)

Handles PDDL file discovery, organization, and reading operations.

**Key Features:**
- Automatic domain and problem file discovery
- Support for multiple naming conventions (`domain.pddl`, `{domain}_domain.pddl`)
- Flexible problem file detection (`instance-*.pddl`, `problem_*.pddl`)
- Domain-based directory organization

**Example Usage:**
```python
from core.file_manager import FileManager

fm = FileManager()
domains_data = fm.find_pddl_files("src/data")

for domain_data in domains_data:
    print(f"Domain: {domain_data.domain_name}")
    print(f"Problems: {len(domain_data.problem_paths)}")
```

**Methods:**
- `find_pddl_files(problems_path)` - Discover all PDDL domains and problems
- `read_file(file_path)` - Read file content with error handling
- `save_file(file_path, content)` - Save content to file
- `ensure_directory_exists(directory_path)` - Create directories if needed

### ModelManager (`model_manager.py`)

Manages loading and interaction with large language models.

**Key Features:**
- Automatic model type detection (llama3, phi4, gemma3)
- GPU/CPU device management with multi-GPU support
- Chat template support for all models
- Iterative plan generation with validation
- Response generation with various parameters

**Example Usage:**
```python
from core.model_manager import ModelManager
from prompts.prompts import system_prompt_pddl

# Initialize and load model
mm = ModelManager("src/models/Phi4")
model, tokenizer = mm.load()

# Generate response
messages = [
    {"role": "system", "content": system_prompt_pddl},
    {"role": "user", "content": "Generate a plan for this PDDL problem..."},
]

response = mm.generate_response(
    messages,
    max_tokens=5000,
    sampling=False,
    temperature=0.0,
    top_k=50,
    include_prompt=False,
    skip_special_tokens=True,
)
```

**Methods:**
- `load()` - Load model and tokenizer
- `generate_response(prompt, **kwargs)` - Generate single response
- `iterative_planning_with_validation(domain_path, problem_path, prompt, **kwargs)` - Generate plan with validation
- `get_model_info()` - Get model information and status

### PDDLProcessor (`pddl_processor.py`)

Processes PDDL planning problems and orchestrates plan generation using LLMs.

**Key Features:**
- Domain-problem pair processing
- Domain-specific prompt generation (Tetris, generic)
- Plan validation and iterative refinement
- Batch processing capabilities
- Chain of Thought (CoT) support

**Example Usage:**
```python
from core.pddl_processor import PDDLProcessor

# Initialize with loaded model manager
processor = PDDLProcessor(
    model_manager=mm,
    output_dir="./results"
)

# Process single domain
result = processor.process_domain_with_validation(
    domain_data=domain_info,
    max_iterations=3,
    enable_cot=False
)

# Process multiple domains
batch_results = processor.batch_process_domains(
    domains_data=all_domains,
    max_iterations=3
)
```

**Methods:**
- `process_domain_with_validation(domain_data, **kwargs)` - Process all problems in a domain
- `batch_process_domains(domains_data, **kwargs)` - Process multiple domains
- `get_processor_info()` - Get processor configuration information

### PDDLPlanner (`pddl_planner.py`)

Main orchestrator class that coordinates all components for complete planning workflows.

**Key Features:**
- Component coordination (FileManager, ModelManager, PDDLProcessor)
- Smart model path resolution and auto-selection
- Domain filtering and batch processing
- Comprehensive error handling and progress reporting
- Results tracking and final summaries

**Example Usage:**
```python
from core.pddl_planner import PDDLPlanner

# Initialize with arguments
planner = PDDLPlanner(args, config)

# Setup all components
planner.setup()

# Run planning process
planner.run()

# Get results
results = planner.get_results()
```

**Methods:**
- `setup()` - Initialize and configure all components
- `run()` - Execute the complete planning pipeline
- `get_results()` - Get processing results
- `get_planner_info()` - Get planner configuration information

## Data Flow

```
1. FileManager discovers PDDL files
   ↓
2. PDDLPlanner coordinates components
   ↓
3. ModelManager loads and manages LLM
   ↓
4. PDDLProcessor generates and validates plans
   ↓
5. Results saved and reported
```

## Integration Example

Complete workflow using all core components:

```python
from core import PDDLPlanner
from utils.configuration import load_config

# Load configuration
config = load_config()

# Setup arguments (normally from command line)
class Args:
    problems_path = "src/data"
    weights_path = "src/models/Llama3"
    output_dir = "./results"
    domain = "tetris"  # Process only tetris domain
    max_iterations = 3
    batch = False
    cot = False
    sampling = False
    add_system_prompt = True
    include_prompt = False
    skip_special_tokens = True
    max_tokens = 5000
    temperature = 0.6
    top_k = 10
    model = "auto"
    verbose = False
    log_level = "INFO"
    log_file = None

args = Args()

# Initialize and run planner
planner = PDDLPlanner(args, config)
planner.setup()  # Loads model, discovers files
planner.run()    # Processes domains and generates plans

# Get results
results = planner.get_results()
stats = results.get("overall_stats", {})
total = stats.get("total_problems", 0)
success = stats.get("total_successful", 0)
rate = (success / total) * 100 if total else 0
print(f"Success rate: {rate:.1f}%")
```

## Configuration

The core module integrates with the configuration system:

- `PROBLEMS_PATH` - Path to PDDL domains directory
- `MODEL_PATH` - Path to model weights
- `MODEL_OUTPUT` - Default output directory
- `VAL_PATH` - Path to VAL validator (used by processor)

## Error Handling

All core classes include comprehensive error handling:

- **FileManager**: Handles missing files, permission errors, invalid PDDL
- **ModelManager**: Handles model loading failures, memory issues, generation errors
- **PDDLProcessor**: Handles validation failures, prompt errors, processing timeouts
- **PDDLPlanner**: Coordinates error handling across all components

## Testing

Test files are available in `src/tests/`:
- `test_file_manager.py` - FileManager functionality
- `test_model_manager.py` - Basic ModelManager testing
- `test_model_manager_comprehensive.py` - Real model testing
- `test_pddl_processor.py` - PDDLProcessor integration
- `test_pddl_planner.py` - Complete pipeline testing

## Dependencies

- `torch` - PyTorch for model operations
- `transformers` - Hugging Face transformers library
- `pathlib` - Modern path handling
- `typing` - Type hints for better code clarity

## Performance Notes

- **Model Loading**: Requires significant memory (16GB for Llama3 8B, 32GB for Phi-4 14B, 48GB+ for Gemma3 27B)
- **GPU Acceleration**: Automatic CUDA detection and usage
- **CPU Fallback**: Works on CPU but with slower inference
- **Memory Management**: Efficient handling of large models and batch processing