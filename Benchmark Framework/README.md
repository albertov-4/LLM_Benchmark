# Benchmark Framework

This folder contains a framework for comparing multiple LLMs on the same
planning tasks with consistent protocols and metrics.

Goals:
- compare different models on the same task base
- separate tasks, protocols, models and evaluation
- define three difficulty tiers: `easy`, `medium`, `hard`
- save outputs and metrics in comparable formats
- avoid runtime dependency on repositories inside `References/`
- support real runs with NVIDIA API, local Hugging Face, Ollama, llama.cpp and `VAL`

Structure:

```text
Benchmark Framework/
|-- tasks/
|-- protocols/
|-- models/
|-- prompts/
|-- runner/
|-- evaluators/
|-- outputs/
|-- analysis/
`-- config/
```

Design principles:
- tasks are independent from models
- prompting protocols are independent from tasks
- every model uses an adapter with a common interface
- parsing, validation and metrics are shared across all models
- raw outputs and evaluated outputs are separated
- folder hierarchy and naming conventions are the source of truth
- manifests are optional

Recommended structure for each task family:

```text
tasks/
`-- <task_family>/
    |-- README.md
    |-- domain/
    |   `-- domain.pddl
    |-- easy/
    |   |-- instance-01.pddl
    |   `-- instance-02.pddl
    |-- medium/
    |   `-- ...
    `-- hard/
        `-- ...
```

How to use this structure:
1. Create a task family under `tasks/`.
2. Add `domain/domain.pddl`.
3. Put `.pddl` instances inside `easy`, `medium` and `hard`.
4. Register models in `models/model_registry_*.yaml`.
5. Choose a protocol from `protocols/`.
6. Run the benchmark with `run_benchmark.py`.
7. Inspect raw outputs in the timestamped run folder.
8. Inspect parsed outputs and metrics in the matching timestamped folders.

Key folders:
- `tasks/`: benchmark task definitions
- `protocols/`: protocol definitions
- `models/`: model registries and adapters
- `prompts/`: shared prompt files
- `evaluators/`: parser, validator, metrics and error taxonomy
- `runner/`: benchmark orchestration
- `analysis/`: notebooks and final reports

Notes:
- each task family must have a PDDL domain and at least one instance in one tier
- manifests or index files can be added later, but they are not required

Recommended entry points:
- run with the default registry: `python "Benchmark Framework/run_benchmark.py" --use-real-validator`
- clear generated outputs: `python "Benchmark Framework/clear_outputs.py"`
- select a backend registry: `python "Benchmark Framework/run_benchmark.py" --adapter hf_local --protocol-id direct_plan --use-real-validator`
- run one model: `python "Benchmark Framework/run_benchmark.py" --model-id <model_id> --use-real-validator`
- run one protocol: `python "Benchmark Framework/run_benchmark.py" --protocol-id direct_plan --use-real-validator`
- filter one task family, tier or instance: `python "Benchmark Framework/run_benchmark.py" --task-family <task_family> --tier <tier> --instance-id <instance_id> --use-real-validator`
- use a custom registry: `python "Benchmark Framework/run_benchmark.py" --model-registry-path "models/model_registry_ollama.yaml" --model-id <model_id>`
- the launcher creates timestamped folders inside `raw`, `parsed` and `scored`
- the suite summary is saved to `outputs/scored/<timestamp>/suite_result.json`
- execution prints one `START`, `DONE` or `ERROR` line per job
- per-job artifacts are saved at:
  - `outputs/raw/<timestamp>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json`
  - `outputs/parsed/<timestamp>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json`
  - `outputs/scored/<timestamp>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json`
- `raw` stores messages, generation payloads and model text
- `parsed` stores parser output and format issues
- `scored` stores validation results, repair feedback, metrics and artifact paths

Quick setup:
- Python dependencies: [requirements.txt](requirements.txt)
- environment and validator setup: [SETUP.md](SETUP.md)
- protocol explanation: [protocols/README.md](protocols/README.md)
