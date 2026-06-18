# Benchmark Framework

`Benchmark_Framework` evaluates language models on PDDL planning tasks with a
single execution pipeline: build a prompt, generate a plan, parse the model
output, validate the candidate plan with VAL, and save comparable artifacts.

The framework is organized so that task definitions, prompting protocols,
model backends, parsing, validation, metrics, and generated artifacts stay
separate. Analysis notebooks and reports live at repository top level under
`analysis/` so the reusable execution path remains focused.

## At A Glance

The benchmark flow is:

```text
tasks + prompts + protocol + model registry
        |
        v
run_benchmark.py -> runner/run_suite.py -> runner/run_case.py
        |
        v
model adapter -> parser -> VAL validator -> metrics
        |
        v
outputs/raw + outputs/parsed + outputs/scored + optional logs
```

The main user-facing entry point is `run_benchmark.py`. Most runtime behavior
is selected through CLI flags, model registries under `models/`, protocol YAML
files under `protocols/`, and prompt text under `prompts/`.

## What Is Included

```text
Benchmark_Framework/
|-- docs/                 operational notes that do not belong in quickstart docs
|-- evaluators/           parser, VAL adapter, metrics, error taxonomy
|-- Leonardo_script/      SLURM scripts for Leonardo/HPC runs
|-- models/               model registries and backend adapters
|-- outputs/              generated raw, parsed, scored, and log artifacts
|-- prompts/              system, task-family, example, and repair prompts
|-- protocols/            direct and iterative repair protocol YAML files
|-- runner/               suite and single-case orchestration
|-- scripts/              model preparation and complexity scoring utilities
|-- slurm_logs/           generated SLURM stdout/stderr logs
|-- tasks/                PDDL domains and benchmark instances
|-- tests/                unit and integration-style tests
|-- utils/                bundled VAL binaries for Linux and Windows
|-- clear_outputs.py
`-- run_benchmark.py
```

## Current Benchmark Matrix

The repository currently contains six task families:

| Task family | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| `block-grouping` | 4 | 4 | 4 |
| `expedition` | 4 | 4 | 4 |
| `fo-counters` | 4 | 4 | 4 |
| `fo-sailing` | 4 | 4 | 4 |
| `rover` | 4 | 4 | 4 |
| `settlersnumeric` | 4 | 4 | 4 |

Tasks are discovered from:

```text
tasks/<task_family>/domain/domain.pddl
tasks/<task_family>/<easy|medium|hard>/*.pddl
```

Folders starting with `_` and the optional `tasks/metadata/` folder are not
treated as benchmark task families.

## Supported Model Backends

Model definitions live in registry files under `models/`:

- `model_registry_nvidia.yaml`: NVIDIA API through an OpenAI-compatible client.
- `model_registry_hf.yaml`: local Hugging Face Transformers models.
- `model_registry_ollama.yaml`: local Ollama server models.
- `model_registry_llama_cpp.yaml`: local GGUF models through llama.cpp CLI.

Every backend is wrapped by an adapter exposing the same `generate(messages)`
contract. The runner stores adapter output in a normalized shape so metrics and
analysis do not depend on provider-specific response formats.

## Protocols

Protocols live in `protocols/` and control prompting and evaluation behavior:

- `direct_plan`: one attempt, plan-only output.
- `direct_plan_with_rationale`: one attempt, rationale allowed before the final
  extractable plan.
- `iterative_repair`: repeated attempts with validator feedback until a valid
  plan is found or the iteration budget is exhausted.

The final suite matrix is:

```text
selected models x selected protocols x selected task cases
```

## Run Lifecycle

1. `run_benchmark.py` parses CLI flags, resolves the selected model registry,
   builds a run id, and delegates to the suite runner.
2. `runner/run_suite.py` discovers task cases, protocols, and enabled model
   entries, filters the matrix, optionally preflights PDDL files with VAL, and
   starts each job.
3. `runner/run_case.py` builds messages, calls the selected model adapter,
   parses candidate PDDL actions, validates every non-empty action prefix, and
   computes metrics.
4. `evaluators/` normalizes parser, validator, error, and metric payloads so
   all backends can be compared.
5. `outputs/` receives per-case artifacts and suite summaries. Top-level
   analysis notebooks and reports read those artifacts later.

## Setup

Install Python dependencies from the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r Benchmark_Framework/requirements.txt
```

The real validator is VAL. It can be found on `PATH` or passed explicitly with
`--validator-command`, including a path to one of the bundled platform
executables under `utils/`. See [SETUP.md](SETUP.md) for local, API, and HPC
setup details.

## Running Benchmarks

Run the default NVIDIA registry with the real validator:

```powershell
python Benchmark_Framework/run_benchmark.py --use-real-validator
```

Run one protocol:

```powershell
python Benchmark_Framework/run_benchmark.py --protocol-id direct_plan --use-real-validator
```

Run one model:

```powershell
python Benchmark_Framework/run_benchmark.py --model-id nvidia_gemma_4_31b_it --use-real-validator
```

Run selected NVIDIA API models in parallel model lanes:

```powershell
python Benchmark_Framework/run_benchmark.py --adapter nvidia_api --parallel-nvidia-models --use-real-validator
```

Limit concurrent NVIDIA model lanes:

```powershell
python Benchmark_Framework/run_benchmark.py --adapter nvidia_api --parallel-nvidia-models --max-concurrent-nvidia-models 3 --use-real-validator
```

With `--parallel-nvidia-models`, each NVIDIA model runs its selected protocols
and task instances sequentially inside its own lane. Different NVIDIA model
lanes run concurrently. Non-NVIDIA adapters keep the normal sequential runner
behavior. Detailed lane logs are written to:

```text
outputs/logs/<run_id>/<model_id>.log
```

On PowerShell, follow one model lane while the benchmark is running with:

```powershell
Get-Content -Wait Benchmark_Framework/outputs/logs/<run_id>/<model_id>.log
```

Select a backend registry:

```powershell
python Benchmark_Framework/run_benchmark.py --adapter hf_local --protocol-id direct_plan --use-real-validator
```

Restrict the task matrix:

```powershell
python Benchmark_Framework/run_benchmark.py --task-family fo-sailing --tier easy --instance-id pfile1 --use-real-validator
```

Check PDDL domain/problem files before launching model jobs:

```powershell
python Benchmark_Framework/run_benchmark.py --preflight-tasks --use-real-validator
```

Use an explicit registry or run id:

```powershell
python Benchmark_Framework/run_benchmark.py --model-registry-path models/model_registry_ollama.yaml --run-id local_ollama_check
```

## Outputs

Each run uses a timestamped `run_id` unless `--run-id` is provided. Per-case
artifacts are saved under:

```text
outputs/raw/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
outputs/parsed/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
outputs/scored/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
```

The suite summary is saved to:

```text
outputs/scored/<run_id>/suite_result.json
```

Split Leonardo runs can also save one suite summary per submitted model-task
job:

```text
outputs/scored/<run_id>/suite_results/<model_id>__<task_family>.json
```

`raw` stores prompts and generation payloads, `parsed` stores extracted plans
and parser issues, and `scored` stores validation results, repair feedback,
metrics, and artifact paths. During validation the runner checks action
prefixes, records the first valid prefix when present, and distinguishes it from
the validity of the full generated plan.

Generated outputs and NVIDIA lane logs can be removed interactively with:

```powershell
python Benchmark_Framework/clear_outputs.py
```

## Additional Documentation

- [SETUP.md](SETUP.md): environment, validator, model registry, and common run
  setup.
- [../analysis/README.md](../analysis/README.md): notebooks, reports, and
  analysis workflow.
- [docs/README.md](docs/README.md): longer operational notes.
- [../analysis/domain_complexity/README.md](../analysis/domain_complexity/README.md):
  generated complexity reports.
- [docs/model_preparation.md](docs/model_preparation.md): Hugging Face model
  preparation for local and HPC runs.
- [outputs/README.md](outputs/README.md): generated artifact layout and cleanup.
- [protocols/README.md](protocols/README.md): protocol fields and repair flow.
- [scripts/README.md](scripts/README.md): utility script commands and outputs.
- [models/README.md](models/README.md): registry fields and backend settings.
- [runner/README.md](runner/README.md): suite discovery and execution behavior.
- [evaluators/README.md](evaluators/README.md): parser, validator, metrics, and
  error taxonomy.
- [slurm_logs/README.md](slurm_logs/README.md): Leonardo/SLURM log location.
- [tasks/README.md](tasks/README.md): task layout and current task inventory.
- [Leonardo_script/README.md](Leonardo_script/README.md): SLURM workflows for
  model preparation, cache checks, benchmark tests, and full HPC runs.
