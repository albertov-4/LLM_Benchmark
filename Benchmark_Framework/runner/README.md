# Runner

The runner contains the orchestration code for benchmark execution.

## Main Modules

- `run_suite.py`: discovers tasks, loads protocols and model registry entries,
  builds the run matrix, runs jobs, and aggregates suite results.
- `run_case.py`: executes one model on one task instance with one protocol.

The CLI entry point is `Benchmark_Framework/run_benchmark.py`.

## Discovery

Tasks are discovered from the folder structure:

```text
tasks/<task_family>/domain/domain.pddl
tasks/<task_family>/easy/*.pddl
tasks/<task_family>/medium/*.pddl
tasks/<task_family>/hard/*.pddl
```

The runner skips task folders whose names start with `_` and also skips
`tasks/metadata/`. Protocols are discovered from `protocols/*.yaml`. Model
entries are loaded from the selected registry, and disabled entries are ignored
unless a specific `--model-id` selects them through the registry filter.

## Suite Flow

`run_suite.py` builds:

```text
selected models x selected protocols x selected task cases
```

The matrix can be restricted with:

- `--model-id`
- `--protocol-id`
- `--task-family`
- `--tier`
- `--instance-id`
- `--adapter`
- `--model-registry-path`

Progress is printed as `START`, `DONE`, or `ERROR` for each job. The suite
summary stays compact and points to per-job artifacts.

## NVIDIA Model Lanes

The default suite execution is sequential. Passing `--parallel-nvidia-models`
enables parallel lanes only for model entries whose adapter is `nvidia_api`.
Each NVIDIA model gets one lane; inside that lane, the runner processes selected
protocols in order and, for each protocol, task cases in order. This keeps every
single model's benchmark path sequential while allowing different NVIDIA-hosted
LLMs to wait on API calls at the same time.

Use `--max-concurrent-nvidia-models N` to cap how many NVIDIA model lanes run at
once. If the flag is omitted, the runner uses all selected NVIDIA models. Model
entries using Hugging Face, Ollama, llama.cpp, or any other adapter remain in
the normal sequential flow even when NVIDIA lane parallelism is enabled.

When an output root is configured, detailed stdout from each NVIDIA lane is
written to `outputs/logs/<run_id>/<model_id>.log`. The main terminal prints
lane-level completion lines and the final suite summary, while generation,
streaming, parsing, and validation details stay in the per-model log files.

## Single-Case Flow

For each job, `run_case.py`:

1. Reads the PDDL domain and problem.
2. Builds chat-style messages from the selected protocol and prompt files.
3. Calls the model adapter.
4. Parses raw model text into candidate PDDL actions.
5. Validates each non-empty action prefix with VAL or the configured validator.
6. Adds validator feedback for iterative repair protocols.
7. Computes normalized metrics.
8. Writes `raw`, `parsed`, and `scored` artifacts when an output root is set.

Prefix validation records the first valid prefix, whether the full generated
plan is valid, and whether extra actions appeared after the first valid prefix.

## Validation

If `--use-real-validator` is set, the runner builds a VAL-backed validator. The
validator command can be passed explicitly with `--validator-command`; otherwise
the runner tries common `Validate`/`validate` names and local bundled paths.

`--preflight-tasks` validates selected domain/problem pairs before model jobs
start. If preflight fails, the suite returns orchestration errors and does not
call model adapters.

Without a real validator, the runner uses an unavailable-validator fallback that
returns a normalized `validator_unavailable` result. This keeps tests and dry
orchestration paths deterministic.

## Artifacts

For one run id:

```text
outputs/raw/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
outputs/parsed/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
outputs/scored/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
```

`raw` stores prompts and generation payloads. `parsed` stores parser output and
format issues. `scored` stores validation results, repair feedback, metrics,
and artifact paths.
