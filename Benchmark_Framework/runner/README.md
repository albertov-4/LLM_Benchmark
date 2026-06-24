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

Known adapter names are constructed fail-fast. If an `hf_local`, `ollama`,
`nvidia_api`, or `llama_cpp_cli` entry has bad config or missing adapter code,
the runner reports an orchestration error instead of silently substituting an
unavailable adapter. Unknown or empty adapter names still use the
unavailable-adapter placeholder for dry orchestration paths.

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
4. Parses `raw_text` and provider `reasoning_text` into separate plan sections.
5. Validates each non-empty prefix of `parsed_plan.raw.actions` with VAL or the configured validator.
6. Validates decoded reasoning candidates individually, then stores the best diagnostic candidate in `parsed_plan.reasoning.actions`.
7. Adds validator feedback for iterative repair protocols.
8. Computes normalized metrics.
9. Writes `raw`, `parsed`, and `scored` artifacts when an output root is set.

Prefix validation records the first valid prefix, whether the full generated
plan is valid, and whether extra actions appeared after the first valid prefix.
Reasoning-candidate selection validates every parser-provided candidate, including composite candidates
built from nearby compressed reasoning fragments. Selection prefers fully valid, non-truncated,
final-marker-adjacent candidates; this never changes official `solved`, metrics, or raw validation.
The advanced evaluation report may later use these diagnostic reasoning actions
and validation fields to measure whether the model reasoned about the same plan
it wrote.

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

`raw` stores prompts and generation payloads, including adapter-provided
`raw_text` and optional `reasoning_text`. `parsed` stores `parsed_plan.raw` for
the official extracted plan and `parsed_plan.reasoning` for diagnostic reasoning
plan extraction; the reasoning section stores only a `source_ref`, not the full
reasoning text. `scored` stores official raw validation plus diagnostic
`reasoning_validation_result` fields and reasoning-candidate selection metadata
when reasoning candidates were decoded. Metrics, repair, and `solved` still use
only `parsed_plan.raw`. If a decoded reasoning plan validates while raw fails,
repair feedback may include that decoded reasoning action sequence as a hint for
the next final answer. Analysis code can compare raw and reasoning action
sequences, but the runner still treats `parsed_plan.raw` as the only official
answer.
