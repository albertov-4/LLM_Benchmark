# Runner

This folder contains the execution logic for the benchmark.

Main files:
- `run_case.py`: runs one model on one task with one protocol
- `run_suite.py`: builds and runs the full benchmark matrix

Responsibilities:
- discover task instances from the folder structure
- load protocols, model registry entries and prompt files
- create the selected model adapter and validator
- execute the generation, parsing, validation and repair loop
- save `raw`, `parsed` and `scored` artifacts

Task discovery:
- tasks are discovered from the directory hierarchy
- the expected layout is:
  - `tasks/<task_family>/domain/domain.pddl`
  - `tasks/<task_family>/easy/*.pddl`
  - `tasks/<task_family>/medium/*.pddl`
  - `tasks/<task_family>/hard/*.pddl`

Single-run flow:
- `run_case.py` reads the domain and problem files
- it builds chat-style messages from the protocol and prompt bundle
- the adapter generates raw model text
- the parser extracts candidate PDDL actions
- the validator checks the candidate plan
- iterative protocols add validator feedback and retry until success or budget exhaustion
- metrics are computed from the normalized final result

Saved artifacts:
- `raw`: messages sent to the model, generation payloads and raw text
- `parsed`: parsed plans and parser-level issues
- `scored`: validation results, repair feedback, metrics and artifact paths
- streaming adapters can mark generation payloads with `partial_output`, `stream_complete`, `stream_error` and `timed_out_by_job_limit`

Suite flow:
- `run_suite.py` builds the model x protocol x task matrix
- progress is printed as `START`, `DONE` or `ERROR`
- `--model-id`, `--protocol-id`, `--task-family`, `--tier` and `--instance-id` restrict the matrix
- `--adapter` selects the matching model registry in the CLI
- `suite_results` stays compact and points to per-job artifact files

Validator selection:
- an explicit `validator_factory` has priority
- otherwise an explicit `validator` instance is used
- otherwise `use_real_validator=True` creates a `VALValidatorAdapter`
- if no real validator is configured, `_UnavailableValidator` returns a normalized validator error
