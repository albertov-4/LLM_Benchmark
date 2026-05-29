# Evaluators

The evaluator layer is shared by all model backends and protocols. Its purpose
is to keep parsing, validation, error categories, and metrics comparable across
runs.

## Components

- `parser.py`: extracts candidate PDDL actions from raw model output.
- `validator.py`: defines the validator interface and the VAL adapter.
- `preflight.py`: checks PDDL domain/problem pairs before model generation.
- `metrics.py`: computes normalized per-run metrics.
- `error_taxonomy.py`: provides shared error categories.

## Parser

The parser turns model text into a `ParsedPlan` with:

- `actions`
- `reasoning`
- `format_issues`

It handles common model-output patterns such as reasoning before the plan,
Markdown fences, numbered lists, bullet lists, and parenthesized actions inside
verbose text. The parser does not decide whether a plan is correct; it only
extracts the candidate action sequence for validation.

Typical parser issues include:

- `empty_output`
- `markdown_fences_removed`
- `reasoning_before_plan_removed`
- `actions_embedded_in_text`
- `no_parenthesized_actions_found`

## Validator

`VALValidatorAdapter` writes a candidate plan to a temporary file, calls the
external VAL executable, collects stdout/stderr/return code, and normalizes the
result.

Main validation fields:

- `valid`
- `status`
- `error_type`
- `feedback_text`
- `failed_step`
- `failed_action`
- `goal_satisfied`
- `plan_length`
- `validation_time_ms`
- `raw_validator_output`
- `details`

Technical failures such as missing executables, timeouts, and validator crashes
are represented as normalized validator results rather than backend-specific
exceptions.

## Prefix Validation

The runner validates every non-empty action prefix. This captures cases where a
model finds a valid plan and then appends extra actions. Scored artifacts record
the first valid prefix, whether the final full action sequence is valid, and
how many extra actions followed the first valid prefix.

## Repair Feedback

`build_feedback_from_validation(...)` and the runner's repair-feedback logic
convert validation results into concise feedback for iterative protocols. The
model is asked to return a complete corrected plan, not just a patch for the
failed step.

## Metrics

Core metrics are computed from normalized run data:

- `validity_at_1`
- `validity_at_k`
- `repair_success`
- `iterations_to_valid`
- `plan_length`
- `error_type`
- `hit_iteration_limit`

Metrics are intentionally separated from raw model text so reports can compare
models and protocols without re-parsing provider-specific output.
