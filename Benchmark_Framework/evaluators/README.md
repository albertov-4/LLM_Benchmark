# Evaluators

The evaluator layer is shared by all model backends and protocols. Its purpose
is to keep parsing, validation, error categories, and metrics comparable across
runs. Post-run report helpers live in `../reporting/`; this directory is for
benchmark-time parsing, validation, and metric normalization.

## Components

- `parser.py`: extracts domain-valid PDDL actions from raw and reasoning text.
- `validator.py`: defines the validator interface and the VAL adapter.
- `preflight.py`: checks PDDL domain/problem pairs before model generation.
- `metrics.py`: computes normalized per-run metrics.
- `error_taxonomy.py`: provides shared error categories.

## Parser

The parser turns model text into a `ParsedPlan` with two sections:

- `raw`: the official plan extracted from `raw_text`.
- `reasoning`: a diagnostic plan extracted from provider `reasoning_text`.

Each section has `actions` and `format_issues`. `raw` also records
`contains_reasoning` and `source_kind`; `reasoning` records a `source_ref` back
to `generation.reasoning_text` in the raw artifact. The reasoning section is
kept so analysis can compare the plan the model appeared to think through with
the plan it finally wrote; it is not an alternate official answer.

When `domain_text` is available, the parser accepts only parenthesized forms
whose first token is an action declared in the domain and whose arity matches
that action. Predicates, goals, fluent expressions, and unknown actions are not
scored as plan actions. The parser does not check preconditions or simulate
state.

Safe local repeats are expanded when attached to a valid action, including
`repeat N times (action ...)`, `N times (action ...)`, `(repeat N times) (action ...)`,
`(action ...) xN`, `(action ...) *N`, `(action ...) N times`, `(action ...) repeated N times`,
and word counts such as `(action ...) twice`, `(action ...) three times`, or `(action ...) nine times`.
The parser also accepts complete non-parenthesized repeats such as `58 times go_south b0`,
`repeat 58 times go_south b0`, and `go_south b0 repeated 58 times` when the action name and arity
match the domain. Parenthesized local repeat shorthand such as `(repeat up 2 times)` or
`(repeat go_south 59 times)` is accepted only when a previous one-argument action already established
the current object. For one-argument actions, the parser also supports safe domain-derived compression
such as `b4 up 2`, `b2 up x9`, `b2 up nine times`, and progressive numbered compressed lists such as
`1 b4 up 2`.

Reasoning text can contain multiple candidate plans. The parser exposes those candidates to the
runner, including whether a candidate is near a final-answer marker and whether it appears truncated.
When nearby compressed fragments look like parts of the same plan, the parser also creates a composite
candidate while keeping the original smaller candidates. The runner validates candidates individually
and writes the selected diagnostic candidate back to `parsed_plan.reasoning.actions`.

Parser issues:

- `empty_output`: `raw_text` was empty or whitespace only.
- `reasoning_text_empty`: provider `reasoning_text` was empty or missing.
- `markdown_fences_removed`: Markdown code fences were removed before parsing.
- `plan_section_marker_removed`: a plan marker such as `Plan:` or `Final answer:` was removed.
- `reasoning_before_plan_removed`: text before the detected raw plan section was separated from the plan.
- `raw_text_contains_reasoning_like_content`: `raw_text` appears to include reasoning or prose around the actions.
- `actions_embedded_in_text`: a valid action appeared on a line that also contained non-action text.
- `empty_parenthesized_expression_found`: an empty `()` expression was found.
- `unknown_action_names_found`: a parenthesized expression did not start with an action from the domain.
- `wrong_action_arity`: a known domain action had the wrong number of arguments.
- `ambiguous_compressed_action`: a compressed alias was missing or did not map to exactly one one-argument domain action.
- `compressed_actions_expanded`: a repeat or compressed action form was expanded into grounded actions.
- `multiple_reasoning_candidate_plans`: more than one candidate action sequence was found in `reasoning_text`.
- `ambiguous_reasoning_plan_selection`: multiple reasoning candidates tied under the selection heuristic.
- `truncated_reasoning_candidate`: the selected reasoning candidate ended near an incomplete action fragment.
- `no_valid_domain_actions_found`: no domain-valid actions were found when `domain_text` was available.
- `no_parenthesized_actions_found`: no parenthesized actions were found in legacy parsing without `domain_text`.

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
exceptions. Reasoning validation fields use the same normalized shape, but they
are diagnostic inputs for analysis only.

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
models and protocols without re-parsing provider-specific output. Plan length is
computed from validator output first, then from `parsed_plan.raw.actions`; legacy
`parsed_plan.actions` is only a compatibility fallback for older artifacts. CoT
plan alignment reports use existing parsed and scored data and never call the
validator again.
