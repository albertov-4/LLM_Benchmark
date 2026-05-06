# Evaluators

This folder contains the shared evaluation components used by the benchmark.

Components:
- `parser.py`: converts raw model output into a structured candidate plan
- `validator.py`: defines the validator interface and the `ValidationResult` structure
- `metrics.py`: computes per-run and aggregate metrics
- `error_taxonomy.py`: defines the shared error categories used by parsing, validation and scoring

Guiding principle:
- every model is evaluated through the same parser and validator
- observed differences should come from model behavior, not from different evaluation pipelines

Parser behavior:
- `parser.py` converts model text into a `ParsedPlan` with `actions`, `reasoning` and `format_issues`
- the parser does not decide whether a plan is correct; it only extracts the candidate action sequence to validate
- the parser handles common LLM output patterns such as reasoning before the plan, Markdown fences, numbered lists, bullet lists and PDDL actions embedded in verbose text

Parser flow:
- empty output returns `empty_output`
- markers such as `Plan:` or `Final plan:` are used to separate reasoning from the final plan section
- Markdown code fences are removed without losing their content
- parenthesized actions are extracted even when they appear inside mixed text
- internal action spacing is normalized
- anomalies are recorded in `format_issues`, for example `markdown_fences_removed`, `reasoning_before_plan_removed`, `actions_embedded_in_text` or `no_parenthesized_actions_found`

Validator behavior:
- `validator.py` returns a `ValidationResult`, a normalized object describing the outcome of validating one candidate plan
- the validator evaluates a single attempt, not the full iterative repair loop
- iteration counts belong to the complete run result, not to the validator result
- validation always separates the general status from the specific failure category

Validator components:
- `ValidatorAdapter` defines the interface expected by the runner
- `VALValidatorConfig` describes how to invoke an external validator
- `VALValidatorAdapter` writes the candidate plan to a temporary file, calls the external validator, handles timeouts and crashes, and normalizes the result
- `build_feedback_from_validation(...)` converts a validation result into concise feedback that can be reused by iterative repair

Main `ValidationResult` fields:
- `valid`: boolean validation outcome
- `status`: one of `valid`, `invalid`, `parse_error`, `timeout`, `validator_error`
- `error_type`: more specific failure category, when available
- `feedback_text`: short message that can be passed back to the model during repair
- `failed_step` and `failed_action`: location of the failure in the plan, when available
- `goal_satisfied`: distinguishes executable plans that miss the goal from plans that fail earlier
- `plan_length`, `validation_time_ms`, `raw_validator_output` and `details`: support analysis, debugging and audit trails

Real validator flow:
- the parser output is written as a temporary plan file
- the external validator receives domain, problem and plan paths
- stdout, stderr and return code are collected
- technical errors such as timeout, missing executable or validator crash are converted into consistent `ValidationResult` values
- validator output is mapped into shared categories such as `valid`, `invalid_precondition`, `unsatisfied_goal`, `unknown_action` or other benchmark errors

Error taxonomy:
- `error_taxonomy.py` provides a controlled vocabulary for comparable results across models and tasks
- logical plan errors include `empty_plan`, `syntax_error`, `unknown_action`, `invalid_precondition` and `unsatisfied_goal`
- technical pipeline errors include `parse_error`, `timeout`, `validator_crash` and `validator_unavailable`
- `unknown` is used only when the failure cannot be classified more specifically

Metrics:
- `metrics.py` converts a complete run result into comparable measurements
- metrics are based on normalized run data, not directly on raw model text
- this layer separates raw artifacts from values ready for tables, reports and model comparisons

Core metrics:
- `validity_at_1`: whether the first attempt solved the task
- `validity_at_k`: whether the task was solved within the allowed iteration budget
- `repair_success`: whether repair solved a task that was not solved on the first attempt
- `iterations_to_valid`: number of iterations needed to reach a valid plan
- `plan_length`: length of the final candidate plan
- `error_type`: final error category when the run does not solve the task
- `hit_iteration_limit`: whether the run exceeded the allowed iteration budget
