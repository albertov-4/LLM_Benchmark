# Protocols

This folder defines the experimental protocols used by the benchmark.

A protocol describes how the model is queried. It does not select the task or
the model. The final matrix is built from:

```text
selected models x selected protocols x discovered tasks
```

Run one protocol:

```powershell
python "Benchmark Framework/run_benchmark.py" --protocol-id direct_plan --use-real-validator
```

Combine protocol and task filters:

```powershell
python "Benchmark Framework/run_benchmark.py" --protocol-id iterative_repair --task-family <task_family> --tier <tier> --use-real-validator
```

## Available Protocols

`direct_plan.yaml`

The model receives the domain, problem and formatting instructions. It should
produce the final plan directly, without explicit reasoning or repair.

Measures:
- first-attempt validity
- output-format compliance
- planning without external feedback

`direct_plan_with_rationale.yaml`

The model may produce textual reasoning, but the final plan must remain
extractable. The parser extracts PDDL actions from the model output.

Measures:
- whether rationale improves plan quality
- whether rationale stays consistent with final actions
- how much extra text is introduced compared with plan-only output

`iterative_repair.yaml`

The model generates a plan, the validator checks it and, if validation fails,
the runner adds feedback to the next attempt.

Measures:
- whether the model can self-correct after external feedback
- how many iterations are needed to reach a valid plan
- which errors persist after repair

## YAML Fields

`protocol_id`

Stable protocol identifier used by the runner and in the output files.

Example:

```yaml
protocol_id: iterative_repair
```

`description`

Human-readable description. It documents the experimental intent but does not
directly affect execution.

`prompting`

Controls which prompt components are included.

Main fields:
- `use_system_prompt`: includes `prompts/system.txt`
- `include_domain_prompt`: includes `prompts/<task_family>.txt`
- if `include_domain_prompt: true`, `prompts/<task_family>.txt` is required; `prompts/default.txt` is not used as an automatic fallback
- `include_examples`: includes examples when available
- `include_chain_of_thought`: allows rationale instructions for non-plan-only protocols; plan-only protocols still require action-only output
- `include_external_feedback`: enables validator feedback in later attempts

`generation`

Describes generation settings passed to adapters when supported.

Main fields:
- `mode`: descriptive label, such as `deterministic`, `semi_deterministic` or `repair_loop`
- `temperature`: controls sampling variability
- `top_k`: limits sampling to top token candidates when supported
- `max_tokens`: maximum number of generated tokens

Not every adapter uses every field in the same way. The runner passes common
parameters where supported.

`evaluation`

Controls the evaluation loop.

Main fields:
- `max_iterations`: maximum number of attempts per task
- `require_final_plan_only`: when `true`, the final output should contain only PDDL actions

In `direct_plan` and `direct_plan_with_rationale`, `max_iterations` is usually
`1`. In `iterative_repair`, it can be greater than `1`.

`primary_questions`

Experimental questions associated with the protocol. They document what should
be analyzed after the run.

## Iterative Repair Flow

The repair loop works as follows:

1. The runner builds the initial prompt with domain, problem and instructions.
2. The model generates a candidate plan.
3. The parser extracts PDDL actions from the generated text.
4. If no actions are found, a parse error is created.
5. If actions are found, `VAL` validates the plan against the domain and problem.
6. If the plan is valid, the run ends with `solved: true`.
7. If the plan is invalid, the runner creates feedback.
8. The feedback is added to the next prompt.
9. The loop continues until a valid plan is found or `max_iterations` is reached.

The base feedback text lives in:

```text
Benchmark Framework/prompts/feedback.txt
```

## Output And Analysis

Final results include:
- `solved`: whether the task was solved
- `iterations_used`: number of attempts used
- `max_iterations`: protocol iteration budget
- `stopped_by_iteration_limit`: whether the run stopped because the budget was exhausted
- `validation_result`: final validation result
- `metrics`: derived metrics such as `repair_success` and `iterations_to_valid`

Per-job outputs are separated by level:
- `raw`: contains `messages`, `generation`, `raw_output` and `raw_generations`
- `parsed`: contains `parsed_plan` for each attempt
- `scored`: contains `validation_result`, `feedback_to_next_iteration`, final metrics and artifact paths

The `attempts` field exists in all three levels, but with different content.
This avoids duplicating every detail in every file.
