# Protocols

Protocols define how a model is prompted and how many attempts are allowed. A
protocol does not select the task or model. The suite matrix is built from:

```text
selected models x selected protocols x selected tasks
```

Run one protocol:

```powershell
python Benchmark_Framework/run_benchmark.py --protocol-id direct_plan --use-real-validator
```

Combine protocol and task filters:

```powershell
python Benchmark_Framework/run_benchmark.py --protocol-id iterative_repair --task-family fo-sailing --tier easy --use-real-validator
```

## Available Protocols

`direct_plan.yaml`

The model receives the domain, problem, and formatting instructions. It should
return only the final plan as parenthesized PDDL actions. This measures
first-attempt validity and format compliance without repair.

`direct_plan_with_rationale.yaml`

The model may include a short rationale before the final action sequence. The
parser still extracts PDDL actions from the final text. This measures whether
rationale helps plan quality and whether the final plan remains extractable.

`iterative_repair.yaml`

The model generates a plan, VAL validates it, and the runner appends validation
feedback to the next attempt when validation fails. This measures whether the
model can correct plans after external feedback and how many iterations are
needed.

## YAML Fields

`protocol_id` is the stable identifier used in filters and output paths.

`description` documents the experimental intent.

`prompting` controls prompt assembly:

- `use_system_prompt`: include `prompts/system.txt`.
- `include_domain_prompt`: require and include `prompts/<task_family>.txt`.
- `include_examples`: append `prompts/examples/<task_family>.txt` when present.
- `include_chain_of_thought`: allow rationale instructions where the protocol
  permits them.
- `include_external_feedback`: enable validator feedback in later attempts.

There is no automatic fallback for missing task-family prompts. If a protocol
requires a domain prompt and `prompts/<task_family>.txt` is missing, the run
fails before querying the model.

`generation` stores common generation settings such as `mode`, `temperature`,
`top_k`, and `max_tokens`. Adapters use supported fields where applicable.

`evaluation` controls the loop:

- `max_iterations`: attempt budget for one task.
- `require_final_plan_only`: whether the final answer should contain only PDDL
  actions.

## Iterative Repair Flow

1. Build the initial prompt from system, task-family, examples, domain, and
   problem text.
2. Generate a candidate plan.
3. Parse domain-valid PDDL actions from `raw_text`; decode `reasoning_text` only for diagnostics.
4. Validate extracted `parsed_plan.raw.actions` prefixes with VAL.
5. Stop if a valid plan is found.
6. Build concise feedback from the validation result.
7. Add the feedback to the next prompt.
8. Repeat until success or `max_iterations`.

The base repair prompt is `prompts/feedback.txt`. Repair feedback is based on
validator output and raw parse issues only; reasoning text is not used to fix or
score a final answer.

## Outputs

Protocol ids are part of every per-case artifact path:

```text
outputs/raw/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
outputs/parsed/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
outputs/scored/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
```

Scored artifacts include `solved`, `iterations_used`, `max_iterations`,
`stopped_by_iteration_limit`, final validation results, metrics, and per-attempt
repair feedback.
