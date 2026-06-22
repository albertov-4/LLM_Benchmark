# Outputs

This folder stores generated benchmark artifacts. It is the main handoff point
from execution to analysis and is intentionally kept inside
`Benchmark_Framework/` because it is the runner's default output location.

The JSON files in this tree are generated artifacts, not framework source code.
They are tracked in this repository to preserve the completed benchmark runs
used by the analysis.

## Layout

```text
outputs/
|-- raw/<run_id>/...      prompts, messages, and generation payloads
|-- parsed/<run_id>/...   extracted plans and parser issues
|-- scored/<run_id>/...   validation, repair, metrics, and summaries
`-- logs/<run_id>/...     optional per-model lane logs
```

For one benchmark case, the runner writes matching files under:

```text
outputs/raw/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
outputs/parsed/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
outputs/scored/<run_id>/<model_id>/<protocol_id>/<task_family>/<tier>/<instance_id>.json
```

The suite summary is:

```text
outputs/scored/<run_id>/suite_result.json
```

Split Leonardo jobs can also write one summary per model-task job:

```text
outputs/scored/<run_id>/suite_results/<model_id>__<task_family>.json
```

## What Each Layer Means

- `raw`: what was sent to the model and what the adapter returned.
- `parsed`: `parsed_plan.raw` official actions plus `parsed_plan.reasoning` diagnostic actions.
- `scored`: what VAL and the metric layer concluded from `parsed_plan.raw.actions`.
- `logs`: stdout routed from parallel NVIDIA model lanes, when enabled.

## Parsed Plan Schema

Parsed artifacts store attempts as:

```json
"parsed_plan": {
  "raw": {
    "actions": [],
    "format_issues": [],
    "contains_reasoning": false,
    "source_kind": "clean_raw_plan"
  },
  "reasoning": {
    "actions": [],
    "format_issues": [],
    "source_ref": {"artifact": "raw", "field": "generation.reasoning_text"}
  }
}
```

Only `raw.actions` is official for validation, repair, metrics, and `solved`.
`reasoning.actions` is validated in scored artifacts when present, but those
`reasoning_validation_result` fields are diagnostic and never repair or replace
the final answer.

## Cleanup

Generated outputs can be listed and removed interactively with:

```powershell
python Benchmark_Framework/clear_outputs.py
```

The cleanup script targets generated children under `raw`, `parsed`, `scored`,
`runs`, and `logs` while preserving the folder structure and `.gitkeep` files.
Review the printed deletion plan before confirming.
