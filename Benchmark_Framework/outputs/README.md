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
- `parsed`: what the parser extracted from model text.
- `scored`: what VAL and the metric layer concluded.
- `logs`: stdout routed from parallel NVIDIA model lanes, when enabled.

## Cleanup

Generated outputs can be listed and removed interactively with:

```powershell
python Benchmark_Framework/clear_outputs.py
```

The cleanup script targets generated children under `raw`, `parsed`, `scored`,
`runs`, and `logs` while preserving the folder structure and `.gitkeep` files.
Review the printed deletion plan before confirming.
