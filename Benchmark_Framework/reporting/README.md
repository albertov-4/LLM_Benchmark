# Reporting Helpers

This package contains reusable post-run report helpers for advanced planning
reports. It is separate from `evaluators/`, which runs during benchmark execution.

## Modules

- `cot_alignment.py`: pure utilities for comparing final raw plans with plans
  extracted from reasoning text, plus semantic-support and proxy scoring.
- `plots.py`: plotting helpers used by `advanced_planning_evaluation_sp.py` when
  report plots are requested.

Keep CLI prompts, artifact loading, report assembly, and JSON writing in
`advanced_planning_evaluation_sp.py` unless another script actually needs them.
