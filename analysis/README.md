# Analysis

This folder contains the analysis layer for benchmark results. It is separate
from execution: benchmark runs write artifacts under
`Benchmark_Framework/outputs/`, and analysis notebooks, reports, or the advanced
evaluation CLI read those artifacts afterward.

## Contents

```text
analysis/
|-- domain_complexity/   generated task-complexity summaries
|-- notebooks/           exploratory and reporting notebooks
`-- reports/             exported reports, figures, LaTeX sources, and PDFs
```

Generated advanced-evaluation JSON reports and plot folders are written at the
repository root under `results/` by `Benchmark_Framework/advanced_planning_evaluation_sp.py`.
Use `notebooks/` when inspecting raw benchmark behavior, comparing models,
building aggregate tables, or preparing plots interactively. Use `reports/` for
shareable exported documents derived from completed runs.

## Data Sources

The source of truth for run results is:

```text
Benchmark_Framework/outputs/scored/<run_id>/suite_result.json
```

Per-case details are stored beside that summary:

```text
Benchmark_Framework/outputs/raw/<run_id>/...
Benchmark_Framework/outputs/parsed/<run_id>/...
Benchmark_Framework/outputs/scored/<run_id>/...
```

Domain complexity inputs, when needed for plots or tables, come from
`analysis/domain_complexity/`.

## Workflow

1. Run one or more benchmark suites.
2. Inspect results in notebooks under `analysis/notebooks/`, or run the reusable
   advanced evaluation CLI:

```powershell
python Benchmark_Framework/advanced_planning_evaluation_sp.py
```

3. Point the notebook or CLI at the relevant `run_id` or output folder.
4. Generate tables, figures, JSON reports, and optional plots.
5. Save final report artifacts under `analysis/reports/` or generated evaluation
   outputs under `results/`.

Do not treat notebooks, PDFs, plots, or `results/` JSON as the canonical
benchmark data. They are derived views over the JSON artifacts written by the
runner.
