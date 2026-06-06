# Analysis

This folder contains the analysis layer for benchmark results. It is separate
from execution: benchmark runs write artifacts under `outputs/`, and analysis
notebooks or reports read those artifacts afterward.

## Contents

```text
analysis/
|-- notebooks/   exploratory and reporting notebooks
`-- reports/     exported reports, figures, LaTeX sources, and PDFs
```

Use `notebooks/` when inspecting raw benchmark behavior, comparing models,
building aggregate tables, or preparing plots. Use `reports/` for shareable
outputs derived from completed runs.

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
`domains_complexity/`.

## Workflow

1. Run one or more benchmark suites.
2. Open a notebook under `analysis/notebooks/`.
3. Point the notebook at the relevant `run_id` or output folder.
4. Generate tables and figures.
5. Save final report artifacts under `analysis/reports/`.

Do not treat notebooks or reports as the canonical benchmark data. They are
derived views over the JSON artifacts written by the runner.
