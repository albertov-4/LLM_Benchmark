# Analysis Notebooks

This folder contains notebooks for inspecting benchmark outputs and preparing
figures or tables.

## Available Notebooks

- `Results Analysis.ipynb`: baseline inspection of benchmark results.
- `advanced_planning_evaluation.ipynb`: advanced planning metrics using the
  current pooled aggregation style.
- `advanced_planning_evaluation_run_aware.ipynb`: run-aware advanced analysis.
  Use this when `outputs/parsed` and `outputs/scored` contain two or more
  benchmark runs, or when you want to inspect run-to-run variability.

The reusable script version of the advanced workflow is
`Benchmark_Framework/advanced_planning_evaluation_sp.py`. It reads the same
benchmark artifacts and writes model-centric JSON reports and optional plots
under the repository-level `results/` folder.

Use scored artifacts for quantitative analysis:

```text
Benchmark_Framework/outputs/scored/<run_id>/
```

Use raw artifacts for prompt and model-output inspection:

```text
Benchmark_Framework/outputs/raw/<run_id>/
```

The run-aware notebook expects parsed and scored artifacts in matching layouts:

```text
Benchmark_Framework/outputs/parsed/<run_id>/<model>/<protocol>/<domain>/<difficulty>/<instance>.json
Benchmark_Framework/outputs/scored/<run_id>/<model>/<protocol>/<domain>/<difficulty>/<instance>.json
```

It keeps three data levels visible:

- per-case rows in `df_raw` / `df_metrics`;
- per-run summaries in `run_table`;
- fair aggregate tables in `agg_table` and `model_overall`.

For aggregate plots, `advanced_planning_evaluation_run_aware.ipynb` first
computes metrics per run and then averages across runs. This gives each run
equal weight. The older pooled style combines all rows directly, so larger or
more complete runs contribute more heavily to the final mean.

Typical analyses include:

- model comparisons;
- solve rate by protocol;
- performance by difficulty tier;
- degradation from `easy` to `hard`;
- iterative repair behavior;
- final error-type distributions.

Notebooks should treat `suite_result.json` as the suite-level index and follow
artifact paths from there when per-case details are needed.
