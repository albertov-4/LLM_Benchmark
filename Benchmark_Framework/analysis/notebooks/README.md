# Analysis Notebooks

This folder contains notebooks for inspecting benchmark outputs and preparing
figures or tables.

Use scored artifacts for quantitative analysis:

```text
Benchmark_Framework/outputs/scored/<run_id>/
```

Use raw artifacts for prompt and model-output inspection:

```text
Benchmark_Framework/outputs/raw/<run_id>/
```

Typical analyses include:

- model comparisons;
- solve rate by protocol;
- performance by difficulty tier;
- degradation from `easy` to `hard`;
- iterative repair behavior;
- final error-type distributions.

Notebooks should treat `suite_result.json` as the suite-level index and follow
artifact paths from there when per-case details are needed.
