# Reports

This folder stores exported benchmark reports and report assets. It may contain
LaTeX sources, generated PDFs, figures, and supporting files.

Current report artifacts are generated from benchmark outputs and domain
complexity summaries. Machine-readable advanced-evaluation JSON reports and
plot folders are generated separately under the repository-level `results/`
folder. Report artifacts are useful for sharing final tables and plots, but
they are not used by the runner.

Recommended organization:

- keep final comparative reports here;
- keep generated figures beside the report that uses them;
- record the `run_id` or input data source inside the report source;
- avoid using this folder as the source of truth for raw benchmark results.

The source of truth for benchmark runs remains:

```text
Benchmark_Framework/outputs/scored/<run_id>/suite_result.json
```
