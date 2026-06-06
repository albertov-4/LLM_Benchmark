# Domain Complexity Reports

This folder stores generated summaries of PDDL domain and instance complexity.
The benchmark runner does not require these files to execute model runs; they
are analysis inputs for comparing task difficulty and reporting domain
properties.

## Generated Files

- `complexity_scores.csv`: per-instance complexity rows in tabular form.
- `complexity_scores.json`: the same per-instance data as JSON.
- `domain_summary.csv`: per-domain aggregate summaries in tabular form.
- `domain_summary.json`: the same per-domain summaries as JSON.

## Regeneration

From the repository root:

```powershell
python Benchmark_Framework/scripts/score_domains_complexity.py --domains-dir Benchmark_Framework/tasks --output-dir Benchmark_Framework/domains_complexity
```

The script scans task families under `tasks/`, ignores support/template
folders, scores the available PDDL instances, and rewrites the CSV and JSON
reports in this directory.

## Consumers

Analysis notebooks and report sources may read these summaries to correlate
benchmark performance with task complexity. The source of truth for executable
task definitions remains `tasks/<task_family>/domain/domain.pddl` and the
problem files under each difficulty tier.
