# SLURM Logs

This folder stores generated stdout and stderr logs from Leonardo/SLURM jobs.
It is separate from benchmark result artifacts under `outputs/`.

## What Writes Here

Scripts under `Leonardo_script/` use SLURM `--output` and `--error` directives
that write log files here. The split benchmark launcher creates per-run log
folders such as:

```text
Benchmark_Framework/slurm_logs/<run_id>/
```

Single utility jobs may write files directly under `slurm_logs/`, for example
cache checks, setup jobs, or small benchmark tests.

## Relationship To Outputs

- `slurm_logs/`: scheduler logs, shell setup diagnostics, environment checks,
  and job-level stderr/stdout.
- `outputs/`: benchmark JSON artifacts, suite summaries, and optional
  per-model runner logs.

Use `slurm_logs/` to debug whether a job started correctly. Use `outputs/` to
analyze benchmark results.

The `.gitkeep` file preserves this directory in Git. Generated log files should
not be treated as source files.
