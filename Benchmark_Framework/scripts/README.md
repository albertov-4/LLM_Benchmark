# Scripts

This folder contains utility scripts that support benchmark operation but are
not part of the per-case execution loop.

## `prepare_models.py`

Prepares enabled Hugging Face model entries so local or HPC benchmark jobs can
load weights from disk instead of downloading during GPU runs.

Common commands:

```powershell
python Benchmark_Framework/scripts/prepare_models.py --models-dir models_cache
python Benchmark_Framework/scripts/prepare_models.py --model-id hf_gemma_4_31b_it
python Benchmark_Framework/scripts/prepare_models.py --dry-run
python Benchmark_Framework/scripts/prepare_models.py --offline
```

Supported options:

- `--model-registry-path`: registry path, defaulting to
  `models/model_registry_hf.yaml` relative to `Benchmark_Framework`.
- `--models-dir`: preparation directory, defaulting to `models_cache` relative
  to `Benchmark_Framework`.
- `--model-id`: prepare one enabled registry entry.
- `--offline`: do not download; fail if required files are missing.
- `--dry-run`: print planned work without downloading.

## `score_domains_complexity.py`

Scores PDDL task-family instances and writes complexity reports.

```powershell
python Benchmark_Framework/scripts/score_domains_complexity.py --domains-dir Benchmark_Framework/tasks --output-dir analysis/domain_complexity
```

Supported options:

- `--domains-dir`: directory containing planning domain folders.
- `--output-dir`: directory where CSV and JSON reports are written.

## Boundaries

Benchmark execution starts from `run_benchmark.py`, not from this folder.
Cleanup is handled by `clear_outputs.py` at the framework root and by
`Leonardo_script/clear_outputs.sh` on SLURM systems.
