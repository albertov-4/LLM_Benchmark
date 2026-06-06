# Config

This folder stores benchmark-level reference metadata. It is useful for
documenting intended defaults, shared naming, and future configuration
consolidation, but it is not the primary runtime API for the current runner.

## Files

- `benchmark.yaml`: benchmark name, version, task/output defaults, metric names,
  and result-schema path templates.
- `compute.yaml`: reference execution defaults, generation defaults, cache path,
  and log path notes.

## Current Runtime Boundary

Current execution behavior mostly comes from:

- CLI flags passed to `run_benchmark.py`;
- model registries under `models/`;
- protocol YAML files under `protocols/`;
- prompt files under `prompts/`;
- defaults inside `runner/` and `run_benchmark.py`.

Changing `benchmark.yaml` or `compute.yaml` alone should not be assumed to
change a benchmark run. If future code starts reading these files directly,
update this README and the relevant setup/run documentation at the same time.

## Related Documentation

- [../README.md](../README.md): benchmark overview and run lifecycle.
- [../SETUP.md](../SETUP.md): setup and runtime configuration sources.
- [../runner/README.md](../runner/README.md): current suite execution behavior.
