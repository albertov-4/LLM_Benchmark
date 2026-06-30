# Docs

This folder contains longer operational notes that are too specific for the
main README or setup quickstart.

## Current Documents

- `leonardo_setup_from_zero.md`: complete from-zero Leonardo setup, including
  access, sparse checkout, `our_env`, `gptoss_env`, Hugging Face downloads, and
  offline cache checks.
- `model_preparation.md`: model-cache notes for local and HPC runs.

## Documentation Boundaries

Use top-level README files for subsystem orientation:

- `../README.md` for the full framework map.
- `../SETUP.md` for environment, validator, and first-run setup.
- `../models/README.md` for registry and adapter configuration.
- `../Leonardo_script/README.md` for SLURM workflow wrappers.

Use this folder for longer procedures that need more detail than a quickstart
section can carry.
