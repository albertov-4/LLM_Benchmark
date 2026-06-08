# Docs

This folder contains longer operational notes that are too specific for the
main README or setup quickstart.

## Current Documents

- `model_preparation.md`: how to prepare Hugging Face model weights for local
  and HPC runs, including offline execution expectations.

## Documentation Boundaries

Use top-level README files for subsystem orientation:

- `../README.md` for the full framework map.
- `../SETUP.md` for environment, validator, and first-run setup.
- `../models/README.md` for registry and adapter configuration.
- `../Leonardo_script/README.md` for SLURM workflows.

Use this folder for longer procedures that need more detail than a quickstart
section can carry.
