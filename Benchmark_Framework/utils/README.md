# Utilities

This folder contains bundled VAL binaries used for PDDL parsing, validation,
and related planning utilities.

## Platform Folders

```text
utils/
|-- linux64/bin/
`-- win64/bin/
```

Use the executable that matches the machine running the benchmark:

- Linux: `utils/linux64/bin/Validate`
- Windows: `utils/win64/bin/Validate.exe`

The benchmark primarily needs `Validate`. Other bundled tools come from the VAL
distribution and are kept for inspection or manual PDDL debugging.

## Validator Usage

When `Validate` is on `PATH`, the benchmark can resolve it automatically:

```powershell
python Benchmark_Framework/run_benchmark.py --use-real-validator
```

You can also pass the executable explicitly:

```powershell
python Benchmark_Framework/run_benchmark.py --use-real-validator --validator-command Benchmark_Framework/utils/win64/bin/Validate.exe
```

On Linux:

```bash
python Benchmark_Framework/run_benchmark.py --use-real-validator --validator-command Benchmark_Framework/utils/linux64/bin/Validate
```

## Direct VAL Checks

Validate a domain and problem:

```bash
Validate domain.pddl problem.pddl
```

Validate a plan:

```bash
Validate -v -t 0.001 domain.pddl problem.pddl plan.txt
```

The runner uses VAL through `evaluators/validator.py`, normalizes the result,
and records stdout/stderr in scored artifacts for auditability.
