# LLM_Benchmark

This repository contains a benchmark for evaluating language models on PDDL
planning tasks. The active execution framework builds prompts, calls model
backends, parses candidate plans, validates them with VAL, and writes comparable
raw, parsed, and scored artifacts for later analysis.

## Repository Layout

```text
LLM_Benchmark/
|-- Benchmark_Framework/   benchmark runner, tasks, adapters, prompts, tests
|-- analysis/              notebooks, reports, and domain-complexity summaries
|-- results/               generated advanced-evaluation JSON and plots
`-- archive/               papers, reference repos, and historical material
```

`Benchmark_Framework/` is the runnable benchmark. `analysis/` reads completed
benchmark outputs and explains the analysis workflow. `results/` stores generated
advanced-evaluation JSON reports and plot folders. `archive/` is kept for
background material and is not part of the execution path.

## Quick Start

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r Benchmark_Framework/requirements/leonardo-our-env.txt
```

Run a benchmark with the real VAL validator:

```powershell
python Benchmark_Framework/run_benchmark.py --use-real-validator
```

Run a narrower smoke-style benchmark:

```powershell
python Benchmark_Framework/run_benchmark.py --protocol-id direct_plan --task-family fo-sailing --tier easy --instance-id pfile1 --use-real-validator
```

Run the test suite:

```powershell
python -m unittest discover -s Benchmark_Framework/tests -p "test_*.py"
```

## Documentation

- [Benchmark Framework](Benchmark_Framework/README.md): execution flow, task
  matrix, adapters, protocols, outputs, and subsystem links.
- [Setup](Benchmark_Framework/SETUP.md): local, API, VAL, and Leonardo/HPC
  setup details.
- [Analysis](analysis/README.md): how notebooks and reports consume benchmark
  artifacts.
- [Archive](archive/README.md): reference repositories, papers, and historical
  planning-domain material.
