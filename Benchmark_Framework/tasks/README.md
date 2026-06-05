# Tasks

This folder contains the PDDL task families used by the benchmark. The runner
discovers task cases from the directory layout, so folder names and file
placement are part of the benchmark interface.

## Required Layout

```text
tasks/
`-- <task_family>/
    |-- domain/
    |   `-- domain.pddl
    |-- easy/
    |   `-- <instance_id>.pddl
    |-- medium/
    |   `-- <instance_id>.pddl
    `-- hard/
        `-- <instance_id>.pddl
```

Each task family must provide one domain file and at least one problem instance
in one difficulty folder. The current benchmark uses all three tiers.

## Current Inventory

| Task family | Easy | Medium | Hard |
| --- | ---: | ---: | ---: |
| `block-grouping` | 4 | 4 | 4 |
| `expedition` | 4 | 4 | 4 |
| `fo-counters` | 4 | 4 | 4 |
| `fo-sailing` | 4 | 4 | 4 |
| `rover` | 4 | 4 | 4 |
| `settlersnumeric` | 4 | 4 | 4 |

The instance ids are the `.pddl` file stems, such as `pfile1`.

## Discovery Rules

- `tasks/<task_family>/domain/domain.pddl` is used for every instance in the
  family.
- `easy`, `medium`, and `hard` folders are scanned for `*.pddl` files.
- Folders starting with `_` are treated as templates or support folders.
- `metadata/` is optional support data and is not treated as a task family.
- Additional README files inside `domain/`, `easy/`, `medium/`, or `hard/` are
  not needed for runner behavior.

## Adding A Task Family

1. Copy `_template_domain` or create a new folder under `tasks/`.
2. Add `domain/domain.pddl`.
3. Add problem files under `easy`, `medium`, and `hard`.
4. Add `prompts/<task_family>.txt`.
5. Add `prompts/examples/<task_family>.txt` if protocols with examples should
   use examples for the family.
6. Run a preflight check before model jobs:

```powershell
python Benchmark_Framework/run_benchmark.py --task-family <task_family> --preflight-tasks --use-real-validator
```

## Metadata And Complexity Reports

`metadata/` may contain split definitions, notes, or support files for analysis.
The core runner does not depend on it.

Domain complexity summaries are generated separately under
`domains_complexity/` by:

```powershell
python Benchmark_Framework/scripts/score_domains_complexity.py --domains-dir Benchmark_Framework/tasks --output-dir Benchmark_Framework/domains_complexity
```
