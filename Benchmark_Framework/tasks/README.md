# Tasks

This folder contains the benchmark task definitions.

Each task family should contain:
- one folder named after the task family, for example `<task_family>/`
- `domain/domain.pddl`
- three difficulty folders: `easy`, `medium` and `hard`
- `.pddl` instances directly inside each difficulty folder
- one task-family README explaining the domain and the difficulty split

Recommended layout:

```text
tasks/
`-- <task_family>/
    |-- README.md
    |-- domain/
    |   `-- domain.pddl
    |-- easy/
    |   `-- <instance_id>.pddl
    |-- medium/
    |   `-- <instance_id>.pddl
    `-- hard/
        `-- <instance_id>.pddl
```

Task-family README:
- describe the planning domain in general terms
- explain what changes across `easy`, `medium` and `hard`
- document relevant assumptions about objects, predicates, actions and numeric constraints
- avoid placing additional README files inside `domain/`, `easy/`, `medium` or `hard`

Discovery rules:
- the runner discovers task cases from the folder structure
- folders starting with `_` are templates or support folders and are not benchmark task families
- `metadata/` is optional and can contain support files, split definitions or analysis metadata
