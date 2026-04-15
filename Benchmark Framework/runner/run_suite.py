"""Suite-level benchmark scaffold based on directory discovery.

This file is intentionally written as executable pseudocode:
- some helpers already work
- the orchestration flow is spelled out step by step
- the last wiring points are left as TODOs so the user can adapt them
"""

from dataclasses import dataclass
from pathlib import Path


TIERS = ("easy", "medium", "hard")


@dataclass
class DiscoveredTaskCase:
    task_family: str
    tier: str
    instance_id: str
    domain_file: Path
    problem_file: Path


@dataclass
class SuiteJob:
    model_id: str
    protocol_id: str
    task_case: DiscoveredTaskCase


def discover_task_cases(tasks_root: str | Path) -> list[DiscoveredTaskCase]:
    """Discover all benchmark cases from the task folder hierarchy.

    Expected layout:
        tasks/<task_family>/domain/domain.pddl
        tasks/<task_family>/easy/*.pddl
        tasks/<task_family>/medium/*.pddl
        tasks/<task_family>/hard/*.pddl
    """
    root = Path(tasks_root)
    cases: list[DiscoveredTaskCase] = []

    if not root.exists():
        return cases

    for family_dir in sorted(root.iterdir()):
        if not family_dir.is_dir() or family_dir.name.startswith("_") or family_dir.name == "metadata":
            continue

        domain_file = family_dir / "domain" / "domain.pddl"
        for tier in TIERS:
            tier_dir = family_dir / tier
            if not tier_dir.exists():
                continue
            for problem_file in sorted(tier_dir.glob("*.pddl")):
                cases.append(
                    DiscoveredTaskCase(
                        task_family=family_dir.name,
                        tier=tier,
                        instance_id=problem_file.stem,
                        domain_file=domain_file,
                        problem_file=problem_file,
                    )
                )

    return cases


def discover_protocol_ids(protocols_root: str | Path) -> list[str]:
    """Return protocol ids based on yaml filenames.

    Example:
        protocols/direct_plan.yaml -> direct_plan
    """
    root = Path(protocols_root)
    if not root.exists():
        return []
    return sorted(file_path.stem for file_path in root.glob("*.yaml"))


def load_model_ids_from_registry_pseudocode(model_registry_path: str | Path) -> list[str]:
    """Extract model ids from the registry using a lightweight placeholder parser.

    Pseudocode for a future robust implementation:
    1. Open `model_registry.yaml`
    2. Parse the YAML structure properly
    3. Read `models[*].model_id`
    4. Return the list in registry order

    Current placeholder:
    - scan the file line by line
    - keep values after `model_id:`
    """
    registry_path = Path(model_registry_path)
    if not registry_path.exists():
        return []

    model_ids: list[str] = []
    for line in registry_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("model_id:"):
            model_ids.append(stripped.split(":", 1)[1].strip())
    return model_ids


def build_run_matrix(
    task_cases: list[DiscoveredTaskCase],
    model_ids: list[str],
    protocol_ids: list[str],
) -> list[SuiteJob]:
    """Return the full matrix model x protocol x discovered task case."""
    matrix: list[SuiteJob] = []

    for model_id in model_ids:
        for protocol_id in protocol_ids:
            for task_case in task_cases:
                matrix.append(
                    SuiteJob(
                        model_id=model_id,
                        protocol_id=protocol_id,
                        task_case=task_case,
                    )
                )

    return matrix


def summarize_suite_inputs(
    task_cases: list[DiscoveredTaskCase],
    model_ids: list[str],
    protocol_ids: list[str],
) -> dict[str, int]:
    """Return a small summary useful before launching a full run."""
    return {
        "num_task_cases": len(task_cases),
        "num_models": len(model_ids),
        "num_protocols": len(protocol_ids),
        "num_jobs": len(task_cases) * len(model_ids) * len(protocol_ids),
    }


def run_suite(
    tasks_root: str | Path = "tasks",
    protocols_root: str | Path = "protocols",
    model_registry_path: str | Path = "models/model_registry.yaml",
) -> dict[str, object]:
    """Pseudocode entry point for a full benchmark campaign.

    What this function already does:
    - discover task cases
    - discover protocol ids
    - extract model ids from the registry
    - build the benchmark matrix

    What you are expected to add next:
    - load each protocol yaml into a structured spec
    - instantiate the correct model adapter for each model
    - call `run_case(...)` for each SuiteJob
    - aggregate final statistics
    """
    task_cases = discover_task_cases(tasks_root)
    protocol_ids = discover_protocol_ids(protocols_root)
    model_ids = load_model_ids_from_registry_pseudocode(model_registry_path)
    run_matrix = build_run_matrix(task_cases, model_ids, protocol_ids)

    # PSEUDOCODE:
    # suite_results = []
    # for job in run_matrix:
    #     protocol_spec = load_protocol_spec(job.protocol_id)
    #     model_adapter = build_model_adapter(job.model_id)
    #     result = run_case(job.task_case, protocol_spec, model_adapter, validator)
    #     suite_results.append(result)
    #
    # return aggregate_suite_results(suite_results)

    return {
        "summary": summarize_suite_inputs(task_cases, model_ids, protocol_ids),
        "task_cases": task_cases,
        "protocol_ids": protocol_ids,
        "model_ids": model_ids,
        "run_matrix": run_matrix,
    }
