"""Discovery helpers for benchmark suites."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


TIERS = ("easy", "medium", "hard")


@dataclass(slots=True)
class DiscoveredTaskCase:
    task_family: str
    tier: str
    instance_id: str
    domain_file: Path
    problem_file: Path


@dataclass(slots=True)
class SuiteJob:
    model_id: str
    protocol_id: str
    task_case: DiscoveredTaskCase


def discover_task_cases(tasks_root: str | Path) -> list[DiscoveredTaskCase]:
    """Discover all benchmark cases from the task folder hierarchy."""
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


def filter_task_cases(
    task_cases: list[DiscoveredTaskCase],
    task_family: str | None = None,
    tier: str | None = None,
    instance_id: str | None = None,
) -> list[DiscoveredTaskCase]:
    """Filter discovered task cases by optional task selectors."""
    filtered_cases = task_cases
    if task_family is not None:
        filtered_cases = [
            task_case
            for task_case in filtered_cases
            if task_case.task_family == task_family
        ]
    if tier is not None:
        filtered_cases = [
            task_case
            for task_case in filtered_cases
            if task_case.tier == tier
        ]
    if instance_id is not None:
        filtered_cases = [
            task_case
            for task_case in filtered_cases
            if task_case.instance_id == instance_id
        ]
    return filtered_cases


def discover_protocol_ids(protocols_root: str | Path) -> list[str]:
    """Return protocol ids based on yaml filenames."""
    root = Path(protocols_root)
    if not root.exists():
        return []
    return sorted(file_path.stem for file_path in root.glob("*.yaml"))


def build_run_matrix(
    task_cases: list[DiscoveredTaskCase],
    model_ids: list[str],
    protocol_ids: list[str],
) -> list[SuiteJob]:
    """Return the full matrix model x protocol x discovered task case."""
    return [
        SuiteJob(model_id=model_id, protocol_id=protocol_id, task_case=task_case)
        for model_id in model_ids
        for protocol_id in protocol_ids
        for task_case in task_cases
    ]


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


def task_case_key(task_case: DiscoveredTaskCase) -> tuple[str, str, str]:
    """Return a stable key for one discovered task case."""
    return (task_case.task_family, task_case.tier, task_case.instance_id)
