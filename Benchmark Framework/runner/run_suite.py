"""Suite-level benchmark scaffold."""

from pathlib import Path


def load_task_index(task_index_path: str) -> Path:
    """Resolve the task index path used by suite runs."""
    return Path(task_index_path)


def build_run_matrix() -> list[dict]:
    """Return the run matrix model x protocol x task."""
    return []


def run_suite() -> None:
    """Placeholder entry point for a full benchmark campaign."""
    raise NotImplementedError("Implement suite execution using task index and registry files.")
