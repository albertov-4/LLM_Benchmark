"""Config loading helpers shared by runner scripts."""

from pathlib import Path
from typing import Any

import yaml


def parse_scalar(raw_value: str) -> Any:
    """Parse one scalar value with YAML semantics and legacy `none` support."""
    value = raw_value.strip()
    if not value:
        return ""

    lower_value = value.lower()
    if lower_value in {"null", "none"}:
        return None

    parsed = yaml.safe_load(value)
    if parsed is None and lower_value not in {"null", "~"}:
        return value
    return parsed


def load_model_registry_entries(model_registry_path: str | Path) -> list[dict[str, Any]]:
    """Extract model entries from the `models:` section of a registry file."""
    registry_path = Path(model_registry_path)
    if not registry_path.exists():
        return []

    loaded = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        return []

    models = loaded.get("models", [])
    if not isinstance(models, list):
        return []

    return [
        dict(entry)
        for entry in models
        if isinstance(entry, dict)
    ]
