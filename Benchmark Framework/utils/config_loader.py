"""Small config loading helpers shared by runner scripts."""

from pathlib import Path
from typing import Any


def parse_scalar(raw_value: str) -> Any:
    """Parse simple scalar values used in the framework YAML files."""
    value = raw_value.strip()
    if not value:
        return ""

    lower_value = value.lower()
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False
    if lower_value in {"null", "none"}:
        return None

    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        return value[1:-1]

    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_model_registry_entries(model_registry_path: str | Path) -> list[dict[str, Any]]:
    """Extract model entries from the `models:` section of a registry file."""
    registry_path = Path(model_registry_path)
    if not registry_path.exists():
        return []

    model_entries: list[dict[str, Any]] = []
    current_entry: dict[str, Any] | None = None
    in_models_section = False

    for line in registry_path.read_text(encoding="utf-8").splitlines():
        raw_line = line.rstrip()
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        if indent == 0:
            if current_entry is not None:
                model_entries.append(current_entry)
                current_entry = None
            in_models_section = stripped == "models:"
            continue

        if not in_models_section:
            continue

        if indent == 2 and stripped.startswith("- "):
            if current_entry is not None:
                model_entries.append(current_entry)
            current_entry = {}
            item_content = stripped[2:]
            if ":" in item_content:
                key, _, raw_value = item_content.partition(":")
                current_entry[key.strip()] = parse_scalar(raw_value)
            continue

        if current_entry is not None and indent >= 4 and ":" in stripped and not stripped.startswith("- "):
            key, _, raw_value = stripped.partition(":")
            current_entry[key.strip()] = parse_scalar(raw_value)

    if current_entry is not None:
        model_entries.append(current_entry)

    return model_entries
