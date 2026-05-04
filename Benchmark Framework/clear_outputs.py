"""Safely clear generated benchmark outputs.

The script keeps the output folder structure and `.gitkeep` files intact, but
removes generated artifacts after an explicit interactive confirmation.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


OUTPUT_SUBDIRECTORIES = ("raw", "parsed", "scored")
KEEP_FILE_NAMES = {".gitkeep"}


def collect_output_targets(outputs_root: str | Path) -> list[Path]:
    """Return generated output files/directories that can be safely removed."""
    root = Path(outputs_root)
    targets: list[Path] = []

    for subdirectory_name in OUTPUT_SUBDIRECTORIES:
        subdirectory = root / subdirectory_name
        if not subdirectory.exists():
            continue
        for child in sorted(subdirectory.iterdir(), key=lambda path: str(path).lower()):
            if child.name in KEEP_FILE_NAMES:
                continue
            targets.append(child)

    return targets


def print_targets(targets: list[Path], framework_root: Path) -> None:
    """Print the deletion plan in a readable form."""
    print("Generated output targets found:")
    for target in targets:
        try:
            display_path = target.relative_to(framework_root)
        except ValueError:
            display_path = target
        marker = "[dir]" if target.is_dir() else "[file]"
        print(f"  {marker} {display_path}")


def delete_targets(targets: list[Path]) -> None:
    """Delete the selected output targets."""
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def main() -> int:
    framework_root = Path(__file__).resolve().parent
    outputs_root = framework_root / "outputs"
    targets = collect_output_targets(outputs_root)

    if not targets:
        print("No generated output files found. Nothing to delete.")
        return 0

    print_targets(targets, framework_root)
    try:
        answer = input("Delete these generated outputs? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    if answer != "y":
        print("Deletion cancelled. No files were removed.")
        return 0

    delete_targets(targets)
    print("Generated outputs deleted. Folder structure and .gitkeep files were preserved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
