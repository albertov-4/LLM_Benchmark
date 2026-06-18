"""Filesystem helpers for discovering and persisting PDDL artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DomainBundle:
    """Structured representation of a domain plus its problems."""

    domain_name: str
    domain_path: Path
    domain_text: str
    problem_paths: List[Path] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        """Return a dict representation for backward compatibility."""
        return {
            "domain_name": self.domain_name,
            "domain_path": str(self.domain_path),
            "domain_text": self.domain_text,
            "problem_paths": [str(p) for p in self.problem_paths],
        }


class FileManager:
    """Utility class for reading and writing assets on disk."""

    def read_file(self, file_path: str | Path) -> Optional[str]:
        path = Path(file_path)
        if not path.exists():
            logger.warning("File not found: %s", path)
            return None

        try:
            return path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.error("Error reading %s: %s", path, exc)
            return None

    def save_file(self, output_file_path: str | Path, content: str) -> bool:
        path = Path(output_file_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            logger.info("Saved file: %s", path)
            return True
        except OSError as exc:
            logger.error("Unable to write %s: %s", path, exc)
            return False

    def find_pddl_files(self, problems_path: str | Path) -> List[DomainBundle]:
        root = Path(problems_path)
        if not root.exists():
            logger.error("Problems path does not exist: %s", root)
            return []

        bundles: List[DomainBundle] = []
        logger.info("Scanning PDDL domains in %s", root)

        # Check if the root path itself is a domain directory
        if self._locate_domain_file(root):
            logger.info("Root path identified as a single domain directory.")
            bundle = self._build_domain_bundle(root)
            if bundle:
                bundles.append(bundle)
        else:
            # Otherwise, iterate over subdirectories
            for domain_dir in sorted(self._iter_domain_directories(root)):
                bundle = self._build_domain_bundle(domain_dir)
                if bundle:
                    bundles.append(bundle)

        logger.info("Discovered %d domain(s)", len(bundles))
        return bundles

    def ensure_directory_exists(self, directory_path: str | Path) -> bool:
        try:
            Path(directory_path).mkdir(parents=True, exist_ok=True)
            return True
        except OSError as exc:
            logger.error("Cannot create directory %s: %s", directory_path, exc)
            return False

    def list_files(self, directory_path: str | Path, extension: Optional[str] = None) -> List[str]:
        path = Path(directory_path)
        if not path.exists():
            return []

        files = [str(child) for child in path.iterdir() if child.is_file()]
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return sorted(files)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _iter_domain_directories(self, root: Path) -> Iterable[Path]:
        for child in root.iterdir():
            if not child.is_dir() or child.name.startswith("."):
                continue
            if child.name.lower() in {"readme", "docs", "__pycache__"}:
                continue
            yield child

    def _build_domain_bundle(self, domain_dir: Path) -> Optional[DomainBundle]:
        domain_file = self._locate_domain_file(domain_dir)
        if not domain_file:
            logger.warning("Skipping %s: domain file not found", domain_dir)
            return None

        domain_text = self.read_file(domain_file)
        if domain_text is None:
            logger.warning("Skipping %s: unable to read domain file", domain_dir)
            return None

        problem_paths = self._collect_problem_files(domain_dir, domain_file.name)
        if not problem_paths:
            logger.warning("Skipping %s: no problem files detected", domain_dir)
            return None

        logger.debug(
            "Domain %s -> %d problem(s)", domain_dir.name, len(problem_paths)
        )

        return DomainBundle(
            domain_name=domain_dir.name,
            domain_path=domain_file,
            domain_text=domain_text,
            problem_paths=problem_paths,
        )

    def _locate_domain_file(self, domain_dir: Path) -> Optional[Path]:
        candidates = [domain_dir / "domain.pddl", domain_dir / f"{domain_dir.name}_domain.pddl"]
        candidates.extend(domain_dir.glob("*_domain.pddl"))

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _collect_problem_files(self, domain_dir: Path, domain_filename: str) -> List[Path]:
        problems = [
            child
            for child in domain_dir.glob("*.pddl")
            if child.name != domain_filename
        ]
        return sorted(problems)


__all__ = ["FileManager", "DomainBundle"]