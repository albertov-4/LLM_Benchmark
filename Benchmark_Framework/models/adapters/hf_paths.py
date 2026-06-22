"""Shared Hugging Face repo/cache path helpers."""

from __future__ import annotations

from pathlib import Path
import re


HF_REPO_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)


def looks_like_hf_repo_id(value: str) -> bool:
    """Return whether a value has the shape of a Hugging Face repo id."""
    text = value.strip()
    if not text or "\\" in text or text.startswith(("/", ".")):
        return False
    if ":" in text or " " in text:
        return False
    return HF_REPO_ID_PATTERN.match(text) is not None


def candidate_local_paths(
    path_text: str,
    framework_root: Path,
    extra_base: Path | None = None,
) -> list[Path]:
    """Return plausible local paths for a registry path value."""
    raw_path = Path(path_text).expanduser()
    if raw_path.is_absolute():
        return [raw_path]

    candidates = [Path.cwd() / raw_path]
    if extra_base is not None:
        candidates.append(extra_base / raw_path)
    candidates.append(framework_root / raw_path)
    return candidates


def existing_local_model_path(
    path_text: str,
    framework_root: Path,
    extra_base: Path | None = None,
) -> Path | None:
    """Return an existing local model path for a registry path, if any."""
    if not path_text.strip():
        return None

    for candidate in candidate_local_paths(path_text, framework_root, extra_base):
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def local_dir_for_hf_repo(models_dir: Path, repo_id: str) -> Path:
    """Return the cache directory used for one Hugging Face repo id."""
    return models_dir / repo_id.replace("/", "__")


def is_prepared_model_dir(path_value: Path) -> bool:
    """Return whether a prepared model cache directory is populated."""
    return path_value.exists() and path_value.is_dir() and any(path_value.iterdir())
