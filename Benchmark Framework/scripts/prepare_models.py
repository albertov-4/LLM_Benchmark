"""Prepare benchmark models before running local or HPC jobs.

The benchmark runner should not download large model weights during an HPC GPU
job. This script provides a separate preparation step for enabled registry
models, starting with `hf_local` entries.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any


HF_REPO_ID_PATTERN = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)


@dataclass(slots=True)
class ModelRegistryEntry:
    model_id: str
    adapter: str
    provider: str
    enabled: bool
    weights_path: str
    raw_entry: dict[str, Any]


@dataclass(slots=True)
class PreparedModel:
    model_id: str
    adapter: str
    source: str
    local_path: str | None
    status: str
    message: str
    success: bool


class MissingHuggingFaceHubError(RuntimeError):
    """Raised when `huggingface_hub` is required but not installed."""


def _framework_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_config_loader():
    module_path = _framework_root() / "utils" / "config_loader.py"
    spec = spec_from_file_location("benchmark_framework_prepare_models_config_loader", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_framework_path(path_value: str | Path, framework_root: Path) -> Path:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return framework_root / candidate


def load_model_registry(model_registry_path: str | Path) -> list[ModelRegistryEntry]:
    registry_path = Path(model_registry_path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    model_entries = _load_config_loader().load_model_registry_entries(registry_path)

    registry_entries: list[ModelRegistryEntry] = []
    for raw_entry in model_entries:
        registry_entries.append(
            ModelRegistryEntry(
                model_id=str(raw_entry.get("model_id", "")),
                adapter=str(raw_entry.get("adapter", "")),
                provider=str(raw_entry.get("provider", "")),
                enabled=bool(raw_entry.get("enabled", True)),
                weights_path=str(raw_entry.get("weights_path", "") or ""),
                raw_entry=raw_entry,
            )
        )

    return registry_entries


def looks_like_hf_repo_id(value: str) -> bool:
    text = value.strip()
    if not text or "\\" in text or text.startswith(("/", ".")):
        return False
    if ":" in text or " " in text:
        return False
    return HF_REPO_ID_PATTERN.match(text) is not None


def _candidate_local_paths(path_text: str, registry_path: Path, framework_root: Path) -> list[Path]:
    raw_path = Path(path_text).expanduser()
    if raw_path.is_absolute():
        return [raw_path]

    return [
        Path.cwd() / raw_path,
        registry_path.parent / raw_path,
        framework_root / raw_path,
    ]


def is_existing_local_model(path_text: str, registry_path: Path, framework_root: Path) -> Path | None:
    if not path_text.strip():
        return None

    for candidate in _candidate_local_paths(path_text, registry_path, framework_root):
        if candidate.exists() and candidate.is_dir():
            return candidate.resolve()
    return None


def local_dir_for_hf_repo(models_dir: Path, repo_id: str) -> Path:
    safe_name = repo_id.replace("/", "__")
    return models_dir / safe_name


def is_prepared_model_dir(path_value: Path) -> bool:
    return path_value.exists() and path_value.is_dir() and any(path_value.iterdir())


def download_hf_snapshot(repo_id: str, local_dir: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise MissingHuggingFaceHubError(
            "Please install huggingface_hub to download models."
        ) from exc

    downloaded_path = snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
    )
    return Path(downloaded_path)


def skip_non_hf_model(entry: ModelRegistryEntry) -> PreparedModel:
    provider = entry.provider.strip()
    adapter = entry.adapter.strip()

    if adapter == "ollama" or provider == "ollama_local":
        message = f"Skipping {provider or adapter} model {entry.model_id}: preparation is managed by Ollama."
    elif adapter == "nvidia_api" or provider == "nvidia_api":
        message = f"Skipping {provider or adapter} model {entry.model_id}: remote API model, no local download needed."
    elif adapter == "openai_compatible" or provider == "openai_compatible":
        message = f"Skipping {provider or adapter} model {entry.model_id}: server-managed model."
    else:
        message = f"Skipping {provider or adapter} model {entry.model_id}: preparation is not managed by this script."

    return PreparedModel(
        model_id=entry.model_id,
        adapter=entry.adapter,
        source="",
        local_path=None,
        status="skipped",
        message=message,
        success=True,
    )


def resolve_hf_model_source(entry: ModelRegistryEntry) -> str:
    weights_path = entry.weights_path.strip()
    if weights_path:
        return weights_path
    if looks_like_hf_repo_id(entry.model_id):
        return entry.model_id
    return ""


def prepare_hf_model(
    entry: ModelRegistryEntry,
    models_dir: Path,
    registry_path: Path,
    framework_root: Path,
    offline: bool = False,
    dry_run: bool = False,
) -> PreparedModel:
    source = resolve_hf_model_source(entry)
    if not source:
        return PreparedModel(
            model_id=entry.model_id,
            adapter=entry.adapter,
            source="",
            local_path=None,
            status="warning",
            message=(
                f"Model {entry.model_id}: no weights_path was provided and model_id "
                "does not look like a Hugging Face repo id."
            ),
            success=False,
        )

    existing_local_path = is_existing_local_model(source, registry_path, framework_root)
    if existing_local_path is not None:
        return PreparedModel(
            model_id=entry.model_id,
            adapter=entry.adapter,
            source=source,
            local_path=str(existing_local_path),
            status="available",
            message=f"Model {entry.model_id}: already available at {existing_local_path}",
            success=True,
        )

    if not looks_like_hf_repo_id(source):
        return PreparedModel(
            model_id=entry.model_id,
            adapter=entry.adapter,
            source=source,
            local_path=None,
            status="missing_local_path",
            message=(
                f"Model {entry.model_id}: {source} is not an existing local directory "
                "and does not look like a Hugging Face repo id."
            ),
            success=False,
        )

    target_dir = local_dir_for_hf_repo(models_dir, source)
    if is_prepared_model_dir(target_dir):
        return PreparedModel(
            model_id=entry.model_id,
            adapter=entry.adapter,
            source=source,
            local_path=str(target_dir.resolve()),
            status="available",
            message=f"Model {entry.model_id}: already prepared at {target_dir.resolve()}",
            success=True,
        )

    if offline:
        return PreparedModel(
            model_id=entry.model_id,
            adapter=entry.adapter,
            source=source,
            local_path=str(target_dir),
            status="missing_offline",
            message=(
                f"Model {entry.model_id}: missing local files for {source}. "
                f"Expected prepared directory: {target_dir}"
            ),
            success=False,
        )

    if dry_run:
        return PreparedModel(
            model_id=entry.model_id,
            adapter=entry.adapter,
            source=source,
            local_path=str(target_dir),
            status="dry_run",
            message=f"Model {entry.model_id}: would download {source} into {target_dir}",
            success=True,
        )

    try:
        downloaded_path = download_hf_snapshot(source, target_dir)
    except MissingHuggingFaceHubError as exc:
        return PreparedModel(
            model_id=entry.model_id,
            adapter=entry.adapter,
            source=source,
            local_path=str(target_dir),
            status="missing_dependency",
            message=str(exc),
            success=False,
        )
    except Exception as exc:  # pragma: no cover - depends on network and HF Hub
        return PreparedModel(
            model_id=entry.model_id,
            adapter=entry.adapter,
            source=source,
            local_path=str(target_dir),
            status="download_failed",
            message=f"Model {entry.model_id}: failed to download {source}: {exc}",
            success=False,
        )

    return PreparedModel(
        model_id=entry.model_id,
        adapter=entry.adapter,
        source=source,
        local_path=str(downloaded_path),
        status="downloaded",
        message=f"Model {entry.model_id}: downloaded {source} into {downloaded_path}",
        success=True,
    )


def prepare_models_from_registry(
    model_registry_path: str | Path,
    models_dir: str | Path,
    model_id: str | None = None,
    offline: bool = False,
    dry_run: bool = False,
) -> list[PreparedModel]:
    framework_root = _framework_root()
    registry_path = resolve_framework_path(model_registry_path, framework_root)
    resolved_models_dir = resolve_framework_path(models_dir, framework_root)

    entries = [
        entry
        for entry in load_model_registry(registry_path)
        if entry.enabled and (model_id is None or entry.model_id == model_id)
    ]

    results: list[PreparedModel] = []
    for entry in entries:
        if entry.adapter != "hf_local":
            results.append(skip_non_hf_model(entry))
            continue

        results.append(
            prepare_hf_model(
                entry=entry,
                models_dir=resolved_models_dir,
                registry_path=registry_path,
                framework_root=framework_root,
                offline=offline,
                dry_run=dry_run,
            )
        )

    return results


def print_preparation_results(results: list[PreparedModel]) -> None:
    if not results:
        print("No enabled models matched the requested selection.")
        return

    for result in results:
        prefix = "OK" if result.success else "ERROR"
        print(f"[{prefix}] {result.message}")
        if result.local_path:
            print(f"      local_path: {result.local_path}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare enabled benchmark models.")
    parser.add_argument(
        "--model-registry-path",
        default="models/model_registry_nvidia.yaml",
        help="Model registry path relative to Benchmark Framework unless absolute.",
    )
    parser.add_argument(
        "--models-dir",
        default="models_cache",
        help="Directory used for prepared models, relative to Benchmark Framework unless absolute.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Prepare only one enabled model from the registry.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Do not download; fail if required model files are not already local.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen without downloading anything.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    try:
        results = prepare_models_from_registry(
            model_registry_path=args.model_registry_path,
            models_dir=args.models_dir,
            model_id=args.model_id,
            offline=args.offline,
            dry_run=args.dry_run,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print_preparation_results(results)

    if args.model_id and not results:
        return 1
    if any(not result.success for result in results):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
