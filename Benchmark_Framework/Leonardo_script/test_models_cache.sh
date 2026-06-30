set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUBMIT_DIR="$(cd "${SLURM_SUBMIT_DIR:-${PWD}}" && pwd)"

FRAMEWORK_DIR=""
for CANDIDATE_FRAMEWORK_DIR in "${SCRIPT_DIR}/.." "${SUBMIT_DIR}" "${SUBMIT_DIR}/.." "${SUBMIT_DIR}/Benchmark_Framework" "${SUBMIT_DIR}/../Benchmark_Framework"; do
    if [ -f "${CANDIDATE_FRAMEWORK_DIR}/run_benchmark.py" ]; then
        FRAMEWORK_DIR="$(cd "${CANDIDATE_FRAMEWORK_DIR}" && pwd)"
        break
    fi
done

if [ -z "${FRAMEWORK_DIR}" ]; then
    echo "ERROR: cannot locate Benchmark_Framework from script dir or SLURM_SUBMIT_DIR."
    echo "Submit from the repository root or set SLURM_SUBMIT_DIR to the repository path."
    exit 1
fi

REPO_ROOT="$(cd "${FRAMEWORK_DIR}/.." && pwd)"

cd "${FRAMEWORK_DIR}"

module purge
module load python/3.11.7

VENV_ACTIVATED=0
for CANDIDATE_VENV_DIR in "${PYTHON_VENV:-}" "${VIRTUAL_ENV:-}" "${REPO_ROOT}/../our_env" "${REPO_ROOT}/our_env" "${FRAMEWORK_DIR}/our_env" "${REPO_ROOT}/project_venv" "${REPO_ROOT}/venv" "${REPO_ROOT}/.venv" "${REPO_ROOT}/.venv-new" "${FRAMEWORK_DIR}/project_venv" "${FRAMEWORK_DIR}/venv"; do
    if [ -z "${CANDIDATE_VENV_DIR}" ]; then
        continue
    fi
    VENV_DIR="$(cd "${CANDIDATE_VENV_DIR}" 2>/dev/null && pwd || true)"
    if [ -n "${VENV_DIR}" ] && [ -f "${VENV_DIR}/bin/activate" ]; then
        # shellcheck disable=SC1091
        source "${VENV_DIR}/bin/activate"
        echo "Activated venv: ${VENV_DIR}"
        VENV_ACTIVATED=1
        break
    fi
done

if [ "${VENV_ACTIVATED}" != "1" ]; then
    echo "ERROR: no Python venv found."
    echo "Set PYTHON_VENV=/absolute/path/to/your/venv, activate a venv before sbatch, or create ../our_env."
    exit 1
fi

export HF_HOME="${HF_HOME:-${FRAMEWORK_DIR}/models_cache/.hf_home}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

MODELS_DIR="${MODELS_DIR:-models_cache}"

echo "Job id: ${SLURM_JOB_ID:-manual}"
echo "Node: ${SLURMD_NODENAME:-local}"
echo "Repo root: ${REPO_ROOT}"
echo "Framework dir: ${FRAMEWORK_DIR}"
echo "Models dir: ${MODELS_DIR}"
echo "HF_HOME: ${HF_HOME}"
python --version

python - "${MODELS_DIR}" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


def resolve_models_dir(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def check_weight_indexes(model_dir: Path) -> list[str]:
    errors: list[str] = []
    index_paths = sorted(model_dir.glob("*.index.json"))

    for index_path in index_paths:
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{index_path.name}: cannot read index JSON: {exc}")
            continue

        weight_map = index.get("weight_map", {})
        if not isinstance(weight_map, dict):
            errors.append(f"{index_path.name}: missing or invalid weight_map")
            continue

        for filename in sorted(set(weight_map.values())):
            weight_path = model_dir / filename
            if not weight_path.exists():
                errors.append(f"missing weight shard: {filename}")
            elif weight_path.stat().st_size == 0:
                errors.append(f"empty weight shard: {filename}")

    return errors


def check_direct_weight_files(model_dir: Path) -> list[str]:
    weight_files = [
        *model_dir.glob("*.safetensors"),
        *model_dir.glob("*.bin"),
        *model_dir.glob("*.pt"),
    ]
    if not weight_files:
        return ["no weight files found"]
    return [f"empty weight file: {path.name}" for path in weight_files if path.stat().st_size == 0]


def check_model_dir(model_dir: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []

    try:
        AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        errors.append(f"AutoConfig failed: {exc}")

    try:
        AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    except Exception as exc:
        errors.append(f"AutoTokenizer failed: {exc}")

    errors.extend(check_weight_indexes(model_dir))
    errors.extend(check_direct_weight_files(model_dir))

    partial_files = sorted(
        [
            *model_dir.rglob("*.incomplete"),
            *model_dir.rglob("*.lock"),
        ]
    )
    errors.extend(f"partial/cache lock file present: {path.relative_to(model_dir)}" for path in partial_files)

    return not errors, errors


def main() -> int:
    models_dir = resolve_models_dir(sys.argv[1])
    if not models_dir.exists():
        print(f"ERROR: models directory does not exist: {models_dir}")
        return 1

    model_dirs = sorted(path for path in models_dir.iterdir() if path.is_dir() and not path.name.startswith("."))
    if not model_dirs:
        print(f"ERROR: no model directories found in {models_dir}")
        return 1

    failed = 0
    print(f"Checking {len(model_dirs)} model directories in {models_dir}")
    for model_dir in model_dirs:
        ok, errors = check_model_dir(model_dir)
        if ok:
            print(f"[OK] {model_dir.name}")
        else:
            failed += 1
            print(f"[ERROR] {model_dir.name}")
            for error in errors:
                print(f"  - {error}")

    if failed:
        print(f"FAILED: {failed}/{len(model_dirs)} model directories have issues.")
        return 1

    print(f"OK: all {len(model_dirs)} model directories look complete.")
    return 0


raise SystemExit(main())
PY
