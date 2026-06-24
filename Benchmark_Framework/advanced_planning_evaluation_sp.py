"""Advanced planning evaluation CLI.

This script turns ``analysis/notebooks/advanced_planning_evaluation.ipynb`` into
an interactive Python workflow. It reads benchmark artifacts from
``Benchmark_Framework/outputs``, computes the same core metrics as the notebook,
and writes a model-centric JSON report under ``results/``.
"""

from __future__ import annotations

import json
import math
import re
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd


# ── Fuzzy matching threshold ───────────────────────────────────────────────────
# An action name is accepted as a "fuzzy match" to a legal action if its
# Levenshtein distance to the closest legal name is ≤ MAX_FUZZY_DISTANCE.
# This prevents penalising minor transcription variations ("pick-up" vs "pickup")
# while still catching inventions with no close legal neighbour.
MAX_FUZZY_DISTANCE = 2

# Number of bootstrap resampling iterations for the 95% CI on PS.
# Seed is fixed (42) inside build_aggregate_tables for reproducibility.
BOOTSTRAP_N = 1000

# Valid difficulty tier names as written in the tasks directory structure.
DIFF_VALID = {"easy", "medium", "hard"}
# Display order for difficulty-level plots (unknown goes last).
DIFF_ORDER = ["easy", "medium", "hard", "unknown"]

# ── Composite Planning Score weights ──────────────────────────────────────────
# PS = 0.25·FASR + 0.20·IWSR + 0.20·exec_ratio + 0.20·(1−halluc) + 0.15·PAS
# FASR is weighted highest because first-attempt success is the only signal that
# cannot be inflated by retries. IWSR and exec_ratio capture structural quality
# from complementary angles. PAS has the lowest weight because it is undefined
# (NaN) for perfectly-executable plans, requiring a 0.5 prior substitution.
COMPOSITE_WEIGHTS = {
    "fasr": 0.25,
    "iwsr": 0.20,
    "exec_ratio": 0.20,
    "one_minus_halluc": 0.20,
    "pas": 0.15,
}
# When CoT alignment is available (chain-of-thought enabled), an optional +0.05
# bonus is added and all weights are renormalised so PS stays in [0, 1].
COT_BONUS_WEIGHT = 0.05

# ── Within-domain ranking directions ──────────────────────────────────────────
# For each metric listed here, "max" means rank 1 = highest value (better),
# "min" means rank 1 = lowest value (better). Used in build_aggregate_tables
# and domain_ranking_heatmap to give rank 1 consistently to the best performer.
RANK_METRICS = {
    "Success_Rate": "max",
    "FASR": "max",
    "IWSR": "max",
    "Exec": "max",
    "Halluc": "min",
    "PAS": "max",
    "CoT_Alignment": "max",
    "Retry_Gap": "min",
    "Temporal_Distance": "min",
}

ROW_METRIC_COLUMNS = [
    "Run_id",
    "Model",
    "Domain",
    "Problem",
    "Difficulty",
    "Protocol",
    "Valid",
    "Length",
    "Iterations",
    "Chain_of_Thought",
    "parsed_file_path",
    "pddl_available",
    "hallucinated_action_count",
    "fuzzy_hallucinated_count",
    "object_hallucination_count",
    "total_action_count",
    "total_arg_count",
    "hallucination_rate",
    "fuzzy_hallucination_rate",
    "object_hallucination_rate",
    "inverse_hallucination_rate",
    "executability_prefix_length",
    "executability_ratio",
    "sequencing_error_count",
    "state_fabrication_count",
    "precondition_awareness_score",
    "mean_temporal_distance",
    "cot_action_coverage",
    "cot_object_coverage",
    "cot_term_coverage",
    "cot_semantic_support_score",
    "cot_plan_alignment_score",
    "cot_plan_alignment_proxy_score",
    "cot_alignment_status",
    "cot_alignment_confidence",
    "cot_reasoning_plan_available",
    "cot_exact_sequence_match",
    "strict_or_proxy_alignment_value",
    "cot_alignment",
    "_iter1",
    "_iwsr_contrib",
]

_ACT_RE = re.compile(r"\(\s*([^\s()]+)((?:\s+[^\s()]+)*)\s*\)")


def add_warning(warnings_out: list[dict[str, Any]], warning_type: str, message: str, **extra: Any) -> None:
    """Append a structured warning dict to the warnings accumulator.

    Metric correlation: none — cross-cutting diagnostic facility.
    Rationale: Centralising warning collection into a typed list rather than
    printing to stderr allows the full warning trace to be embedded in the JSON
    report under the ``warnings`` key. Downstream consumers (notebooks, CI) can
    filter by ``type`` to surface only relevant issues.
    Code purpose: replaces ad-hoc ``print`` / ``logging.warn`` calls with a
    structured record that is serialisable and queryable.
    Detail: builds ``{"type": warning_type, "message": message, **extra}``
    and appends it; the ``**extra`` kwargs carry context such as run_id, domain,
    or file path so the reader can locate the problematic artifact without
    cross-referencing log lines.
    """
    payload = {"type": warning_type, "message": message}
    payload.update(extra)
    warnings_out.append(payload)


def json_safe(value: Any) -> Any:
    """Convert pandas/numpy/NaN values into JSON-safe primitives.

    Metric correlation: none — output serialisation utility.
    Rationale: numpy and pandas types (np.float64, np.int64, pd.NA) are not
    natively JSON-serialisable; ``json.dumps`` raises a TypeError on them.
    This function performs a deep recursive conversion so the entire report dict
    can be written with a single ``json.dumps`` call without a custom encoder.
    Code purpose: ensures the final JSON report is portable and loadable by any
    standard JSON parser regardless of the numeric backend used during computation.
    Detail: booleans are coerced before integers (np.bool_ is a subclass of
    np.integer in some numpy versions). Non-finite floats (inf, -inf, NaN) become
    ``null`` since JSON has no representation for them. Dicts, lists, tuples, and
    sets are recursively processed; all other types are returned unchanged.
    """
    if isinstance(value, dict):
        return {str(key): json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else None
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def records_from_df(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Serialise a DataFrame to a JSON-safe list of dicts (one per row).

    Metric correlation: none — output serialisation utility.
    Rationale: wraps pandas ``to_dict(orient="records")`` with ``json_safe``
    in a single call so callers do not need to remember both steps.
    Code purpose: produces the per-table ``"records"`` arrays that are embedded
    in each model's JSON payload under ``tables.*``.
    Detail: returns ``[]`` for empty DataFrames to avoid serialising column
    metadata without rows.
    """
    if df.empty:
        return []
    return json_safe(df.to_dict(orient="records"))


def scalar_float(value: Any, default: float = float("nan")) -> float:
    """Coerce any value to a plain Python float, substituting ``default`` for NaN/None.

    Metric correlation: used in composite score computation and profile evaluation
    to guard against NaN inputs from incomplete PDDL contexts or missing CoT data.
    Rationale: arithmetic on NaN propagates silently, producing NaN composite
    scores that appear as ``null`` in JSON. Explicit coercion with a documented
    default makes the substitution visible in code rather than hidden in pandas
    arithmetic rules.
    Code purpose: single point of NaN-to-default conversion for all scalar metric
    reads; the ``default`` parameter lets callers choose a neutral value (0.0 for
    additive terms, 0.5 for PAS where 50/50 is the uninformative prior).
    Detail: ``pd.isna`` is checked first because numpy floats and pandas NA both
    pass the ``isinstance`` checks but may not be caught by ``math.isnan``.
    """
    try:
        if pd.isna(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def nanmean_or_default(series: pd.Series, default: float = float("nan")) -> float:
    """Return the mean of a series after dropping NaNs, or ``default`` if all are NaN.

    Metric correlation: used in groupby aggregations for PAS and CoT_Alignment,
    which are NaN when PDDL context is missing or CoT was not enabled.
    Rationale: pandas ``mean()`` returns NaN for all-NaN series (correct), but
    returns NaN for partially-NaN series by default — which is also correct but
    may be undesired when we want a 0.5 PAS prior for groups with no PDDL context.
    The explicit drop-then-mean pattern makes the NaN handling explicit.
    Code purpose: aggregation kernel for ``build_aggregate_tables`` groupby lambdas
    where the caller supplies an appropriate ``default`` (0.5 for PAS, NaN for CoT).
    Detail: ``pd.to_numeric(errors="coerce")`` converts any non-numeric stragglers
    (e.g. string "nan") to NaN before dropping.
    """
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.mean()) if len(values) else default


def find_framework_root(start: Optional[Path] = None) -> Path:
    """Locate the Benchmark_Framework root by walking up the directory tree.

    Metric correlation: none — filesystem navigation.
    Rationale: the script may be invoked from any working directory (repo root,
    ``analysis/``, a notebook, CI). Hard-coding a relative path breaks portability.
    Walking upward until the ``outputs/`` + ``tasks/`` sentinel directories are
    found makes the script location-independent without requiring environment
    variables or config files.
    Code purpose: returns the authoritative ``Benchmark_Framework`` root Path so
    all downstream functions can construct absolute sub-paths (outputs/parsed,
    outputs/scored, tasks/, protocols/) without guessing.
    Detail: checks both the directory itself (if invoked from within the framework)
    and a ``Benchmark_Framework`` subdirectory (if invoked from the repo root).
    Raises ``FileNotFoundError`` with a descriptive message if neither is found.
    """
    start = (start or Path.cwd()).resolve()
    candidates = [start, *start.parents]
    for candidate in candidates:
        if (candidate / "outputs").exists() and (candidate / "tasks").exists():
            return candidate
        nested = candidate / "Benchmark_Framework"
        if (nested / "outputs").exists() and (nested / "tasks").exists():
            return nested
    raise FileNotFoundError("Could not locate Benchmark_Framework root.")


def default_results_dir(framework_root: Path) -> Path:
    return framework_root.parent / "results"


def sanitize_json_filename(raw_name: str) -> str:
    """Return a safe JSON filename, preserving a user-readable stem."""
    name = raw_name.strip().replace("\\", "_").replace("/", "_")
    if not name:
        name = "advanced_planning_evaluation"
    if name.lower().endswith(".json"):
        name = name[:-5]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._-")
    if not name:
        name = "advanced_planning_evaluation"
    return f"{name}.json"


def yes_no_prompt(prompt: str, default: bool = False, input_fn: Callable[[str], str] = input) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    while True:
        answer = input_fn(f"{prompt} {suffix}: ").strip().lower()
        if not answer:
            return default
        if answer in {"y", "yes", "s", "si", "si"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Risposta non valida. Usa y oppure n.")


def collect_run_status(outputs_root: Path) -> list[dict[str, Any]]:
    """Scan outputs/{raw,parsed,scored} and report per-run completeness.

    Metric correlation: none — data completeness gate. Only complete runs (present
    in all three layers) are eligible for metric computation.
    Rationale: the benchmark pipeline writes artifacts in three sequential layers.
    A run may be partially complete if it was interrupted. Including partial runs
    would silently produce wrong aggregate metrics (missing scored files would be
    treated as invalid plans, biasing Success_Rate downward).
    Code purpose: returns a list of status dicts, one per run_id found across any
    layer, with boolean flags per layer and a ``complete`` flag. This feeds
    ``select_run_ids_interactively`` which only exposes complete runs to the user.
    Detail: each layer dir is iterated for top-level directories; their names are
    the run_ids. A run is ``complete`` iff its directory exists in all three layers.
    Missing layers are listed under ``"missing"`` for diagnostic display.
    """
    layers = ("raw", "parsed", "scored")
    layer_runs: dict[str, set[str]] = {}
    for layer in layers:
        layer_dir = outputs_root / layer
        if not layer_dir.exists():
            layer_runs[layer] = set()
            continue
        layer_runs[layer] = {
            path.name
            for path in layer_dir.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }

    all_run_ids = sorted(set().union(*layer_runs.values()))
    statuses: list[dict[str, Any]] = []
    for run_id in all_run_ids:
        presence = {layer: run_id in layer_runs[layer] for layer in layers}
        missing = [layer for layer, present in presence.items() if not present]
        statuses.append(
            {
                "run_id": run_id,
                **presence,
                "complete": not missing,
                "missing": missing,
            }
        )
    return statuses


def complete_run_ids(statuses: Iterable[dict[str, Any]]) -> list[str]:
    return [status["run_id"] for status in statuses if status.get("complete")]


def print_run_status(statuses: list[dict[str, Any]]) -> None:
    complete = complete_run_ids(statuses)
    incomplete = [status for status in statuses if not status.get("complete")]

    print("\nRun completi:")
    if complete:
        for run_id in complete:
            print(f"  - {run_id}")
    else:
        print("  Nessun run completo trovato.")

    print("\nRun incompleti:")
    if incomplete:
        for status in incomplete:
            missing = ", ".join(status["missing"])
            print(f"  - {status['run_id']} (manca: {missing})")
    else:
        print("  Nessun run incompleto trovato.")


def select_run_ids_interactively(
    statuses: list[dict[str, Any]],
    input_fn: Callable[[str], str] = input,
) -> tuple[bool, list[str], list[dict[str, Any]]]:
    warnings_out: list[dict[str, Any]] = []
    complete = set(complete_run_ids(statuses))
    if not complete:
        raise RuntimeError("No complete run ids are available in raw, parsed, and scored.")

    do_merge = yes_no_prompt("Vuoi fare il merge logico di piu run?", default=False, input_fn=input_fn)
    selected: list[str] = []

    if do_merge:
        print("Inserisci i run id uno alla volta. Scrivi 'stop' per terminare.")
        while True:
            run_id = input_fn("Run id: ").strip()
            if run_id.lower() == "stop":
                if selected:
                    break
                print("Devi selezionare almeno un run completo prima di usare stop.")
                continue
            if run_id not in complete:
                print("Run id non valido o incompleto. Riprova.")
                add_warning(warnings_out, "rejected_run_id", "Run id is not complete.", run_id=run_id)
                continue
            if run_id in selected:
                print("Run id gia selezionato.")
                continue
            selected.append(run_id)
            print(f"Aggiunto: {run_id}")
    else:
        while True:
            run_id = input_fn("Run id da analizzare: ").strip()
            if run_id in complete:
                selected = [run_id]
                break
            print("Run id non valido o incompleto. Riprova.")
            add_warning(warnings_out, "rejected_run_id", "Run id is not complete.", run_id=run_id)

    return do_merge, selected, warnings_out


def choose_output_paths_interactively(
    results_dir: Path,
    timestamp_file: str,
    input_fn: Callable[[str], str] = input,
) -> tuple[Path, Path, bool, str]:
    use_custom_name = yes_no_prompt("Vuoi dare un nome diverso al file JSON?", default=False, input_fn=input_fn)
    if use_custom_name:
        raw_name = input_fn("Nome file JSON: ").strip()
        json_name = sanitize_json_filename(raw_name)
    else:
        json_name = f"advanced_planning_evaluation_{timestamp_file}.json"

    json_path = results_dir / json_name
    plots_dir = results_dir / json_path.stem
    return json_path, plots_dir, use_custom_name, json_name


def levenshtein(s1: str, s2: str) -> int:
    """Compute the Levenshtein edit distance between two strings.

    Metric correlation: fuzzy hallucination rate — an action name is counted as
    a fuzzy hallucination only if its edit distance to every legal action name
    exceeds ``MAX_FUZZY_DISTANCE`` (default 2).
    Rationale: LLMs sometimes produce near-miss action names ("pick-up" instead
    of "pickup") that are semantically correct but lexically different. A strict
    exact-match hallucination check would penalise these unfairly, overstating
    hallucination. The fuzzy rate with distance ≤ 2 captures genuine vocabulary
    failures while being tolerant of minor transcription errors.
    Code purpose: row-by-row edit distance helper called inside
    ``compute_hallucination_metrics`` for each (generated_action, legal_action) pair.
    Detail: space-optimised Wagner-Fischer DP using a single rolling row of length
    ``len(s2)+1``; always ensures ``len(s1) >= len(s2)`` for the inner loop.
    """
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    previous = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        current = [i]
        for j, c2 in enumerate(s2, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (c1 != c2)))
        previous = current
    return previous[-1]


def parse_action(action_str: str) -> tuple[Optional[str], list[str]]:
    """Parse a PDDL action string "(name arg1 arg2 …)" into (name, [args]).

    Metric correlation: hallucination rate, executability ratio, PAS — all metrics
    are computed by iterating over parsed (name, args) pairs from the model's plan.
    Rationale: the model outputs free-form text containing action tuples. Extracting
    the action name and argument list with a single regex avoids per-character
    string splitting that would break on nested parentheses or extra whitespace.
    Code purpose: lowest-level tokeniser for a single plan step; called in
    ``compute_hallucination_metrics`` and ``compute_precondition_metrics``.
    Detail: ``_ACT_RE`` matches one parenthesized PDDL action atom.
    captures the head token as group 1 and the argument token list as group 2.
    Returns ``(None, [])`` if the string does not match PDDL atom syntax.
    """
    match = _ACT_RE.match(action_str.strip())
    if not match:
        return None, []
    name = match.group(1).lower()
    args = match.group(2).strip().lower().split() if match.group(2).strip() else []
    return name, args


def safe_get(data: Any, path: str | Iterable[Any], default: Any = None) -> Any:
    parts = path.split(".") if isinstance(path, str) else list(path)
    current = data
    missing = object()
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part, missing)
        elif isinstance(current, list) and isinstance(part, int) and 0 <= part < len(current):
            current = current[part]
        else:
            return default
        if current is missing:
            return default
    return current


def parsed_plan_raw_actions(parsed_plan: dict[str, Any]) -> list[str]:
    raw_plan = parsed_plan.get("raw")
    if isinstance(raw_plan, dict) and isinstance(raw_plan.get("actions"), list):
        return raw_plan["actions"]
    actions = parsed_plan.get("actions")
    return actions if isinstance(actions, list) else []


def parsed_plan_reasoning_actions(parsed_plan: dict[str, Any]) -> list[str]:
    reasoning = parsed_plan.get("reasoning")
    if isinstance(reasoning, dict) and isinstance(reasoning.get("actions"), list):
        return reasoning["actions"]
    return []


def parsed_plan_reasoning_text(parsed_plan: dict[str, Any]) -> str:
    reasoning = parsed_plan.get("reasoning")
    return reasoning if isinstance(reasoning, str) else ""


def raw_attempt_reasoning_text(raw_attempt: dict[str, Any] | None) -> str:
    value = safe_get(raw_attempt or {}, "generation.reasoning_text", "")
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def extract_pddl_actions_from_text(text: str) -> list[str]:
    return re.findall(r"\([^()\n]+\)", text or "")


def parse_domain_pddl(domain_path: Path, warnings_out: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract action names, predicate names, and per-action schemas from a PDDL domain file.

    Metric correlation: all PDDL-grounded metrics (hallucination rate, executability,
    PAS, temporal distance) depend on the action schema parsed here.
    Rationale: a lightweight regex-based parser avoids a full PDDL library dependency
    while covering the propositional and simple numeric subsets used in the benchmark.
    It is intentionally permissive — syntax errors emit a warning but do not abort.
    Code purpose: produces the ``domain_info`` dict consumed by
    ``compute_hallucination_metrics``, ``compute_precondition_metrics``, and
    ``PDDLSimulator``. The ``"schemas"`` sub-dict maps each action name to its raw
    ``:parameters``, ``:precondition``, and ``:effect`` strings for grounding.
    Detail: comments are stripped first (``; … \\n``), then the text is lowercased
    for case-insensitive matching. Predicate names are extracted from the
    ``:predicates`` block; functions from ``:functions``; each ``:action`` block is
    located by searching for the next ``:action``/``:durative`` boundary.
    """
    result = {"action_names": set(), "predicates": set(), "functions": set(), "schemas": {}, "path": str(domain_path)}
    try:
        text = domain_path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r";[^\n]*", "", text).lower()

        predicates_match = re.search(r"\(:predicates(.*?)\)(?=\s*\(:|\s*\))", text, re.DOTALL)
        if predicates_match:
            result["predicates"] = set(re.findall(r"\(\s*([a-z][a-z0-9_-]*)", predicates_match.group(1)))

        functions_match = re.search(r"\(:functions(.*?)\)(?=\s*\(:|\s*\))", text, re.DOTALL)
        if functions_match:
            result["functions"] = set(re.findall(r"\(\s*([a-z][a-z0-9_-]*)", functions_match.group(1)))

        for match in re.finditer(r"\(:action\s+([a-z][a-z0-9_-]*)", text):
            action_name = match.group(1)
            start = match.start()
            next_match = re.search(r"\(:action|\(:durative|\Z", text[start + 1 :])
            block = text[start : start + 1 + (next_match.start() if next_match else len(text))]
            result["action_names"].add(action_name)

            params_match = re.search(r":parameters\s*\(([^)]*)\)", block)
            params = re.findall(r"\?([a-z][a-z0-9_-]*)", params_match.group(1)) if params_match else []

            prec_pos = block.find(":precondition")
            eff_pos = block.find(":effect")
            prec_raw = block[prec_pos + len(":precondition") : eff_pos].strip() if prec_pos != -1 and eff_pos != -1 else ""
            eff_raw = block[eff_pos + len(":effect") :].strip() if eff_pos != -1 else ""
            result["schemas"][action_name] = {"params": params, "prec_raw": prec_raw, "eff_raw": eff_raw}
    except Exception as exc:
        add_warning(warnings_out, "domain_parser_error", str(exc), path=str(domain_path))
    return result


def parse_problem_pddl(problem_path: Path, warnings_out: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract objects, initial propositions, and numeric fluents from a PDDL problem file.

    Metric correlation: object hallucination rate uses ``"objects"``; PDDLSimulator
    uses ``"init_atoms"`` and ``"init_numeric"`` to initialise the world state.
    Rationale: each problem instance defines its own object vocabulary and initial
    state. Object hallucination (using argument tokens not in ``:objects``) is a
    separate signal from action-name hallucination — a model may know the legal
    action names but still invent new objects, indicating shallow grounding.
    Code purpose: produces the ``problem_info`` dict keyed by
    ``(domain, difficulty, instance)`` for use in per-row metric computation.
    Detail: the ``:objects`` block is tokenised after stripping type declarations
    (tokens starting with "-" are type separators). The ``:init`` block is scanned
    for numeric assignments ``(= (fluent …) value)`` and propositional atoms
    ``(predicate arg…)``; logical keywords are excluded from the atom parse.
    """
    result = {
        "objects": set(),
        "init_atoms": set(),
        "init_numeric": {},
        "path": str(problem_path),
        "difficulty": problem_path.parent.name,
    }
    try:
        text = problem_path.read_text(encoding="utf-8", errors="replace")
        text = re.sub(r";[^\n]*", "", text).lower()

        objects_match = re.search(r"\(:objects(.*?)\)(?=\s*\(:|\s*\))", text, re.DOTALL)
        if objects_match:
            for token in objects_match.group(1).split():
                if not token.startswith("-") and re.match(r"^[a-z][a-z0-9_-]*$", token):
                    result["objects"].add(token)

        init_match = re.search(r"\(:init(.*?)(?=\(:goal|\(:metric|\Z)", text, re.DOTALL)
        if init_match:
            init_text = init_match.group(1)
            for numeric_match in re.finditer(r"\(=\s*\(\s*([a-z][a-z0-9_-]*)([^)]*)\)\s*([\d.]+)\s*\)", init_text):
                key = (numeric_match.group(1),) + tuple(numeric_match.group(2).strip().split())
                result["init_numeric"][key] = float(numeric_match.group(3))
            for atom_match in re.finditer(r"\(([a-z][a-z0-9_-]*(?:\s+[a-z0-9][a-z0-9_-]*)*)\)", init_text):
                tokens = atom_match.group(1).split()
                if tokens and tokens[0] not in {"=", "not", "and", "or", "increase", "decrease"}:
                    result["init_atoms"].add(tuple(tokens))
    except Exception as exc:
        add_warning(warnings_out, "problem_parser_error", str(exc), path=str(problem_path))
    return result


def index_pddl_context(tasks_dir: Path, warnings_out: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[tuple[str, str, str], Any]]:
    domain_info: dict[str, Any] = {}
    problem_info: dict[tuple[str, str, str], Any] = {}
    if not tasks_dir.exists():
        add_warning(warnings_out, "missing_tasks_dir", "Tasks directory does not exist.", path=str(tasks_dir))
        return domain_info, problem_info

    for domain_dir in sorted(tasks_dir.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith("_") or domain_dir.name == "metadata":
            continue
        domain_name = domain_dir.name
        domain_path = domain_dir / "domain" / "domain.pddl"
        if domain_path.exists():
            domain_info[domain_name] = parse_domain_pddl(domain_path, warnings_out)
        else:
            add_warning(warnings_out, "missing_domain_pddl", "Domain file not found.", domain=domain_name, path=str(domain_path))
            domain_info[domain_name] = {
                "action_names": set(),
                "predicates": set(),
                "functions": set(),
                "schemas": {},
                "path": None,
            }

        for difficulty in DIFF_VALID:
            difficulty_dir = domain_dir / difficulty
            if not difficulty_dir.exists():
                continue
            for problem_path in sorted(difficulty_dir.glob("*.pddl")):
                key = (domain_name, difficulty, problem_path.stem)
                problem_info[key] = parse_problem_pddl(problem_path, warnings_out)
    return domain_info, problem_info


def load_protocol_cot_flags(protocols_dir: Path, warnings_out: list[dict[str, Any]]) -> dict[str, bool]:
    protocol_cot: dict[str, bool] = {}
    if not protocols_dir.exists():
        add_warning(warnings_out, "missing_protocols_dir", "Protocols directory does not exist.", path=str(protocols_dir))
        return protocol_cot
    for protocol_file in sorted(protocols_dir.glob("*.yaml")):
        try:
            text = protocol_file.read_text(encoding="utf-8")
            protocol_match = re.search(r"^protocol_id\s*:\s*(\S+)", text, re.MULTILINE)
            cot_match = re.search(r"include_chain_of_thought\s*:\s*(\S+)", text)
            if protocol_match and cot_match:
                protocol_cot[protocol_match.group(1)] = cot_match.group(1).lower() == "true"
        except Exception as exc:
            add_warning(warnings_out, "protocol_read_error", str(exc), path=str(protocol_file))
    return protocol_cot


def read_json_file(path: Path, warnings_out: list[dict[str, Any]], warning_type: str) -> Optional[dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        add_warning(warnings_out, warning_type, str(exc), path=str(path))
        return None


def load_artifact_rows(
    framework_root: Path,
    selected_run_ids: list[str],
    warnings_out: list[dict[str, Any]],
) -> pd.DataFrame:
    """Load all parsed and scored artifacts for the selected runs into a flat DataFrame.

    Metric correlation: all metrics — this function produces the raw per-problem
    row DataFrame that feeds every downstream computation.
    Rationale: the parsed layer stores per-problem JSON files with the model's plan
    attempts; the scored layer stores validation outcomes. Joining them here into a
    single flat DataFrame decouples the filesystem walk from the metric logic and
    ensures every downstream function receives a uniform tabular input.
    Code purpose: iterates ``outputs/parsed/<run_id>/<model>/<protocol>/<domain>/
    <difficulty>/<instance>.json``, extracts the last non-empty plan (to handle
    retry loops), reads the corresponding scored file for ``solved`` / ``iterations``
    flags, and emits one row per problem instance.
    Detail: the parsed file may nest the action list under ``raw.actions`` or
    directly under ``actions`` — ``parsed_plan_raw_actions`` handles both. CoT flag
    is read from ``protocols/<protocol>.yaml``; a non-empty reasoning text also
    triggers CoT=True as a fallback. Private ``_actions`` and ``_cot_text`` columns
    are retained for per-row metric computation and dropped from the JSON output.
    """
    parsed_dir = framework_root / "outputs" / "parsed"
    scored_dir = framework_root / "outputs" / "scored"
    raw_dir = framework_root / "outputs" / "raw"
    protocol_cot = load_protocol_cot_flags(framework_root / "protocols", warnings_out)

    rows: list[dict[str, Any]] = []
    for run_id in selected_run_ids:
        run_dir = parsed_dir / run_id
        if not run_dir.exists():
            add_warning(warnings_out, "missing_parsed_run", "Parsed run directory does not exist.", run_id=run_id, path=str(run_dir))
            continue

        for json_file in sorted(run_dir.rglob("*.json")):
            rel_parts = json_file.relative_to(run_dir).parts
            if len(rel_parts) != 5:
                continue
            model, protocol, domain, difficulty, filename = rel_parts
            if difficulty not in DIFF_VALID:
                continue
            instance = Path(filename).stem

            parsed_data = read_json_file(json_file, warnings_out, "parsed_read_error")
            if parsed_data is None:
                continue
            attempts = parsed_data.get("attempts", [])
            attempts = attempts if isinstance(attempts, list) else []

            raw_path = raw_dir / run_id / model / protocol / domain / difficulty / filename
            raw_data = read_json_file(raw_path, warnings_out, "raw_read_error") if raw_path.exists() else None
            raw_attempts = raw_data.get("attempts", []) if isinstance(raw_data, dict) else []
            raw_attempts = raw_attempts if isinstance(raw_attempts, list) else []

            actions: list[str] = []
            cot_text = ""
            for attempt_index in range(len(attempts) - 1, -1, -1):
                attempt = attempts[attempt_index]
                parsed_plan = attempt.get("parsed_plan") or {}
                candidate = parsed_plan_raw_actions(parsed_plan)
                if candidate:
                    actions = candidate
                    raw_attempt = raw_attempts[attempt_index] if attempt_index < len(raw_attempts) else None
                    cot_text = raw_attempt_reasoning_text(raw_attempt) or parsed_plan_reasoning_text(parsed_plan)
                    break
            if not actions and attempts:
                parsed_plan = attempts[-1].get("parsed_plan") or {}
                actions = parsed_plan_raw_actions(parsed_plan)
                raw_attempt = raw_attempts[-1] if raw_attempts else None
                cot_text = raw_attempt_reasoning_text(raw_attempt) or parsed_plan_reasoning_text(parsed_plan)

            scored_path = scored_dir / run_id / model / protocol / domain / difficulty / filename
            scored = read_json_file(scored_path, warnings_out, "scored_read_error") if scored_path.exists() else None
            if scored is None:
                if not scored_path.exists():
                    add_warning(
                        warnings_out,
                        "missing_scored_file",
                        "No scored file found for parsed artifact.",
                        path=str(scored_path),
                    )
                is_valid = False
                iterations = len(attempts)
                scored_attempts: list[dict[str, Any]] = []
            else:
                is_valid = bool(scored.get("solved", False))
                iterations = scored.get("iterations_used", len(attempts))
                scored_attempts = scored.get("attempts", [])
                scored_attempts = scored_attempts if isinstance(scored_attempts, list) else []

            rows.append(
                {
                    "Model": model,
                    "Domain": domain,
                    "Problem": instance,
                    "Difficulty": difficulty,
                    "Protocol": protocol,
                    "Run_id": run_id,
                    "Valid": is_valid,
                    "Length": len(actions),
                    "Iterations": iterations,
                    "Chain_of_Thought": protocol_cot.get(protocol, False) or bool(cot_text.strip()) or any(parsed_plan_reasoning_actions((attempt.get("parsed_plan") or {})) for attempt in attempts if isinstance(attempt, dict)),
                    "_actions": actions,
                    "_cot_text": cot_text,
                    "_parsed_attempts": attempts,
                    "_scored_attempts": scored_attempts,
                    "_raw_attempts": raw_attempts,
                    "_file_path": str(json_file),
                    "parsed_file_path": str(json_file),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "Model",
                "Domain",
                "Problem",
                "Difficulty",
                "Protocol",
                "Run_id",
                "Valid",
                "Length",
                "Iterations",
                "Chain_of_Thought",
                "_actions",
                "_cot_text",
                "_parsed_attempts",
                "_scored_attempts",
                "_raw_attempts",
                "_file_path",
                "parsed_file_path",
            ]
        )
    df["Iterations"] = pd.to_numeric(df["Iterations"], errors="coerce")
    return df


def compute_hallucination_metrics(actions: list[str], d_info: dict[str, Any], p_info: dict[str, Any]) -> dict[str, Any]:
    """Compute strict, fuzzy, and object hallucination counts and rates for one plan.

    Metric correlation:
    - ``hallucination_rate`` (Halluc): fraction of actions whose names are not in the
      domain's legal action set. High Halluc → No Grounding or early Vocabulary-Only.
    - ``fuzzy_hallucination_rate``: same, but an action is only counted if its
      edit distance to every legal action exceeds MAX_FUZZY_DISTANCE (2). A model
      with low strict-halluc but high fuzzy-halluc is making near-miss naming errors.
    - ``object_hallucination_rate``: fraction of argument tokens not in the problem's
      ``:objects`` block, normalised over total argument count. A high object-halluc
      with low action-halluc signals the model knows the action vocabulary but
      invents new world entities.
    - ``inverse_hallucination_rate`` (IHR = 1 − hallucination_rate): used in
      composite scoring and profile thresholds because higher is better.
    Rationale (Kambhampati 2024, Vafa et al. 2024): hallucination is the most
    basic signal of domain grounding. A model that does not hallucinate has at
    least memorised the schema, even if it cannot sequence correctly.
    Code purpose: per-row metric computation for one model/problem pair, called
    inside ``compute_row_metrics``.
    Detail: iterates actions, calls ``parse_action`` for the name and args, then
    checks membership in ``legal_actions`` (strict) and minimum Levenshtein distance
    (fuzzy). Object hallucination counts each argument token not in
    ``legal_objects``; returns NaN for rates when the plan is empty.
    """
    legal_actions = d_info.get("action_names", set())
    legal_objects = p_info.get("objects", set())
    hall_strict = hall_fuzzy = obj_hall = total_actions = total_args = 0

    for action_str in actions:
        action_name, action_args = parse_action(action_str)
        if action_name is None:
            continue
        total_actions += 1
        total_args += len(action_args)

        if action_name not in legal_actions:
            hall_strict += 1
            if not legal_actions or min(levenshtein(action_name, legal_action) for legal_action in legal_actions) > MAX_FUZZY_DISTANCE:
                hall_fuzzy += 1

        for arg in action_args:
            if legal_objects and arg not in legal_objects:
                obj_hall += 1

    hallucination_rate = hall_strict / total_actions if total_actions else float("nan")
    return {
        "hallucinated_action_count": hall_strict,
        "fuzzy_hallucinated_count": hall_fuzzy,
        "object_hallucination_count": obj_hall,
        "total_action_count": total_actions,
        "total_arg_count": total_args,
        "hallucination_rate": hallucination_rate,
        "fuzzy_hallucination_rate": hall_fuzzy / total_actions if total_actions else float("nan"),
        "object_hallucination_rate": obj_hall / total_args if total_args else float("nan"),
        "inverse_hallucination_rate": 1 - hallucination_rate if not math.isnan(hallucination_rate) else float("nan"),
    }


class PDDLSimulator:
    """Lightweight forward PDDL state simulator for propositional and simple numeric domains.

    Metric correlation:
    - ``executability_ratio`` (ER): how far into the plan the simulator runs before
      the first precondition failure. High ER → model understands local state
      transitions even if it fails globally.
    - ``sequencing_error_count`` / ``state_fabrication_count``: at the failure step,
      each failed propositional precondition is classified as a sequencing error
      (the atom was true at some earlier step — it was deleted by a prior action the
      model forgot to account for) or a state fabrication (the atom was never true —
      the model invented a world state that never existed).
    - ``precondition_awareness_score`` (PAS = seq_errors / total_failures):
      distinguishes "model understands state transitions but mis-orders actions"
      (high PAS, sequencing dominant) from "model fabricates preconditions" (low PAS,
      fabrication dominant). Correlates with Dziri et al. (2023) compositionality
      analysis and Valmeekam et al. (2023) Phase 2 grounding tests.
    - ``mean_temporal_distance``: for sequencing errors, the number of simulator
      steps between the failure point and the last step at which the violated
      precondition was true. Low distance → model is close to the correct ordering;
      high distance → ordering collapse (the prerequisite was deleted many steps ago).
    Rationale: a full external PDDL validator (VAL) would require a subprocess and
    a pre-compiled binary. This class reproduces the validation logic in pure Python
    for portability while keeping enough fidelity for the propositional + numeric-
    inequality fragments used in the benchmark.
    Code purpose: instantiated once per (problem, plan) pair inside
    ``compute_precondition_metrics``; ``prop_hist`` accumulates snapshots of the
    propositional state after each step, enabling ``last_step_atom_true`` lookups
    for temporal distance computation.
    Detail: effects are applied sequentially (delete effects first, then add
    effects, mirroring STRIPS semantics). Numeric fluents support increase/decrease
    assignments and comparison preconditions (>=, <=, >, <). Ungrounded schemas
    (wrong arity) count as fabrication errors and terminate simulation immediately.
    """

    def __init__(self, d_info: dict[str, Any], p_info: dict[str, Any]) -> None:
        self.d_info = d_info
        self.prop = set(p_info.get("init_atoms", set()))
        self.num = dict(p_info.get("init_numeric", {}))
        self.prop_hist = [frozenset(self.prop)]
        self.step = 0

    def _ground(self, raw: str, params: list[str], args: list[str]) -> str:
        result = raw
        for param, arg in zip(params, args):
            result = re.sub(r"\?" + re.escape(param) + r"\b", arg, result)
        return result

    def _eval_num(self, expr: str) -> Optional[float]:
        expr = expr.strip()
        try:
            return float(expr)
        except ValueError:
            pass

        fluent_match = re.match(r"^\(\s*([a-z][a-z0-9_-]*)([^)]*)\)$", expr)
        if fluent_match:
            key = (fluent_match.group(1),) + tuple(fluent_match.group(2).strip().split())
            return self.num.get(key)

        op_match = re.match(r"^\(\s*([+\-*/])\s(.+)\)$", expr, re.DOTALL)
        if op_match:
            op, inner = op_match.group(1), op_match.group(2).strip()
            depth = 0
            operands: list[str] = []
            start = 0
            for index, char in enumerate(inner):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == " " and depth == 0:
                    segment = inner[start:index].strip()
                    if segment:
                        operands.append(segment)
                    start = index + 1
            segment = inner[start:].strip()
            if segment:
                operands.append(segment)
            if len(operands) == 2:
                v1 = self._eval_num(operands[0])
                v2 = self._eval_num(operands[1])
                if v1 is not None and v2 is not None:
                    if op == "+":
                        return v1 + v2
                    if op == "-":
                        return v1 - v2
                    if op == "*":
                        return v1 * v2
                    if op == "/" and v2:
                        return v1 / v2
        return None

    def check_preconditions(self, prec_raw: str, params: list[str], args: list[str]) -> tuple[bool, list[tuple[str, ...]], list[str]]:
        grounded = self._ground(prec_raw, params, args)
        failed_prop: list[tuple[str, ...]] = []
        failed_num: list[str] = []
        funcs = self.d_info.get("functions", set())
        skip_heads = {
            "and",
            "or",
            "not",
            ">=",
            "<=",
            ">",
            "<",
            "=",
            "increase",
            "decrease",
            "when",
            "forall",
            "exists",
            "+",
            "-",
            "*",
            "/",
        }

        for numeric_match in re.finditer(r"\((>=|<=|>|<)\s*([^)]+)\s*([0-9.]+)\s*\)", grounded):
            op, lhs_s, rhs_s = numeric_match.group(1), numeric_match.group(2).strip(), numeric_match.group(3)
            lhs = self._eval_num(lhs_s)
            rhs = float(rhs_s)
            if lhs is None:
                continue
            ok = {">=": lhs >= rhs, "<=": lhs <= rhs, ">": lhs > rhs, "<": lhs < rhs}[op]
            if not ok:
                failed_num.append(numeric_match.group(0))

        no_negative = re.sub(r"\(not\s+\([^)]+\)\s*\)", "", grounded)
        for atom_match in re.finditer(r"\(([a-z][a-z0-9_-]*)([^()]*)\)", no_negative):
            head = atom_match.group(1)
            if head in skip_heads or head in funcs:
                continue
            atom = (head,) + tuple(atom_match.group(2).strip().split()) if atom_match.group(2).strip() else (head,)
            if atom not in self.prop:
                failed_prop.append(atom)

        return not failed_prop and not failed_num, failed_prop, failed_num

    def apply_effects(self, eff_raw: str, params: list[str], args: list[str]) -> None:
        grounded = self._ground(eff_raw, params, args)
        funcs = self.d_info.get("functions", set())
        skip_heads = {"and", "or", "not", ">=", "<=", ">", "<", "=", "increase", "decrease", "when", "forall", "assign", "+", "-", "*", "/"}

        for delete_match in re.finditer(r"\(not\s+\(([a-z][a-z0-9_-]*)([^)]*)\)\s*\)", grounded):
            atom = (delete_match.group(1),) + tuple(delete_match.group(2).strip().split())
            self.prop.discard(atom)

        no_delete = re.sub(r"\(not\s+\([^)]+\)\s*\)", "", grounded)
        for atom_match in re.finditer(r"\(([a-z][a-z0-9_-]*)([^()]*)\)", no_delete):
            head = atom_match.group(1)
            if head in skip_heads or head in funcs:
                continue
            atom = (head,) + tuple(atom_match.group(2).strip().split()) if atom_match.group(2).strip() else (head,)
            self.prop.add(atom)

        for op_name in ("increase", "decrease"):
            pattern = r"\(" + op_name + r"\s+\(\s*([a-z][a-z0-9_-]*)([^)]*)\)\s*([0-9.]+|\([^)]+\))\s*\)"
            for numeric_match in re.finditer(pattern, grounded):
                key = (numeric_match.group(1),) + tuple(numeric_match.group(2).strip().split())
                value = self._eval_num(numeric_match.group(3).strip())
                if value is not None:
                    self.num[key] = self.num.get(key, 0.0) + value if op_name == "increase" else self.num.get(key, 0.0) - value

        self.prop_hist.append(frozenset(self.prop))
        self.step += 1

    def last_step_atom_true(self, atom: tuple[str, ...]) -> int:
        for index in range(len(self.prop_hist) - 1, -1, -1):
            if atom in self.prop_hist[index]:
                return index
        return -1


def compute_precondition_metrics(actions: list[str], d_info: dict[str, Any], p_info: dict[str, Any]) -> dict[str, Any]:
    nan_result = {
        "executability_prefix_length": 0,
        "executability_ratio": float("nan"),
        "sequencing_error_count": 0,
        "state_fabrication_count": 0,
        "precondition_awareness_score": float("nan"),
        "mean_temporal_distance": float("nan"),
    }
    plan_len = len(actions)
    if plan_len == 0:
        return nan_result

    try:
        sim = PDDLSimulator(d_info, p_info)
        schemas = d_info.get("schemas", {})
        first_failure = plan_len
        seq_errors = 0
        fab_errors = 0
        temporal_distances: list[float] = []

        for step, action_str in enumerate(actions):
            action_name, action_args = parse_action(action_str)
            if action_name is None:
                continue
            schema = schemas.get(action_name)
            if schema is None:
                continue

            params = schema.get("params", [])
            if len(params) != len(action_args):
                first_failure = min(first_failure, step)
                fab_errors += 1
                break

            satisfied, failed_prop, failed_num = sim.check_preconditions(schema.get("prec_raw", ""), params, action_args)
            if not satisfied:
                first_failure = min(first_failure, step)
                for atom in failed_prop:
                    last_step = sim.last_step_atom_true(atom)
                    if last_step >= 0:
                        seq_errors += 1
                        temporal_distances.append(step - last_step)
                    else:
                        fab_errors += 1
                fab_errors += len(failed_num)
                break

            sim.apply_effects(schema.get("eff_raw", ""), params, action_args)

        total_fail = seq_errors + fab_errors
        return {
            "executability_prefix_length": first_failure,
            "executability_ratio": first_failure / plan_len,
            "sequencing_error_count": seq_errors,
            "state_fabrication_count": fab_errors,
            "precondition_awareness_score": seq_errors / total_fail if total_fail > 0 else float("nan"),
            "mean_temporal_distance": float(np.mean(temporal_distances)) if temporal_distances else float("nan"),
        }
    except Exception as exc:
        warnings.warn(f"[simulator] {exc}")
        return nan_result


def normalize_action_sequence(actions: list[Any], lowercase: bool = True) -> list[str]:
    result: list[str] = []
    for action in actions or []:
        if not isinstance(action, str):
            continue
        normalized = " ".join(action.strip().split())
        if lowercase:
            normalized = normalized.lower()
        if normalized:
            result.append(normalized)
    return result


def common_prefix_length(left: list[str], right: list[str]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def lcs_length(left: list[str], right: list[str]) -> int:
    if not left or not right:
        return 0
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            current.append(previous[index - 1] + 1 if left_item == right_item else max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def action_bag_overlap_score(left: list[str], right: list[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = sum((Counter(left) & Counter(right)).values())
    return overlap / max(len(left), len(right), 1)


def contiguous_repetition_blocks(actions: list[str]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    start = 0
    while start < len(actions):
        end = start
        while end + 1 < len(actions) and actions[end + 1] == actions[start]:
            end += 1
        if end > start:
            blocks.append({"action": actions[start], "start": start, "end": end, "length": end - start + 1})
        start = end + 1
    return blocks


def repetition_similarity(left: list[str], right: list[str]) -> float:
    left_repeated = sum(count - 1 for count in Counter(left).values() if count > 1)
    right_repeated = sum(count - 1 for count in Counter(right).values() if count > 1)
    denominator = max(left_repeated, right_repeated, 1)
    return 1.0 - (abs(left_repeated - right_repeated) / denominator)


def detect_adjacent_swaps(left: list[str], right: list[str]) -> list[dict[str, Any]]:
    swaps: list[dict[str, Any]] = []
    max_index = min(len(left), len(right)) - 1
    index = 0
    while index < max_index:
        if left[index] == right[index + 1] and left[index + 1] == right[index] and left[index] != left[index + 1]:
            swaps.append({"index": index, "raw_pair": [left[index], left[index + 1]], "reasoning_pair": [right[index], right[index + 1]]})
            index += 2
        else:
            index += 1
    return swaps


def compute_sequence_alignment(raw_actions: list[Any], reasoning_actions: list[Any]) -> dict[str, Any]:
    raw = normalize_action_sequence(raw_actions)
    reasoning = normalize_action_sequence(reasoning_actions)
    max_len = max(len(raw), len(reasoning), 1)
    min_len = min(len(raw), len(reasoning))
    prefix = common_prefix_length(raw, reasoning)
    lcs = lcs_length(raw, reasoning)
    length_ratio = min_len / max_len
    prefix_ratio = prefix / max(min_len, 1)
    lcs_ratio = lcs / max_len
    bag_score = action_bag_overlap_score(raw, reasoning)
    repeat_similarity = repetition_similarity(raw, reasoning)
    adjacent_swaps = detect_adjacent_swaps(raw, reasoning)

    first_mismatch_index = None
    mismatch_examples: list[dict[str, Any]] = []
    displaced_actions: list[dict[str, Any]] = []
    for index in range(max(len(raw), len(reasoning))):
        raw_action = raw[index] if index < len(raw) else None
        reasoning_action = reasoning[index] if index < len(reasoning) else None
        if raw_action == reasoning_action:
            continue
        if first_mismatch_index is None:
            first_mismatch_index = index
        if len(mismatch_examples) < 5:
            mismatch_examples.append({"index": index, "raw_action": raw_action, "reasoning_action": reasoning_action})
        if raw_action in reasoning or reasoning_action in raw:
            displaced_actions.append({"index": index, "raw_action": raw_action, "reasoning_action": reasoning_action})

    raw_counts = Counter(raw)
    reasoning_counts = Counter(reasoning)
    structural_alignment = (
        0.35 * lcs_ratio
        + 0.25 * prefix_ratio
        + 0.20 * bag_score
        + 0.10 * length_ratio
        + 0.10 * repeat_similarity
    )
    raw_blocks = contiguous_repetition_blocks(raw)
    reasoning_blocks = contiguous_repetition_blocks(reasoning)
    return {
        "raw_actions": raw,
        "reasoning_actions": reasoning,
        "raw_plan_length": len(raw),
        "reasoning_plan_length": len(reasoning),
        "length_ratio": length_ratio,
        "exact_sequence_match": raw == reasoning,
        "common_prefix_length": prefix,
        "common_prefix_ratio": prefix_ratio,
        "lcs_length": lcs,
        "lcs_ratio": lcs_ratio,
        "action_bag_overlap_score": bag_score,
        "repetition_similarity": repeat_similarity,
        "structural_alignment": float(np.clip(structural_alignment, 0, 1)),
        "raw_distinct_action_count": len(raw_counts),
        "reasoning_distinct_action_count": len(reasoning_counts),
        "raw_action_frequencies": dict(raw_counts),
        "reasoning_action_frequencies": dict(reasoning_counts),
        "raw_repeated_action_total": sum(count - 1 for count in raw_counts.values() if count > 1),
        "reasoning_repeated_action_total": sum(count - 1 for count in reasoning_counts.values() if count > 1),
        "raw_repetition_block_count": len(raw_blocks),
        "reasoning_repetition_block_count": len(reasoning_blocks),
        "raw_repetition_blocks": raw_blocks,
        "reasoning_repetition_blocks": reasoning_blocks,
        "adjacent_swaps": adjacent_swaps,
        "adjacent_swap_count": len(adjacent_swaps),
        "displaced_actions": displaced_actions,
        "displaced_action_count": len(displaced_actions),
        "first_mismatch_index": first_mismatch_index,
        "mismatch_examples": mismatch_examples,
    }


def compute_cot_semantic_support(cot_text: str, actions: list[str], d_info: dict[str, Any], p_info: dict[str, Any]) -> dict[str, Any]:
    legal_action_names = d_info.get("action_names", set())
    legal_objects = p_info.get("objects", set())
    cot_tokens = set(re.findall(r"[a-z][a-z0-9_-]*", (cot_text or "").lower()))

    plan_action_names: set[str] = set()
    plan_objects: set[str] = set()
    for action in actions:
        name, args = parse_action(action)
        if name:
            plan_action_names.add(name)
            plan_objects.update(args)

    plan_terms = plan_action_names | plan_objects
    cot_action_mentioned = cot_tokens & legal_action_names
    cot_object_mentioned = cot_tokens & legal_objects
    cot_action_cov = len(cot_action_mentioned & plan_action_names) / max(len(plan_action_names), 1)
    cot_object_cov = len(cot_object_mentioned & plan_objects) / max(len(plan_objects), 1)
    cot_term_cov = len(cot_tokens & plan_terms) / max(len(plan_terms), 1)
    return {
        "cot_action_coverage": cot_action_cov,
        "cot_object_coverage": cot_object_cov,
        "cot_term_coverage": cot_term_cov,
        "cot_semantic_support_score": (cot_action_cov + cot_object_cov) / 2,
    }


def compute_cot_alignment(cot_text: str, actions: list[str], d_info: dict[str, Any], p_info: dict[str, Any]) -> dict[str, Any]:
    return compute_cot_semantic_support(cot_text, actions, d_info, p_info)


def _bool_or_none(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _prefix_ratio(prefix: Any, length: int) -> Optional[float]:
    if isinstance(prefix, int) and length > 0:
        return prefix / length
    return None


def compute_cot_alignment_for_attempt(
    parsed_attempt: dict[str, Any],
    scored_attempt: dict[str, Any] | None,
    raw_attempt: dict[str, Any] | None,
    d_info: dict[str, Any],
    p_info: dict[str, Any],
    semantic_proxy_cap: float = 0.35,
) -> dict[str, Any]:
    parsed_plan = parsed_attempt.get("parsed_plan") if isinstance(parsed_attempt, dict) else {}
    parsed_plan = parsed_plan if isinstance(parsed_plan, dict) else {}
    scored_attempt = scored_attempt if isinstance(scored_attempt, dict) else {}
    raw_actions = normalize_action_sequence(parsed_plan_raw_actions(parsed_plan))
    reasoning_actions = normalize_action_sequence(parsed_plan_reasoning_actions(parsed_plan))
    reasoning_text = raw_attempt_reasoning_text(raw_attempt) or parsed_plan_reasoning_text(parsed_plan)
    semantic = (
        compute_cot_semantic_support(reasoning_text, raw_actions, d_info, p_info)
        if reasoning_text.strip()
        else {
            "cot_action_coverage": None,
            "cot_object_coverage": None,
            "cot_term_coverage": None,
            "cot_semantic_support_score": None,
        }
    )

    raw_valid = _bool_or_none(safe_get(scored_attempt, "final_plan_valid"))
    raw_valid_inferred = False
    if raw_valid is None:
        raw_valid = _bool_or_none(safe_get(scored_attempt, "validation_result.valid"))
        raw_valid_inferred = raw_valid is not None
    reasoning_valid = _bool_or_none(safe_get(scored_attempt, "reasoning_final_plan_valid"))
    raw_prefix = safe_get(scored_attempt, "first_valid_prefix_length")
    reasoning_prefix = safe_get(scored_attempt, "reasoning_first_valid_prefix_length")

    sequence: dict[str, Any] = {}
    plan_score = None
    proxy_score = None
    exact_match = None
    confidence = "none"
    status = "no_reasoning_text"
    basis = "none"

    if not reasoning_text.strip() and not reasoning_actions:
        status = "no_reasoning_text"
    elif not raw_actions:
        status = "no_raw_plan"
        confidence = "low"
    elif raw_actions and reasoning_actions:
        sequence = compute_sequence_alignment(raw_actions, reasoning_actions)
        plan_score = sequence["structural_alignment"]
        exact_match = sequence["exact_sequence_match"]
        confidence = "high"
        if raw_valid is True and reasoning_valid is True:
            status = "comparable_and_both_valid"
        elif raw_valid is False and reasoning_valid is False:
            status = "comparable_but_both_invalid"
        elif raw_valid is False:
            status = "comparable_but_raw_invalid"
        elif reasoning_valid is False:
            status = "comparable_but_reasoning_invalid"
        else:
            status = "comparable_plans"
    else:
        score = semantic.get("cot_semantic_support_score")
        proxy_score = score * semantic_proxy_cap if isinstance(score, (int, float)) and math.isfinite(score) else None
        status = "semantic_proxy_only"
        confidence = "low"

    if raw_actions and reasoning_actions:
        if raw_valid is True and reasoning_valid is False:
            basis = "raw"
        elif raw_valid is False and reasoning_valid is True:
            basis = "reasoning"
        elif raw_valid is True and reasoning_valid is True:
            basis = "raw" if len(raw_actions) <= len(reasoning_actions) else "reasoning"
        elif raw_valid is False and reasoning_valid is False:
            basis = "reasoning"
        else:
            basis = "raw"

    raw_prefix_ratio = _prefix_ratio(raw_prefix, len(raw_actions))
    reasoning_prefix_ratio = _prefix_ratio(reasoning_prefix, len(reasoning_actions))
    iteration = parsed_attempt.get("iteration") if isinstance(parsed_attempt, dict) else None
    result = {
        "iteration": iteration,
        "cot_plan_alignment_score": plan_score,
        "cot_plan_alignment_proxy_score": proxy_score,
        "cot_alignment_status": status,
        "cot_alignment_confidence": confidence,
        "cot_reasoning_plan_available": bool(reasoning_actions),
        "cot_exact_sequence_match": exact_match,
        "strict_or_proxy_alignment_value": plan_score if plan_score is not None else proxy_score,
        "raw_valid": raw_valid,
        "raw_valid_inferred": raw_valid_inferred,
        "reasoning_valid": reasoning_valid,
        "raw_first_valid_prefix_length": raw_prefix,
        "reasoning_first_valid_prefix_length": reasoning_prefix,
        "raw_prefix_ratio": raw_prefix_ratio,
        "reasoning_prefix_ratio": reasoning_prefix_ratio,
        "raw_has_shorter_valid_prefix": isinstance(raw_prefix, int) and 0 < raw_prefix < len(raw_actions),
        "reasoning_has_shorter_valid_prefix": isinstance(reasoning_prefix, int) and 0 < reasoning_prefix < len(reasoning_actions),
        "basis": basis,
        **semantic,
        **sequence,
    }
    result.setdefault("raw_plan_length", len(raw_actions))
    result.setdefault("reasoning_plan_length", len(reasoning_actions))
    return result


def select_cot_alignment_attempt(by_iteration: list[dict[str, Any]], solved: bool) -> dict[str, Any]:
    if not by_iteration:
        return {"selected_attempt": None, "selected_attempt_reason": "none", "final": {}}
    if len(by_iteration) == 1:
        selected_index = 0
        reason = "only_attempt"
    elif solved:
        selected_index = next((index for index, item in enumerate(by_iteration) if item.get("raw_valid") is True), len(by_iteration) - 1)
        reason = "first_solved_attempt" if selected_index < len(by_iteration) - 1 or by_iteration[selected_index].get("raw_valid") is True else "last_attempt"
    else:
        selected_index = len(by_iteration) - 1
        reason = "last_attempt"
    final = by_iteration[selected_index]
    return {
        "selected_attempt": final.get("iteration") or selected_index + 1,
        "selected_attempt_reason": reason,
        "final": final,
    }


def compute_row_metrics(
    df_raw: pd.DataFrame,
    domain_info: dict[str, Any],
    problem_info: dict[tuple[str, str, str], Any],
    warnings_out: list[dict[str, Any]],
) -> pd.DataFrame:
    """Attach all per-row metrics to the raw artifact DataFrame.

    Metric correlation: this function computes or attaches every per-row metric
    column listed in ``ROW_METRIC_COLUMNS``:
    - hallucination_rate, fuzzy_hallucination_rate, object_hallucination_rate,
      inverse_hallucination_rate (via ``compute_hallucination_metrics``)
    - executability_ratio, sequencing_error_count, state_fabrication_count,
      precondition_awareness_score, mean_temporal_distance
      (via ``compute_precondition_metrics`` + PDDLSimulator)
    - cot_action_coverage, cot_object_coverage, cot_semantic_support_score
      (via ``compute_cot_semantic_support``)
    - cot_plan_alignment_score and proxy/status diagnostics
      (via ``compute_cot_alignment_for_attempt``)
    - ``_iter1``: True iff Valid=True AND Iterations=1 — contributes to FASR
    - ``_iwsr_contrib``: 1/Iterations if Valid else 0 — contributes to IWSR
    Rationale: computing all metrics in a single row-level pass avoids repeated
    DataFrame scans and keeps the intermediate private columns (_actions, _cot_text)
    alive for the computation without including them in the final output.
    Code purpose: the returned DataFrame is the single source of truth for all
    subsequent ``build_aggregate_tables`` groupby operations and plot generation.
    Detail: PDDL context missing warnings are emitted for rows where neither
    domain actions nor problem objects are available; those rows still get NaN
    metric values rather than being dropped so the problem count stays accurate.
    """
    df = df_raw.copy()
    if df.empty:
        for column in ROW_METRIC_COLUMNS:
            if column not in df.columns:
                df[column] = pd.Series(dtype="float64")
        return df

    halluc_rows: list[dict[str, Any]] = []
    precondition_rows: list[dict[str, Any]] = []
    cot_rows: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        domain = row["Domain"]
        difficulty = row.get("Difficulty", "unknown")
        instance = row["Problem"]
        d_info = domain_info.get(domain, {})
        p_info = problem_info.get((domain, difficulty, instance), {})
        pddl_ok = bool(d_info.get("action_names")) and bool(p_info.get("objects"))

        if not pddl_ok:
            add_warning(
                warnings_out,
                "missing_pddl_context",
                "Domain or problem PDDL context is incomplete for row.",
                domain=domain,
                difficulty=difficulty,
                problem=instance,
                path=row.get("_file_path"),
            )

        actions = row.get("_actions", [])
        halluc = compute_hallucination_metrics(actions, d_info, p_info)
        halluc["pddl_available"] = pddl_ok
        halluc_rows.append(halluc)
        precondition_rows.append(compute_precondition_metrics(actions, d_info, p_info))

        parsed_attempts = row.get("_parsed_attempts", [])
        scored_attempts = row.get("_scored_attempts", [])
        raw_attempts = row.get("_raw_attempts", [])
        parsed_attempts = parsed_attempts if isinstance(parsed_attempts, list) else []
        scored_attempts = scored_attempts if isinstance(scored_attempts, list) else []
        raw_attempts = raw_attempts if isinstance(raw_attempts, list) else []
        if not parsed_attempts and actions:
            parsed_attempts = [{"iteration": 1, "parsed_plan": {"actions": actions}}]

        by_iteration: list[dict[str, Any]] = []
        for attempt_index in range(max(len(parsed_attempts), len(scored_attempts), len(raw_attempts))):
            parsed_attempt = parsed_attempts[attempt_index] if attempt_index < len(parsed_attempts) else {"iteration": attempt_index + 1, "parsed_plan": {}}
            scored_attempt = scored_attempts[attempt_index] if attempt_index < len(scored_attempts) else {}
            raw_attempt = raw_attempts[attempt_index] if attempt_index < len(raw_attempts) else {}
            by_iteration.append(compute_cot_alignment_for_attempt(parsed_attempt, scored_attempt, raw_attempt, d_info, p_info))

        selected = select_cot_alignment_attempt(by_iteration, bool(row.get("Valid", False)))
        final = selected.get("final", {})
        cot_rows.append(
            {
                "cot_action_coverage": final.get("cot_action_coverage"),
                "cot_object_coverage": final.get("cot_object_coverage"),
                "cot_term_coverage": final.get("cot_term_coverage"),
                "cot_semantic_support_score": final.get("cot_semantic_support_score"),
                "cot_plan_alignment_score": final.get("cot_plan_alignment_score"),
                "cot_plan_alignment_proxy_score": final.get("cot_plan_alignment_proxy_score"),
                "cot_alignment_status": final.get("cot_alignment_status"),
                "cot_alignment_confidence": final.get("cot_alignment_confidence"),
                "cot_reasoning_plan_available": final.get("cot_reasoning_plan_available"),
                "cot_exact_sequence_match": final.get("cot_exact_sequence_match"),
                "strict_or_proxy_alignment_value": final.get("strict_or_proxy_alignment_value"),
                "cot_alignment": {**selected, "by_iteration": by_iteration},
            }
        )

    df = pd.concat(
        [
            df.reset_index(drop=True),
            pd.DataFrame(halluc_rows).reset_index(drop=True),
            pd.DataFrame(precondition_rows).reset_index(drop=True),
            pd.DataFrame(cot_rows).reset_index(drop=True),
        ],
        axis=1,
    )
    df["_iter1"] = (df["Valid"] == True) & (df["Iterations"] == 1)
    df["_iwsr_contrib"] = df.apply(
        lambda row: (1.0 / row["Iterations"])
        if row["Valid"] and pd.notna(row["Iterations"]) and row["Iterations"] > 0
        else 0.0,
        axis=1,
    )
    return df


def compute_composite_score(model_stats: dict[str, Any], weights: dict[str, float] = COMPOSITE_WEIGHTS, cot_bonus_w: float = COT_BONUS_WEIGHT) -> float:
    """Compute the Composite Planning Score (PS) for one model from aggregate stats.

    Metric correlation: PS is the primary single-number summary of planning quality,
    encoding five complementary dimensions:
    - FASR (0.25): zero-shot strength — the purest signal of real planning ability.
    - IWSR (0.20): retry efficiency — rewards models that solve quickly when they
      need multiple attempts, penalises stochastic search.
    - exec_ratio (0.20): structural validity — how far the plan runs before the
      first precondition failure, independent of global success.
    - one_minus_halluc (0.20): schema grounding — using only legal action names.
    - PAS (0.15): precondition awareness — sequencing understanding vs. fabrication.
    CoT bonus (0.05): if the strict CoT plan alignment score is finite, the five
    weights are renormalised to sum to 1.0
    after adding the bonus, so PS remains in [0, 1].
    Rationale: FASR is weighted highest because first-attempt success is the only
    measure that cannot be inflated by retries. IWSR and Exec capture different
    aspects of structural quality. PAS gets the lowest weight because it is
    undefined (NaN) for models with zero failures — assigning 0.5 as a prior when
    PAS is unknown means a perfect model is not penalised.
    Code purpose: called in ``build_aggregate_tables`` for both the overall and
    per-domain tables, and in bootstrap CI computation. Also callable standalone.
    Detail: missing or NaN values are replaced with neutral defaults via
    ``scalar_float`` before the weighted sum; ``np.clip`` enforces [0, 1] to
    guard against floating-point edge cases.
    """
    base = (
        weights["fasr"] * scalar_float(model_stats.get("fasr", 0), 0)
        + weights["iwsr"] * scalar_float(model_stats.get("iwsr", 0), 0)
        + weights["exec_ratio"] * scalar_float(model_stats.get("exec_ratio", 0), 0)
        + weights["one_minus_halluc"] * scalar_float(model_stats.get("one_minus_halluc", 0), 0)
        + weights["pas"] * scalar_float(model_stats.get("pas", 0.5), 0.5)
    )
    cot = scalar_float(model_stats.get("cot_alignment", float("nan")))
    if math.isfinite(cot):
        total_w = sum(weights.values()) + cot_bonus_w
        base = (base + cot_bonus_w * cot) / total_w
    return float(np.clip(base, 0, 1))


def compute_iteration_profile(model_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Compute P(Valid | reached attempt k) for k = 1 … max_iterations.

    Metric correlation: iteration profile — discriminates Stochastic Searcher
    (flat P(Valid|k) curve) from Efficient Corrector (rising curve).
    Rationale: overall SR conflates first-attempt successes with retry-dependent
    ones. The iteration profile makes the retry structure explicit: at each attempt
    slot k, "reached_count" is the number of problems that were attempted at least
    k times; "exact_count" is the subset that required exactly k attempts; and
    P(Valid|k) = mean(Valid) among those exact-k rows. A rising curve means the
    model learns from earlier failures; a flat or declining curve means retries are
    independent random draws (Stechly et al. 2023).
    Code purpose: produces the ``iteration_profile`` list stored in each model's
    JSON payload under ``tables.iteration_profile``, and feeds ``profile_is_flat``
    for the Stochastic Searcher profile condition.
    Detail: iterates k from 1 to max_iter; ``reached`` = rows where Iterations ≥ k;
    ``exact`` = rows where Iterations == k. Returns empty list for empty DataFrames.
    """
    max_iter = int(model_df["Iterations"].dropna().max()) if not model_df["Iterations"].dropna().empty else 0
    rows: list[dict[str, Any]] = []
    for k in range(1, max_iter + 1):
        reached = model_df[model_df["Iterations"] >= k]
        exact = reached[reached["Iterations"] == k]
        probability = exact["Valid"].mean() if len(exact) else float("nan")
        rows.append(
            {
                "iteration": k,
                "reached_count": int(len(reached)),
                "exact_count": int(len(exact)),
                "p_valid_given_reached": probability,
            }
        )
    return rows


def profile_is_flat(iteration_profile: list[dict[str, Any]], tolerance: float = 0.10) -> Optional[bool]:
    """Test whether the P(Valid|k) iteration profile is approximately flat.

    Metric correlation: optional condition for the Stochastic Searcher profile.
    Rationale: a flat curve (max − min ≤ tolerance=0.10) indicates the probability
    of success is independent of attempt number, consistent with the model sampling
    independently at each retry rather than correcting. The tolerance of 0.10 gives
    a 10 percentage-point buffer for sampling noise in small problem sets.
    Code purpose: called inside ``classify_capability_profile`` to populate the
    ``p_valid_k_flat`` key in the metrics dict passed to profile condition lambdas.
    Detail: extracts finite P(Valid|k) values, returns None if fewer than 2 finite
    values exist (inconclusive), True if range ≤ tolerance, False otherwise.
    """
    values = [
        scalar_float(row.get("p_valid_given_reached"))
        for row in iteration_profile
        if math.isfinite(scalar_float(row.get("p_valid_given_reached")))
    ]
    if len(values) < 2:
        return None
    return max(values) - min(values) <= tolerance


def build_aggregate_tables(df_metrics: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate per-row metrics into all summary tables used by plots and the JSON report.

    Metric correlation: produces eight tables, each aggregating a different slice:
    - ``overall``: one row per model; all primary metrics (SR, FASR, IWSR, Exec,
      Halluc, IHR, PAS, CoT_Alignment, Temporal_Distance, Retry_Gap, PS).
    - ``by_domain``: one row per (model, domain); same metrics — feeds domain-level
      ranking and the PS stacked bar.
    - ``by_difficulty``: one row per (model, difficulty level) — feeds FASR-by-
      difficulty bar chart.
    - ``cot_summary``: SR and CoT plan alignment split by CoT flag.
    - ``failure_breakdown``: total and proportional sequencing vs. fabrication errors
      per model — feeds failure type stacked bar.
    - ``retry_gap``: SR, FASR, IWSR, and RG (= SR − FASR) sorted by RG.
    - ``composite_score``: PS with 95% bootstrap CI (N=1000, seed=42) and error bars.
    - ``rank_within_domain``: model ranks on every RANK_METRICS metric within each
      domain (rank 1 = best, direction per RANK_METRICS).
    Rationale: separating aggregation from plot generation keeps each layer testable
    and allows the JSON report to embed all tables independently of whether plots
    are requested.
    Code purpose: single entry point for all groupby logic; called once in
    ``build_report`` and passed to both ``maybe_make_plots`` and ``build_model_payloads``.
    Detail: FASR uses the private ``_iter1`` boolean column; IWSR uses ``_iwsr_contrib``.
    Bootstrap CI resamples per-row PS contributions (not per-model means) for a
    more conservative estimate. PS is added to both ``overall`` and ``by_domain``
    tables so they can be used independently for plotting.
    """
    if df_metrics.empty:
        empty = pd.DataFrame()
        return {
            "overall": empty,
            "by_domain": empty,
            "by_difficulty": empty,
            "cot_summary": empty,
            "failure_breakdown": empty,
            "retry_gap": empty,
            "composite_score": empty,
            "rank_within_domain": empty,
        }

    overall = df_metrics.groupby("Model", dropna=False).agg(
        N=("Problem", "count"),
        Runs=("Run_id", "nunique"),
        Domains=("Domain", "nunique"),
        Protocols=("Protocol", "nunique"),
        Success_Rate=("Valid", "mean"),
        FASR=("_iter1", "mean"),
        IWSR=("_iwsr_contrib", "mean"),
        Exec=("executability_ratio", "mean"),
        Halluc=("hallucination_rate", "mean"),
        Fuzzy_Halluc=("fuzzy_hallucination_rate", "mean"),
        Object_Halluc=("object_hallucination_rate", "mean"),
        IHR=("inverse_hallucination_rate", "mean"),
        PAS=("precondition_awareness_score", lambda x: nanmean_or_default(x, 0.5)),
        CoT_Alignment=("cot_plan_alignment_score", lambda x: nanmean_or_default(x, float("nan"))),
        Temporal_Distance=("mean_temporal_distance", lambda x: nanmean_or_default(x, float("nan"))),
        Avg_Length=("Length", "mean"),
        Avg_Iterations=("Iterations", "mean"),
        strict_mean_cot_plan_alignment=("cot_plan_alignment_score", lambda x: nanmean_or_default(x, float("nan"))),
        inclusive_mean_cot_alignment=("strict_or_proxy_alignment_value", lambda x: nanmean_or_default(x, float("nan"))),
        mean_cot_semantic_support=("cot_semantic_support_score", lambda x: nanmean_or_default(x, float("nan"))),
        cot_reasoning_plan_available_rate=("cot_reasoning_plan_available", lambda x: pd.Series(x).fillna(False).astype(bool).mean()),
        cot_exact_sequence_match_rate=("cot_exact_sequence_match", lambda x: pd.Series(x).fillna(False).astype(bool).mean()),
        cot_semantic_proxy_only_rate=("cot_alignment_status", lambda x: (pd.Series(x) == "semantic_proxy_only").mean()),
        cot_comparable_but_both_invalid_rate=("cot_alignment_status", lambda x: (pd.Series(x) == "comparable_but_both_invalid").mean()),
        cot_comparable_and_both_valid_rate=("cot_alignment_status", lambda x: (pd.Series(x) == "comparable_and_both_valid").mean()),
    ).reset_index()
    overall["Retry_Gap"] = overall["Success_Rate"] - overall["FASR"]

    by_domain = df_metrics.groupby(["Model", "Domain"], dropna=False).agg(
        N=("Problem", "count"),
        Runs=("Run_id", "nunique"),
        Success_Rate=("Valid", "mean"),
        FASR=("_iter1", "mean"),
        IWSR=("_iwsr_contrib", "mean"),
        Exec=("executability_ratio", "mean"),
        Halluc=("hallucination_rate", "mean"),
        Fuzzy_Halluc=("fuzzy_hallucination_rate", "mean"),
        Object_Halluc=("object_hallucination_rate", "mean"),
        IHR=("inverse_hallucination_rate", "mean"),
        PAS=("precondition_awareness_score", lambda x: nanmean_or_default(x, 0.5)),
        CoT_Alignment=("cot_plan_alignment_score", lambda x: nanmean_or_default(x, float("nan"))),
        Temporal_Distance=("mean_temporal_distance", lambda x: nanmean_or_default(x, float("nan"))),
        Avg_Length=("Length", "mean"),
        Avg_Iterations=("Iterations", "mean"),
        strict_mean_cot_plan_alignment=("cot_plan_alignment_score", lambda x: nanmean_or_default(x, float("nan"))),
        inclusive_mean_cot_alignment=("strict_or_proxy_alignment_value", lambda x: nanmean_or_default(x, float("nan"))),
        mean_cot_semantic_support=("cot_semantic_support_score", lambda x: nanmean_or_default(x, float("nan"))),
        cot_reasoning_plan_available_rate=("cot_reasoning_plan_available", lambda x: pd.Series(x).fillna(False).astype(bool).mean()),
        cot_exact_sequence_match_rate=("cot_exact_sequence_match", lambda x: pd.Series(x).fillna(False).astype(bool).mean()),
        cot_semantic_proxy_only_rate=("cot_alignment_status", lambda x: (pd.Series(x) == "semantic_proxy_only").mean()),
        cot_comparable_but_both_invalid_rate=("cot_alignment_status", lambda x: (pd.Series(x) == "comparable_but_both_invalid").mean()),
        cot_comparable_and_both_valid_rate=("cot_alignment_status", lambda x: (pd.Series(x) == "comparable_and_both_valid").mean()),
    ).reset_index()
    by_domain["Retry_Gap"] = by_domain["Success_Rate"] - by_domain["FASR"]

    by_difficulty = df_metrics.groupby(["Model", "Difficulty"], dropna=False).agg(
        N=("Problem", "count"),
        Success_Rate=("Valid", "mean"),
        FASR=("_iter1", "mean"),
        IWSR=("_iwsr_contrib", "mean"),
        Exec=("executability_ratio", "mean"),
        Halluc=("hallucination_rate", "mean"),
        IHR=("inverse_hallucination_rate", "mean"),
        PAS=("precondition_awareness_score", lambda x: nanmean_or_default(x, 0.5)),
        strict_mean_cot_plan_alignment=("cot_plan_alignment_score", lambda x: nanmean_or_default(x, float("nan"))),
        inclusive_mean_cot_alignment=("strict_or_proxy_alignment_value", lambda x: nanmean_or_default(x, float("nan"))),
        mean_cot_semantic_support=("cot_semantic_support_score", lambda x: nanmean_or_default(x, float("nan"))),
    ).reset_index()

    cot_summary = df_metrics.groupby(["Model", "Chain_of_Thought"], dropna=False).agg(
        N=("Problem", "count"),
        Success_Rate=("Valid", "mean"),
        CoT_Alignment=("cot_plan_alignment_score", lambda x: nanmean_or_default(x, float("nan"))),
        strict_mean_cot_plan_alignment=("cot_plan_alignment_score", lambda x: nanmean_or_default(x, float("nan"))),
        inclusive_mean_cot_alignment=("strict_or_proxy_alignment_value", lambda x: nanmean_or_default(x, float("nan"))),
        mean_cot_semantic_support=("cot_semantic_support_score", lambda x: nanmean_or_default(x, float("nan"))),
        cot_reasoning_plan_available_rate=("cot_reasoning_plan_available", lambda x: pd.Series(x).fillna(False).astype(bool).mean()),
        cot_exact_sequence_match_rate=("cot_exact_sequence_match", lambda x: pd.Series(x).fillna(False).astype(bool).mean()),
        cot_semantic_proxy_only_rate=("cot_alignment_status", lambda x: (pd.Series(x) == "semantic_proxy_only").mean()),
        cot_comparable_but_both_invalid_rate=("cot_alignment_status", lambda x: (pd.Series(x) == "comparable_but_both_invalid").mean()),
        cot_comparable_and_both_valid_rate=("cot_alignment_status", lambda x: (pd.Series(x) == "comparable_and_both_valid").mean()),
    ).reset_index()

    failure_breakdown = df_metrics.groupby("Model", dropna=False).agg(
        sequencing_error_count=("sequencing_error_count", "sum"),
        state_fabrication_count=("state_fabrication_count", "sum"),
    ).reset_index()
    total_failures = failure_breakdown["sequencing_error_count"] + failure_breakdown["state_fabrication_count"]
    failure_breakdown["total_failures"] = total_failures
    failure_breakdown["sequencing_error_proportion"] = np.where(total_failures > 0, failure_breakdown["sequencing_error_count"] / total_failures, 0)
    failure_breakdown["state_fabrication_proportion"] = np.where(total_failures > 0, failure_breakdown["state_fabrication_count"] / total_failures, 0)

    retry_gap = overall[["Model", "Success_Rate", "FASR", "IWSR", "Retry_Gap"]].sort_values("Retry_Gap", ascending=True)

    for table in (overall, by_domain):
        table["one_minus_halluc"] = 1 - table["Halluc"].fillna(0)
        table["pas_for_score"] = table["PAS"].fillna(0.5)
        table["PS"] = table.apply(
            lambda row: compute_composite_score(
                {
                    "fasr": row["FASR"],
                    "iwsr": row["IWSR"],
                    "exec_ratio": row["Exec"],
                    "one_minus_halluc": row["one_minus_halluc"],
                    "pas": row["pas_for_score"],
                    "cot_alignment": row["CoT_Alignment"],
                }
            ),
            axis=1,
        )

    rng = np.random.default_rng(seed=42)
    ci_rows: list[dict[str, Any]] = []
    for model, model_df in df_metrics.groupby("Model", dropna=False):
        contribs = model_df.apply(
            lambda row: compute_composite_score(
                {
                    "fasr": float(row.get("_iter1", 0) or 0),
                    "iwsr": float(row.get("_iwsr_contrib", 0) or 0),
                    "exec_ratio": scalar_float(row.get("executability_ratio", 0), 0),
                    "one_minus_halluc": 1 - scalar_float(row.get("hallucination_rate", 0), 0)
                    if math.isfinite(scalar_float(row.get("hallucination_rate", float("nan"))))
                    else 0.5,
                    "pas": scalar_float(row.get("precondition_awareness_score", 0.5), 0.5),
                    "cot_alignment": scalar_float(row.get("cot_plan_alignment_score", float("nan"))),
                }
            ),
            axis=1,
        ).values
        if len(contribs) == 0:
            ci_rows.append({"Model": model, "PS_ci_lo": 0.0, "PS_ci_hi": 0.0})
            continue
        boots = [contribs[rng.integers(0, len(contribs), len(contribs))].mean() for _ in range(BOOTSTRAP_N)]
        ci_rows.append({"Model": model, "PS_ci_lo": np.percentile(boots, 2.5), "PS_ci_hi": np.percentile(boots, 97.5)})

    composite_score = overall[["Model", "PS"]].rename(columns={"PS": "PS_overall"}).merge(pd.DataFrame(ci_rows), on="Model", how="left")
    composite_score["err_lo"] = composite_score["PS_overall"] - composite_score["PS_ci_lo"]
    composite_score["err_hi"] = composite_score["PS_ci_hi"] - composite_score["PS_overall"]
    composite_score = composite_score.sort_values("PS_overall", ascending=False)

    rank_rows: list[dict[str, Any]] = []
    for domain, domain_df in by_domain.groupby("Domain", dropna=False):
        sub = domain_df.set_index("Model")
        rank_table = pd.DataFrame(index=sub.index)
        for metric, direction in RANK_METRICS.items():
            if metric not in sub.columns:
                rank_table[metric] = float("nan")
            else:
                rank_table[metric] = sub[metric].rank(
                    ascending=direction == "min",
                    method="min",
                    na_option="bottom",
                )
        rank_table = rank_table.reset_index()
        rank_table["Domain"] = domain
        rank_rows.extend(rank_table.to_dict(orient="records"))
    rank_within_domain = pd.DataFrame(rank_rows)

    return {
        "overall": overall,
        "by_domain": by_domain,
        "by_difficulty": by_difficulty,
        "cot_summary": cot_summary,
        "failure_breakdown": failure_breakdown,
        "retry_gap": retry_gap,
        "composite_score": composite_score,
        "rank_within_domain": rank_within_domain,
    }


# ── Capability profile definitions ────────────────────────────────────────────
# Each entry defines one of the seven capability profiles from the LLM planning
# literature. Conditions are (label_string, predicate_lambda) pairs where the
# lambda receives a metrics dict with keys SR, FASR, IWSR, RG, IHR, ER, PAS,
# CoT, and p_valid_k_flat. The label string is also parsed by
# evaluate_profile_conditions to detect which metric abbreviations appear in it,
# so the condition can be marked "unavailable" rather than "missing" when a
# metric is NaN. Optional conditions do not affect the exact-match decision.
#
# Profile taxonomy:
#   Genuine Planner    — high SR + high FASR + low RG + high IHR + high CoT
#   Efficient Corrector — high SR + low FASR but IWSR ≈ SR (fast repairs)
#   Stochastic Searcher — high SR + low FASR + high RG (retries, flat P(Valid|k))
#   Lucky Retriever    — high SR + high FASR but low IHR + low CoT
#   Understander       — low SR but high ER + high IHR + high PAS
#   Vocabulary-Only    — low SR + high IHR + mid ER + low PAS
#   No Grounding       — near-zero SR + low IHR + low ER + low PAS
PROFILE_DEFINITIONS = [
    {
        "name": "Genuine Planner",
        "interpretation": "Model reads the domain schema, solves on the first attempt, and the reasoning trace is connected to the plan.",
        "key_reference": "Strongest claim",
        "conditions": [
            ("SR > 0.70", lambda m: m["SR"] > 0.70),
            ("FASR > 0.50", lambda m: m["FASR"] > 0.50),
            ("RG < 0.10", lambda m: m["RG"] < 0.10),
            ("IHR > 0.90", lambda m: m["IHR"] > 0.90),
            ("CoT > 0.70", lambda m: m["CoT"] > 0.70),
        ],
    },
    {
        "name": "Efficient Corrector",
        "interpretation": "Model rarely solves first, but successful repairs arrive early enough that IWSR remains close to SR.",
        "key_reference": "Partially consistent with Stechly et al. (2023); verify via P(Valid|k)",
        "conditions": [
            ("SR > 0.50", lambda m: m["SR"] > 0.50),
            ("FASR < 0.30", lambda m: m["FASR"] < 0.30),
            ("IWSR close to SR", lambda m: abs(m["SR"] - m["IWSR"]) <= 0.10),
            ("0.10 <= RG <= 0.30", lambda m: 0.10 <= m["RG"] <= 0.30),
        ],
    },
    {
        "name": "Stochastic Searcher",
        "interpretation": "Reported success is heavily inflated by retries; attempts look like repeated samples rather than correction.",
        "key_reference": "Stechly et al. (2023)",
        "conditions": [
            ("SR >= 0.30", lambda m: m["SR"] >= 0.30),
            ("FASR < 0.20", lambda m: m["FASR"] < 0.20),
            ("RG > 0.30", lambda m: m["RG"] > 0.30),
        ],
        "optional_conditions": [
            ("P(Valid|k) flat", lambda m: m.get("p_valid_k_flat") is True),
        ],
    },
    {
        "name": "Lucky Retriever",
        "interpretation": "Model appears to solve from surface patterns, with weak schema grounding and post-hoc reasoning.",
        "key_reference": "Kambhampati (2024); Valmeekam et al. (2023) obfuscation test",
        "conditions": [
            ("SR > 0.60", lambda m: m["SR"] > 0.60),
            ("FASR > 0.50", lambda m: m["FASR"] > 0.50),
            ("IHR < 0.30", lambda m: m["IHR"] < 0.30),
            ("CoT < 0.40", lambda m: m["CoT"] < 0.40),
        ],
    },
    {
        "name": "Understander",
        "interpretation": "Model understands vocabulary and local preconditions but fails at long-horizon sequencing.",
        "key_reference": "Valmeekam et al. (2023) Phase 2; Dziri et al. (2023)",
        "conditions": [
            ("SR < 0.20", lambda m: m["SR"] < 0.20),
            ("FASR < 0.10", lambda m: m["FASR"] < 0.10),
            ("ER > 0.70", lambda m: m["ER"] > 0.70),
            ("IHR > 0.90", lambda m: m["IHR"] > 0.90),
            ("PAS > 0.70", lambda m: m["PAS"] > 0.70),
        ],
    },
    {
        "name": "Vocabulary-Only",
        "interpretation": "Model uses legal action names but lacks a coherent model of state transitions.",
        "key_reference": "Kambhampati (2024) Phase 1/2; Vafa et al. (2024)",
        "conditions": [
            ("SR < 0.15", lambda m: m["SR"] < 0.15),
            ("IHR > 0.90", lambda m: m["IHR"] > 0.90),
            ("0.30 <= ER <= 0.70", lambda m: 0.30 <= m["ER"] <= 0.70),
            ("PAS < 0.30", lambda m: m["PAS"] < 0.30),
        ],
    },
    {
        "name": "No Grounding",
        "interpretation": "Model is not reading the formal domain; actions, states, and failures are ungrounded.",
        "key_reference": "Valmeekam et al. (2023); McCoy et al. (2024)",
        "conditions": [
            ("SR ~= 0", lambda m: m["SR"] <= 0.05),
            ("IHR < 0.60", lambda m: m["IHR"] < 0.60),
            ("ER < 0.30", lambda m: m["ER"] < 0.30),
            ("PAS < 0.30", lambda m: m["PAS"] < 0.30),
        ],
    },
]


def evaluate_profile_conditions(metrics: dict[str, float], profile: dict[str, Any]) -> dict[str, Any]:
    """Evaluate all threshold conditions for one capability profile against a model's metrics.

    Metric correlation: every profile condition references a subset of
    {SR, FASR, IWSR, RG, IHR, CoT, ER, PAS, p_valid_k_flat} as defined in
    PROFILE_DEFINITIONS.
    Rationale: separating condition evaluation from profile assignment allows the
    JSON report to expose the full condition breakdown (matched / missing /
    unavailable) so users can see how close a model is to each profile boundary,
    not just the binary pass/fail.
    Code purpose: returns a dict with matched, missing, and unavailable condition
    labels plus a boolean ``exact`` flag (all required conditions matched with no
    unavailable ones). Called for every profile definition in
    ``classify_capability_profile``.
    Detail: a condition is "unavailable" if any metric it references is NaN
    (e.g. CoT alignment is NaN when CoT was not enabled); unavailable conditions
    do not count toward missing, preventing a model from being wrongly classified
    as missing a condition it simply cannot be evaluated on. Optional conditions
    (e.g. flat P(Valid|k)) are reported separately and do not affect exactness.
    """
    matched: list[str] = []
    missing: list[str] = []
    unavailable: list[str] = []
    for label, predicate in profile["conditions"]:
        try:
            result = bool(predicate(metrics))
        except Exception:
            result = False
        metric_values = re.findall(r"\b(?:SR|FASR|RG|IHR|CoT|IWSR|ER|PAS)\b", label)
        if any(not math.isfinite(metrics.get(metric, float("nan"))) for metric in metric_values):
            unavailable.append(label)
        elif result:
            matched.append(label)
        else:
            missing.append(label)

    optional = []
    for label, predicate in profile.get("optional_conditions", []):
        try:
            value = predicate(metrics)
        except Exception:
            value = False
        optional.append({"condition": label, "matched": bool(value) if value is not None else None})

    return {
        "profile": profile["name"],
        "matched_conditions": matched,
        "missing_conditions": missing,
        "unavailable_conditions": unavailable,
        "optional_conditions": optional,
        "match_count": len(matched),
        "required_count": len(profile["conditions"]),
        "exact": len(missing) == 0 and len(unavailable) == 0,
    }


def classify_capability_profile(overall_row: dict[str, Any], iteration_profile: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """Assign a capability profile label to a model and return full reasoning notes.

    Metric correlation: all threshold metrics from PROFILE_DEFINITIONS:
    SR, FASR, IWSR, RG (Retry Gap), IHR, ER (Exec), PAS, CoT (alignment),
    and the optional p_valid_k_flat flag from the iteration profile.
    Rationale: the seven profiles (Genuine Planner, Efficient Corrector, Stochastic
    Searcher, Lucky Retriever, Understander, Vocabulary-Only, No Grounding) are
    grounded in the LLM planning literature (Stechly 2023, Kambhampati 2024,
    Valmeekam 2023, Dziri 2023, McCoy 2024). Each profile has a distinct combination
    of metric thresholds that distinguishes it from the others. The full reference
    table is included in the JSON output under ``possible_profiles_reference`` so
    the reader can verify the classification without consulting the source code.
    Code purpose: called in ``build_model_payloads`` for each model's overall row;
    the returned dict is stored under ``models.<model>.reasoning_notes``.
    Detail: evaluates all PROFILE_DEFINITIONS conditions, selects the first exact
    match; if none, picks the profile with the highest matched-condition count
    (ties broken by fewest required conditions) and labels the model
    "Mixed/Unclassified". The threshold_signature (all metric values used) is
    included for reproducibility.
    """
    metrics = {
        "SR": scalar_float(overall_row.get("Success_Rate")),
        "FASR": scalar_float(overall_row.get("FASR")),
        "IWSR": scalar_float(overall_row.get("IWSR")),
        "RG": scalar_float(overall_row.get("Retry_Gap")),
        "IHR": scalar_float(overall_row.get("IHR")),
        "ER": scalar_float(overall_row.get("Exec")),
        "PAS": scalar_float(overall_row.get("PAS")),
        "CoT": scalar_float(overall_row.get("CoT_Alignment")),
        "p_valid_k_flat": profile_is_flat(iteration_profile or []),
    }

    evaluations = [evaluate_profile_conditions(metrics, profile) for profile in PROFILE_DEFINITIONS]
    exact_profile = next((evaluation for evaluation in evaluations if evaluation["exact"]), None)
    if exact_profile is None:
        best = max(evaluations, key=lambda item: (item["match_count"], -item["required_count"]))
        assigned = "Mixed/Unclassified"
        chosen_profile = next(profile for profile in PROFILE_DEFINITIONS if profile["name"] == best["profile"])
        matched = best["matched_conditions"]
        missing = best["missing_conditions"]
        unavailable = best["unavailable_conditions"]
    else:
        assigned = exact_profile["profile"]
        chosen_profile = next(profile for profile in PROFILE_DEFINITIONS if profile["name"] == assigned)
        matched = exact_profile["matched_conditions"]
        missing = exact_profile["missing_conditions"]
        unavailable = exact_profile["unavailable_conditions"]
        best = exact_profile

    return {
        "assigned_profile": assigned,
        "closest_profile": best["profile"],
        "threshold_signature": json_safe(metrics),
        "matched_conditions": matched,
        "missing_conditions": missing,
        "unavailable_conditions": unavailable,
        "interpretation": chosen_profile["interpretation"],
        "key_reference": chosen_profile["key_reference"],
        "possible_profiles_reference": [
            {
                "profile": profile["name"],
                "conditions": [label for label, _ in profile["conditions"]],
                "interpretation": profile["interpretation"],
                "key_reference": profile["key_reference"],
            }
            for profile in PROFILE_DEFINITIONS
        ],
    }


def build_model_payloads(df_metrics: pd.DataFrame, tables: dict[str, pd.DataFrame], saved_plots: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the per-model section of the JSON report.

    Metric correlation: all metrics — assembles row-level metrics, aggregate tables,
    plot references, capability profile classification, and iteration profile into
    one dict per model.
    Rationale: the JSON report is structured around models as the primary key so
    that a consumer (notebook, dashboard, API) can extract everything about a single
    model in O(1) without scanning the entire report. Each model payload is self-
    contained: it includes its own row_metrics table, all relevant aggregate slices,
    the paths to its plots, and the full profile reasoning chain.
    Code purpose: called in ``build_report`` after plots are generated; uses the
    ``saved_plots`` list to build a ``plot_index`` mapping models to their plots.
    Detail: for each model, slices all aggregate tables to model-specific rows,
    computes the iteration profile (not stored globally to save memory), then calls
    ``classify_capability_profile`` with the model's overall row and iteration
    profile. Private columns (_actions, _cot_text, _file_path) are excluded from
    ``row_metrics`` via ``ROW_METRIC_COLUMNS`` filtering.
    """
    models: dict[str, Any] = {}
    if df_metrics.empty:
        return models

    plot_index = defaultdict(list)
    for plot in saved_plots:
        related_models = plot.get("related_models") or []
        for model in related_models:
            plot_index[model].append(plot)

    for model in sorted(df_metrics["Model"].dropna().unique()):
        model_df = df_metrics[df_metrics["Model"] == model].copy()
        model_tables: dict[str, Any] = {}
        for table_name, table_df in tables.items():
            if table_df.empty or "Model" not in table_df.columns:
                model_tables[table_name] = []
                continue
            model_tables[table_name] = records_from_df(table_df[table_df["Model"] == model].copy())

        overall_records = model_tables.get("overall", [])
        overall_row = overall_records[0] if overall_records else {}
        iteration_profile = compute_iteration_profile(model_df)

        row_metric_df = model_df[[column for column in ROW_METRIC_COLUMNS if column in model_df.columns]].copy()
        models[model] = {
            "row_metrics": records_from_df(row_metric_df),
            "tables": model_tables,
            "plots": json_safe(plot_index.get(model, [])),
            "reasoning_notes": classify_capability_profile(overall_row, iteration_profile),
        }
        models[model]["tables"]["iteration_profile"] = json_safe(iteration_profile)

    return models


PLOT_DESCRIPTIONS = {
    "hallucination_heatmap": "Mean hallucination rate by model and domain. Lower is better.",
    "hallucination_by_model_domain": "Action hallucination rate by model and domain.",
    "object_hallucination_by_model_domain": "Object hallucination rate by model and domain.",
    "executability_by_model_domain": "Executability ratio distribution by model and domain.",
    "failure_type_breakdown": "Sequencing errors versus state fabrications per model.",
    "executability_vs_length": "Executability ratio against plan length, faceted by domain.",
    "temporal_distance_by_model": "Mean temporal distance for sequencing errors per model.",
    "cot_alignment_by_model_domain": "Mean CoT plan alignment score by model and domain.",
    "cot_success_rate": "Success rate split by CoT flag.",
    "cot_alignment_validity": "CoT plan alignment distribution for valid versus invalid plans.",
    "fasr_by_model_domain": "First-attempt success rate by model and domain.",
    "sr_vs_fasr": "Overall success rate compared with first-attempt success rate.",
    "fasr_by_difficulty": "First-attempt success rate by difficulty tier.",
    "iwsr_by_model_domain": "Iteration-weighted success rate by model and domain.",
    "sr_fasr_iwsr_by_model": "SR, FASR, and IWSR comparison per model.",
    "retry_gap_by_model": "SR minus FASR; higher values mean stronger retry dependence.",
    "fasr_iwsr_scatter": "FASR versus IWSR with success rate encoded by dot size and color.",
    "success_rate_heatmap": "Success rate heatmap by model and domain.",
    "failure_mode_taxonomy": "Failure taxonomy by hallucination rate and PAS.",
    "composite_scores": "Composite Planning Score by model.",
    "p_valid_given_k": "P(Valid | reached attempt k) iteration profile — discriminates Stochastic Searcher (flat line) from Efficient Corrector (rising curve).",
    "domain_ranking_heatmap": "Normalised within-domain rank heatmap (0=best, 1=worst) for all RANK_METRICS across all domains.",
    "rank_variance": "Mean rank standard deviation across domains per model — high variance signals domain specialisation.",
    "domain_correlation": "Spearman correlation matrix of model success rates across domains — detects redundancy or complementarity.",
    "ps_by_domain_stacked": "Composite Planning Score stacked by domain contribution — reveals whether high PS is broad or concentrated.",
    "metrics_summary_table": "Tabular summary of all key aggregate metrics rendered as a matplotlib figure for archiving.",
    "radar_chart": "Polar radar chart of FASR, IWSR, Exec, IHR, and PAS — larger filled area means stronger overall planning capability.",
}


def maybe_make_plots(
    df_metrics: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    show_plots: bool,
    save_plots: bool,
    plots_dir: Path,
    warnings_out: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate all benchmark visualisation plots and optionally save / display them.

    Metric correlation: all 27 plots cover every computed metric — hallucination
    (strict, fuzzy, object), executability, PAS, temporal distance, CoT plan alignment,
    FASR, IWSR, SR, Retry Gap, PS, within-domain rank, cross-model correlation,
    and the iteration profile.
    Rationale: generating plots inside the main script rather than a notebook
    ensures they are reproducible from the command line and stored alongside the
    JSON report. Each plot is registered with ``register(name, fig, models)`` so
    its path, title, and description are embedded in the JSON report under each
    model's ``plots`` list.
    Code purpose: single function that produces all figures using seaborn / matplotlib,
    handles conditional generation (only if show or save is requested), records
    paths, and closes figures to free memory. Returns a list of plot descriptor
    dicts for ``build_model_payloads``.
    Detail: ``model_palette`` assigns a stable colour per model from the tab20
    palette so the same model always has the same colour across all plots.
    ``register()`` appends to the local ``figures`` list; the save loop at the end
    iterates it once. Figures with insufficient data (all-NaN columns, empty tables)
    are skipped via guard conditions before the plt.subplots call.
    """
    if not show_plots and not save_plots:
        return []
    if df_metrics.empty:
        add_warning(warnings_out, "plot_skipped", "No data available for plotting.")
        return []

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except Exception as exc:
        add_warning(warnings_out, "plot_import_error", str(exc))
        return []

    sns.set_theme(style="whitegrid", palette="tab10")
    saved: list[dict[str, Any]] = []
    figures: list[tuple[str, Any, list[str]]] = []
    all_models = sorted(df_metrics["Model"].dropna().unique())
    base_colors = sns.color_palette("tab20", n_colors=max(len(all_models), 1))
    model_palette = {model: base_colors[index] for index, model in enumerate(all_models)}

    def register(name: str, fig: Any, related_models: Optional[list[str]] = None) -> None:
        figures.append((name, fig, related_models or all_models))

    if not df_metrics["hallucination_rate"].isna().all():
        pivot = df_metrics.pivot_table(values="hallucination_rate", index="Model", columns="Domain", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.4), max(4, len(pivot.index) * 0.5 + 1)))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, vmin=0, vmax=1, ax=ax)
        ax.set_title("Mean Hallucination Rate (Model x Domain)")
        register("hallucination_heatmap", fig)

        agg = df_metrics.groupby(["Model", "Domain"])["hallucination_rate"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=agg, x="Model", y="hallucination_rate", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("Hallucination Rate by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("hallucination_by_model_domain", fig)

        agg_obj = df_metrics.groupby(["Model", "Domain"])["object_hallucination_rate"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=agg_obj, x="Model", y="object_hallucination_rate", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("Object Hallucination Rate by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("object_hallucination_by_model_domain", fig)

    if not df_metrics["executability_ratio"].isna().all():
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.boxplot(data=df_metrics.dropna(subset=["executability_ratio"]), x="Model", y="executability_ratio", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("Executability Ratio by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("executability_by_model_domain", fig)

        fig, ax = plt.subplots(figsize=(8, 5))
        failure = tables["failure_breakdown"].set_index("Model")[["sequencing_error_proportion", "state_fabrication_proportion"]]
        failure.plot(kind="bar", stacked=True, ax=ax, color=["steelblue", "tomato"], edgecolor="white")
        ax.set_title("Failure Type Breakdown per Model")
        ax.tick_params(axis="x", rotation=30)
        register("failure_type_breakdown", fig)

        domains = sorted(df_metrics["Domain"].dropna().unique())
        fig, axes = plt.subplots(max(1, len(domains)), 1, figsize=(6, max(4, 4 * len(domains))), squeeze=False)
        for ax, domain in zip(axes[:, 0], domains):
            sub = df_metrics[df_metrics["Domain"] == domain].dropna(subset=["executability_ratio", "Length"])
            for model, group in sub.groupby("Model"):
                ax.scatter(group["Length"], group["executability_ratio"], label=model, alpha=0.65, s=30, color=model_palette.get(model, "gray"))
            ax.set_title(f"Domain: {domain}")
            ax.set_xlabel("Plan Length")
            ax.set_ylabel("Executability Ratio")
            ax.legend(fontsize=7)
        fig.suptitle("Executability Ratio vs Plan Length")
        fig.tight_layout()
        register("executability_vs_length", fig)

        temporal = df_metrics.dropna(subset=["mean_temporal_distance"])
        if not temporal.empty:
            models_td = sorted(temporal["Model"].unique())
            fig, axes = plt.subplots(max(1, len(models_td)), 1, figsize=(6, max(4, 3.5 * len(models_td))), squeeze=False)
            for ax, model in zip(axes[:, 0], models_td):
                values = temporal[temporal["Model"] == model]["mean_temporal_distance"]
                ax.hist(values.dropna(), bins=10, color=model_palette.get(model, "steelblue"), edgecolor="white", alpha=0.8)
                ax.set_title(model)
                ax.set_xlabel("Mean Temporal Distance")
                ax.set_ylabel("Count")
            fig.suptitle("Temporal Distance for Sequencing Errors")
            fig.tight_layout()
            register("temporal_distance_by_model", fig, models_td)

    if not df_metrics["cot_plan_alignment_score"].isna().all():
        cot_sub = df_metrics.dropna(subset=["cot_plan_alignment_score"])
        agg = cot_sub.groupby(["Model", "Domain"])["cot_plan_alignment_score"].mean().reset_index()
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=agg, x="Model", y="cot_plan_alignment_score", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("Mean CoT Plan Alignment Score by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("cot_alignment_by_model_domain", fig)

        cot_sr = df_metrics.copy()
        cot_sr["_cot_flag"] = cot_sr["Chain_of_Thought"].apply(lambda value: str(value).lower() in {"true", "1", "yes"})
        cot_sr = cot_sr.groupby(["Model", "_cot_flag"])["Valid"].mean().reset_index()
        cot_sr["CoT"] = cot_sr["_cot_flag"].map({True: "CoT=True", False: "CoT=False"})
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=cot_sr, x="Model", y="Valid", hue="CoT", ax=ax, palette="Set1")
        ax.set_title("Success Rate: CoT=True vs CoT=False")
        ax.tick_params(axis="x", rotation=30)
        register("cot_success_rate", fig)

        cot_valid = cot_sub.copy()
        cot_valid["Validity"] = cot_valid["Valid"].map({True: "Valid", False: "Invalid"})
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.violinplot(data=cot_valid, x="Model", y="cot_plan_alignment_score", hue="Validity", ax=ax, palette="Set1", inner="quart", dodge=True, cut=0, bw_adjust=0.8)
        ax.set_title("CoT Plan Alignment Distribution: Valid vs Invalid Plans")
        ax.tick_params(axis="x", rotation=30)
        register("cot_alignment_validity", fig)

    fasr_agg = tables["by_domain"][["Model", "Domain", "FASR", "Success_Rate", "IWSR"]]
    if not fasr_agg.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=fasr_agg, x="Model", y="FASR", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("First-Attempt Success Rate by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("fasr_by_model_domain", fig)

        model_rates = tables["overall"][["Model", "Success_Rate", "FASR"]]
        x_pos = np.arange(len(model_rates))
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(x_pos - 0.18, model_rates["Success_Rate"], 0.36, label="Success Rate", color="steelblue")
        ax.bar(x_pos + 0.18, model_rates["FASR"], 0.36, label="FASR", color="tomato")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(model_rates["Model"], rotation=30)
        ax.set_title("Success Rate vs FASR")
        ax.legend()
        register("sr_vs_fasr", fig)

        diff_table = tables["by_difficulty"].copy()
        diff_table["Difficulty"] = pd.Categorical(diff_table["Difficulty"], categories=DIFF_ORDER, ordered=True)
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=diff_table, x="Difficulty", y="FASR", hue="Model", ax=ax, palette=model_palette, order=DIFF_ORDER)
        ax.set_title("FASR by Difficulty")
        register("fasr_by_difficulty", fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=fasr_agg, x="Model", y="IWSR", hue="Domain", ax=ax, palette="Set2")
        ax.set_title("IWSR by Model and Domain")
        ax.tick_params(axis="x", rotation=30)
        register("iwsr_by_model_domain", fig)

        compare = tables["overall"][["Model", "Success_Rate", "FASR", "IWSR"]]
        x_pos = np.arange(len(compare))
        fig, ax = plt.subplots(figsize=(9, 5))
        width = 0.25
        ax.bar(x_pos - width, compare["Success_Rate"], width, label="Success Rate", color="steelblue")
        ax.bar(x_pos, compare["FASR"], width, label="FASR", color="tomato")
        ax.bar(x_pos + width, compare["IWSR"], width, label="IWSR", color="seagreen")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(compare["Model"], rotation=30)
        ax.set_title("SR vs FASR vs IWSR per Model")
        ax.legend()
        register("sr_fasr_iwsr_by_model", fig)

        rg = tables["retry_gap"].copy()
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["tomato" if value > 0 else "steelblue" for value in rg["Retry_Gap"]]
        ax.barh(rg["Model"], rg["Retry_Gap"], color=colors, edgecolor="white")
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title("Retry Gap (SR - FASR) per Model")
        register("retry_gap_by_model", fig)

        scatter = tables["overall"][["Model", "FASR", "IWSR", "Success_Rate"]]
        fig, ax = plt.subplots(figsize=(8, 6))
        points = ax.scatter(scatter["FASR"], scatter["IWSR"], s=scatter["Success_Rate"] * 1500 + 50, c=scatter["Success_Rate"], cmap="RdYlGn", vmin=0, vmax=1, edgecolors="gray")
        for _, row in scatter.iterrows():
            ax.annotate(row["Model"], (row["FASR"], row["IWSR"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
        fig.colorbar(points, ax=ax, label="Overall Success Rate")
        ax.set_title("FASR vs IWSR (size = Success Rate)")
        register("fasr_iwsr_scatter", fig)

    sr_pivot = df_metrics.groupby(["Model", "Domain"])["Valid"].mean().unstack("Domain").fillna(0)
    if not sr_pivot.empty:
        fig, ax = plt.subplots(figsize=(max(8, len(sr_pivot.columns) * 1.4), max(4, len(sr_pivot.index) * 0.5 + 1)))
        sns.heatmap(sr_pivot, annot=True, fmt=".2f", cmap="YlGn", linewidths=0.4, vmin=0, vmax=1, ax=ax)
        ax.set_title("Success Rate Heatmap (Model x Domain)")
        register("success_rate_heatmap", fig)

    taxonomy = tables["overall"].copy()
    if not taxonomy.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        points = ax.scatter(taxonomy["Halluc"], taxonomy["PAS"], s=taxonomy["FASR"] * 1200 + 80, c=taxonomy["Success_Rate"], cmap="RdYlGn", vmin=0, vmax=1, edgecolors="dimgray")
        for _, row in taxonomy.iterrows():
            ax.annotate(row["Model"], (row["Halluc"], row["PAS"]), textcoords="offset points", xytext=(8, 4), fontsize=8)
        fig.colorbar(points, ax=ax, label="Overall Success Rate")
        ax.set_xlabel("Hallucination Rate")
        ax.set_ylabel("Precondition Awareness Score")
        ax.set_title("Failure Mode Taxonomy")
        register("failure_mode_taxonomy", fig)

    composite = tables["composite_score"]
    if not composite.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ordered = composite.sort_values("PS_overall", ascending=True)
        ax.barh(ordered["Model"], ordered["PS_overall"], xerr=[ordered["err_lo"].clip(lower=0), ordered["err_hi"].clip(lower=0)], color="steelblue", edgecolor="white")
        ax.set_title("Composite Planning Score by Model")
        ax.set_xlabel("PS")
        register("composite_scores", fig)

    # ── Plot 21: P(Valid | reached attempt k) ─────────────────────────────────────
    # Metric: Iteration profile — P(valid | reached attempt k) for k = 1…max_iter.
    # Rationale: A Stochastic Searcher produces a flat curve (each retry is an
    # independent sample, success probability stays constant). An Efficient Corrector
    # shows a rising curve (model leverages feedback, later attempts have higher
    # conditional success). This plot is the primary discriminator between these two
    # profiles (Stechly et al. 2023). The 0.5 dashed baseline marks the inflection
    # point where retries become "more likely to succeed than fail".
    # Data source: compute_iteration_profile() applied per-model to df_metrics.
    if not df_metrics[["Iterations", "Valid"]].dropna().empty:
        _iter_rows: list[dict] = []
        for _model in all_models:
            for _rec in compute_iteration_profile(df_metrics[df_metrics["Model"] == _model]):
                _p = _rec["p_valid_given_reached"]
                if math.isfinite(float(_p)):
                    _iter_rows.append({"Model": _model, "k": _rec["iteration"], "P(Valid|k)": float(_p)})
        _iter_df = pd.DataFrame(_iter_rows)
        if not _iter_df.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for _model in all_models:
                _sub = _iter_df[_iter_df["Model"] == _model].sort_values("k")
                if not _sub.empty:
                    ax.plot(_sub["k"], _sub["P(Valid|k)"], marker="o", linewidth=2, label=_model, color=model_palette.get(_model))
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
            ax.set_title("P(Valid | reached attempt k) — Iteration Profile")
            ax.set_xlabel("Attempt k")
            ax.set_ylabel("P(Valid | reached k)")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=8)
            register("p_valid_given_k", fig)

    # ── Plots 22–23: Within-domain ranking heatmap and rank variance ───────────────
    # Metric: RANK_METRICS — one rank per model per metric per domain.
    # Rationale: Absolute metric values are hard to compare across domains with
    # different difficulty distributions. Ranking within each domain removes this
    # bias and shows each model's relative standing among its peers on each metric.
    # The heatmap normalises ranks to [0,1] (0=best, RdYlGn_r → green=best) so
    # all subplots share the same colour scale regardless of number of models.
    # rank_variance computes the mean std of normalised rank across domains:
    # a low std means the model ranks similarly everywhere (generalist); a high std
    # means it dominates some domains and underperforms in others (specialist).
    # Data source: tables["rank_within_domain"]
    _rank_df = tables.get("rank_within_domain", pd.DataFrame())
    _rank_metrics_cols = [m for m in RANK_METRICS if not _rank_df.empty and m in _rank_df.columns]
    if not _rank_df.empty and _rank_metrics_cols:
        _domains_ranked = sorted(_rank_df["Domain"].dropna().unique())
        _n_dom = len(_domains_ranked)
        _n_cols_grid = min(_n_dom, 3)
        _n_rows_grid = math.ceil(_n_dom / _n_cols_grid)
        _fig_w = max(6, 4.5 * _n_cols_grid)
        _fig_h = max(4, (len(all_models) * 0.8 + 2.0) * _n_rows_grid)
        fig, axes = plt.subplots(_n_rows_grid, _n_cols_grid, figsize=(_fig_w, _fig_h), squeeze=False)
        for _idx, _dom in enumerate(_domains_ranked):
            _ri, _ci = divmod(_idx, _n_cols_grid)
            _ax = axes[_ri][_ci]
            _dom_df = _rank_df[_rank_df["Domain"] == _dom].set_index("Model")
            _pivot = _dom_df[[c for c in _rank_metrics_cols if c in _dom_df.columns]]
            _n_mod = len(_pivot)
            _pivot_norm = (_pivot - 1) / max(_n_mod - 1, 1) if _n_mod > 1 else _pivot.clip(0, 0)
            sns.heatmap(_pivot_norm, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0, vmax=1,
                        linewidths=0.3, ax=_ax, cbar=False)
            _ax.set_title(_dom, fontsize=9)
            _ax.tick_params(axis="x", rotation=45, labelsize=7)
            _ax.tick_params(axis="y", labelsize=7)
        for _idx in range(_n_dom, _n_rows_grid * _n_cols_grid):
            _ri, _ci = divmod(_idx, _n_cols_grid)
            axes[_ri][_ci].set_visible(False)
        fig.suptitle("Normalised Within-Domain Rank (0 = best, 1 = worst)")
        fig.tight_layout()
        register("domain_ranking_heatmap", fig)

        # Rank variance: std of rank across domains reveals domain specialisation
        _var_df = _rank_df.groupby("Model")[_rank_metrics_cols].std().reset_index()
        _var_df["mean_rank_std"] = _var_df[_rank_metrics_cols].mean(axis=1)
        _var_df = _var_df.sort_values("mean_rank_std", ascending=False)
        fig, ax = plt.subplots(figsize=(9, max(4, len(_var_df) * 0.65 + 1.2)))
        _var_colors = [model_palette.get(m, "steelblue") for m in _var_df["Model"]]
        ax.barh(_var_df["Model"], _var_df["mean_rank_std"], color=_var_colors, edgecolor="white")
        ax.set_title("Rank Variance Across Domains — Higher = More Domain-Specific")
        ax.set_xlabel("Mean Std of Normalised Within-Domain Rank")
        register("rank_variance", fig)

    # ── Plot 24: Spearman correlation of success rates across domains ──────────────
    # Metric: SR (success rate) per model per domain.
    # Rationale: Builds a Model×Model Spearman ρ matrix using each model's vector
    # of domain success rates as its "signature". High positive ρ means two models
    # succeed on exactly the same domains (they share the same capability frontier).
    # Near-zero or negative ρ suggests orthogonal capability profiles — one model
    # might be an expert where the other fails, pointing toward ensemble utility.
    # Also reveals whether domain difficulty is model-independent (all models
    # correlate strongly) or model-specific (low cross-model correlation).
    # Data source: df_metrics pivot Model×Domain → scipy/pandas Spearman corr.
    _sr_pivot = df_metrics.groupby(["Model", "Domain"])["Valid"].mean().unstack("Domain").fillna(0)
    if _sr_pivot.shape[0] > 1:
        _corr = _sr_pivot.T.corr(method="spearman")
        fig, ax = plt.subplots(figsize=(max(6, len(_corr) * 1.1), max(5, len(_corr) * 1.1)))
        sns.heatmap(_corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1,
                    linewidths=0.4, ax=ax)
        ax.set_title("Spearman Correlation of Model Success Rates Across Domains")
        register("domain_correlation", fig)

    # ── Plot 25: PS stacked by domain ─────────────────────────────────────────────
    # Metric: Composite Planning Score (PS) per model per domain.
    # Rationale: A high overall PS can hide an uneven domain distribution. If one
    # model scores 0.9 on one domain and 0.1 on four others, its overall PS is
    # similar to a model that scores 0.5 everywhere. The stacked bar makes this
    # visible: an even stack = generalist; a dominant single segment = specialist.
    # Complements the heatmap by giving a cumulative view rather than a comparative
    # one, and allows reading the "total PS budget" each model earns across domains.
    # Data source: tables["by_domain"]["PS"] pivoted Model×Domain.
    _by_domain_tbl = tables.get("by_domain", pd.DataFrame())
    if not _by_domain_tbl.empty and "PS" in _by_domain_tbl.columns:
        _pivot_ps = _by_domain_tbl.pivot_table(values="PS", index="Model", columns="Domain", fill_value=0)
        if not _pivot_ps.empty:
            fig, ax = plt.subplots(figsize=(max(8, len(_pivot_ps.columns) * 1.4), 6))
            _pivot_ps.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
            ax.set_title("Composite Planning Score — Stacked by Domain")
            ax.set_ylabel("Cumulative PS")
            ax.tick_params(axis="x", rotation=30)
            ax.legend(title="Domain", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
            fig.tight_layout()
            register("ps_by_domain_stacked", fig)

    # ── Plot 26: Metrics summary table ────────────────────────────────────────────
    # Metric: All major aggregate metrics from tables["overall"].
    # Rationale: A matplotlib table figure is captured alongside the graphical plots
    # in the plot archive, making it easy to include in a PDF report or share as a
    # PNG without exporting the DataFrame separately. Values are formatted to 3 d.p.;
    # NaN entries are shown as "—" to indicate metric unavailability (e.g. CoT
    # alignment is NaN for protocols without chain-of-thought).
    # Data source: tables["overall"], columns [Model, SR, FASR, IWSR, Exec, IHR,
    # PAS, CoT_Alignment, Retry_Gap, PS].
    _overall_tbl = tables.get("overall", pd.DataFrame())
    if not _overall_tbl.empty:
        _tbl_cols = ["Model", "Success_Rate", "FASR", "IWSR", "Exec", "IHR", "PAS", "CoT_Alignment", "Retry_Gap", "PS"]
        _tbl_avail = [c for c in _tbl_cols if c in _overall_tbl.columns]
        _tbl_sub = _overall_tbl[_tbl_avail].copy()
        for _col in _tbl_avail:
            if _col != "Model":
                _tbl_sub[_col] = _tbl_sub[_col].apply(lambda v: f"{v:.3f}" if pd.notna(v) else "—")
        _fig_w = max(10, len(_tbl_avail) * 1.4)
        _fig_h = max(2.5, len(_tbl_sub) * 0.55 + 1.5)
        fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))
        ax.axis("off")
        _tbl_obj = ax.table(cellText=_tbl_sub.values, colLabels=_tbl_sub.columns,
                            loc="center", cellLoc="center")
        _tbl_obj.auto_set_font_size(False)
        _tbl_obj.set_fontsize(8)
        _tbl_obj.auto_set_column_width(col=list(range(len(_tbl_avail))))
        ax.set_title("Key Metrics Summary", pad=14, fontsize=10)
        register("metrics_summary_table", fig)

    # ── Plot 27: Capability radar chart ───────────────────────────────────────────
    # Metrics: FASR, IWSR, Exec (executability ratio), IHR (inverse hallucination),
    # PAS (precondition awareness) — all already in [0, 1].
    # Rationale: The pentagon shape encodes the five orthogonal capability axes
    # simultaneously. A Genuine Planner fills most of the pentagon; a No Grounding
    # model stays near the origin; a Vocabulary-Only model shows high IHR but low
    # PAS and Exec. The radar makes the profile signature visually immediate,
    # complementing the numerical PROFILE_DEFINITIONS threshold table.
    # Note: matplotlib polar axes require angles in radians; angles list is closed
    # by appending the first element so the polygon outline completes.
    # Data source: tables["overall"].
    if not _overall_tbl.empty:
        _radar_metrics = ["FASR", "IWSR", "Exec", "IHR", "PAS"]
        _avail_radar = [m for m in _radar_metrics if m in _overall_tbl.columns]
        if len(_avail_radar) >= 3:
            _N = len(_avail_radar)
            _angles = [n / float(_N) * 2 * math.pi for n in range(_N)]
            _angles += _angles[:1]
            fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})
            ax.set_theta_offset(math.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_xticks(_angles[:-1])
            ax.set_xticklabels(_avail_radar, fontsize=9)
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=7)
            for _model in all_models:
                _mrow = _overall_tbl[_overall_tbl["Model"] == _model]
                if _mrow.empty:
                    continue
                _vals = [float(_mrow.iloc[0].get(m, 0) or 0) for m in _avail_radar]
                _vals += _vals[:1]
                ax.plot(_angles, _vals, linewidth=1.8, label=_model, color=model_palette.get(_model))
                ax.fill(_angles, _vals, alpha=0.07, color=model_palette.get(_model))
            ax.legend(loc="upper right", bbox_to_anchor=(1.38, 1.18), fontsize=8)
            ax.set_title("Capability Radar: FASR / IWSR / Exec / IHR / PAS", pad=22)
            register("radar_chart", fig)

    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)

    for name, fig, related_models in figures:
        description = PLOT_DESCRIPTIONS.get(name, "")
        print(f"\nGrafico: {name}")
        if description:
            print(f"  {description}")
        path = None
        if save_plots:
            path = plots_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
        saved.append(
            {
                "name": name,
                "title": fig.axes[0].get_title() if fig.axes else name,
                "description": description,
                "path": str(path) if path else None,
                "related_models": related_models,
            }
        )

    if show_plots:
        try:
            plt.show()
        except Exception as exc:
            add_warning(warnings_out, "plot_show_error", str(exc))
    for _, fig, _ in figures:
        plt.close(fig)

    return saved


def build_report(
    framework_root: Path,
    selected_run_ids: list[str],
    merged: bool,
    show_plots: bool,
    save_plots: bool,
    json_path: Path,
    plots_dir: Path,
    custom_json_name: bool,
    timestamp_iso: str,
    warnings_out: list[dict[str, Any]],
) -> dict[str, Any]:
    """Orchestrate the full evaluation pipeline and return the JSON-serialisable report dict.

    Metric correlation: all metrics — this function wires together every pipeline
    stage: PDDL parsing → artifact loading → row metrics → aggregate tables →
    plots → model payloads → report assembly.
    Rationale: a single entry point makes the pipeline easy to test and makes the
    call graph explicit. The report dict is fully JSON-safe (via ``json_safe``) so
    ``write_report`` needs only ``json.dumps``.
    Code purpose: called by ``main()`` with user-provided parameters; also callable
    from tests or notebooks with mock inputs.
    Detail: ``loaded_data_summary`` is a lightweight header giving row count and
    unique model/domain/protocol/difficulty values, useful for quick validation
    before loading the full model payloads. Metadata includes all paths so the
    report is self-describing and reproducible.
    """
    domain_info, problem_info = index_pddl_context(framework_root / "tasks", warnings_out)
    df_raw = load_artifact_rows(framework_root, selected_run_ids, warnings_out)
    df_metrics = compute_row_metrics(df_raw, domain_info, problem_info, warnings_out)
    tables = build_aggregate_tables(df_metrics)
    ensure_plots_dir(save_plots, plots_dir)
    saved_plots = maybe_make_plots(df_metrics, tables, show_plots, save_plots, plots_dir, warnings_out)
    models = build_model_payloads(df_metrics, tables, saved_plots)

    loaded_summary = {
        "row_count": int(len(df_metrics)),
        "models": sorted(df_metrics["Model"].dropna().unique().tolist()) if not df_metrics.empty else [],
        "domains": sorted(df_metrics["Domain"].dropna().unique().tolist()) if not df_metrics.empty else [],
        "protocols": sorted(df_metrics["Protocol"].dropna().unique().tolist()) if not df_metrics.empty else [],
        "difficulties": sorted(df_metrics["Difficulty"].dropna().unique().tolist()) if not df_metrics.empty else [],
    }

    return json_safe(
        {
            "metadata": {
                "merged": merged,
                "selected_run_ids": selected_run_ids,
                "created_at": timestamp_iso,
                "custom_json_name": custom_json_name,
                "json_output_path": str(json_path),
                "plots_output_dir": str(plots_dir) if save_plots else None,
                "show_plots": show_plots,
                "save_plots": save_plots,
                "framework_root": str(framework_root),
                "outputs_root": str(framework_root / "outputs"),
                "parsed_root": str(framework_root / "outputs" / "parsed"),
                "scored_root": str(framework_root / "outputs" / "scored"),
                "tasks_root": str(framework_root / "tasks"),
            },
            "loaded_data_summary": loaded_summary,
            "models": models,
            "warnings": warnings_out,
        }
    )


def write_report(report: dict[str, Any], json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_plots_dir(save_plots: bool, plots_dir: Path) -> None:
    if save_plots:
        plots_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    """Interactive CLI entry point for the advanced planning evaluation pipeline.

    Code purpose: prompts the user to select run IDs, plot options, and output paths,
    then delegates to ``build_report`` and ``write_report``. Returns 0 on success.
    Detail: all user-facing prompts use ``yes_no_prompt`` and plain ``input()``
    so the CLI is fully non-graphical and scriptable (replace ``input`` with a
    callable in tests).
    """
    warnings_out: list[dict[str, Any]] = []
    framework_root = find_framework_root()
    outputs_root = framework_root / "outputs"
    results_dir = default_results_dir(framework_root)

    run_statuses = collect_run_status(outputs_root)
    print_run_status(run_statuses)
    merged, selected_run_ids, selection_warnings = select_run_ids_interactively(run_statuses)
    warnings_out.extend(selection_warnings)

    show_plots = yes_no_prompt("Vuoi mostrare i grafici alla fine dei calcoli?", default=False)
    save_plots = yes_no_prompt("Vuoi salvare i grafici?", default=False)

    now = datetime.now().astimezone()
    timestamp_file = now.strftime("%Y-%m-%d_%H-%M-%S")
    json_path, plots_dir, custom_json_name, json_name = choose_output_paths_interactively(results_dir, timestamp_file)
    print(f"\nJSON selezionato: {results_dir / json_name}")
    if save_plots:
        print(f"Cartella grafici: {plots_dir}")

    report = build_report(
        framework_root=framework_root,
        selected_run_ids=selected_run_ids,
        merged=merged,
        show_plots=show_plots,
        save_plots=save_plots,
        json_path=json_path,
        plots_dir=plots_dir,
        custom_json_name=custom_json_name,
        timestamp_iso=now.isoformat(timespec="seconds"),
        warnings_out=warnings_out,
    )
    write_report(report, json_path)

    print(f"\nReport JSON salvato in: {json_path}")
    if save_plots:
        print(f"Grafici salvati in: {plots_dir}")
    if warnings_out:
        print(f"Warning registrati: {len(warnings_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


