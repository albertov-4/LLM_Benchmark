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
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd

from reporting.cot_alignment import (
    compute_cot_alignment_for_attempt,
    parsed_plan_raw_actions,
    parsed_plan_reasoning_actions,
    parsed_plan_reasoning_text,
    parse_action,
    raw_attempt_reasoning_text,
    select_cot_alignment_attempt,
)
from reporting.plots import compute_iteration_profile, maybe_make_plots


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
    "is_gen_error_instance",
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
        if answer in {"y", "yes", "s", "si"}:
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

            # pass 1: numeric fluents (note the -? for negative initial values)
            for m in re.finditer(r"\(=\s*\(\s*([a-z][a-z0-9_-]*)([^)]*)\)\s*(-?[\d.]+)\s*\)", init_text):
                key = (m.group(1),) + tuple(m.group(2).strip().split())
                result["init_numeric"][key] = float(m.group(3))

            # pass 2: atoms — strip (= (...) val) first so the inner
            # function term isn't re-matched as a boolean atom
            atoms_text = re.sub(r"\(=\s*\([^)]*\)\s*-?[\d.]+\s*\)", " ", init_text)
            for m in re.finditer(r"\(([a-z][a-z0-9_-]*(?:\s+[a-z0-9][a-z0-9_-]*)*)\)", atoms_text):
                tokens = m.group(1).split()
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

            is_gen_error_instance = bool(scored_attempts) and all(
                (a.get("validation_result") or {}).get("status") == "generation_error"
                for a in scored_attempts
                if isinstance(a, dict)
            )

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
                    "is_gen_error_instance": is_gen_error_instance,
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
                "is_gen_error_instance",
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
    - ``inverse_hallucination_rate`` (IHR = 1 - hallucination_rate): used in
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
        #modified in order to be able to discriminate both a parethesized fluent and a bare numeric token
        operand = r"(?:\([^()]*\)|[^()\s]+)"
        cmp_pattern = r"\((>=|<=|>|<|=)\s*(" + operand + r")\s+(" + operand + r")\s*\)"
        for numeric_match in re.finditer(cmp_pattern, grounded):
            op, lhs_s, rhs_s = numeric_match.group(1), numeric_match.group(2).strip(), numeric_match.group(3).strip()
            lhs = self._eval_num(lhs_s)
            rhs = self._eval_num(rhs_s)
            if lhs is None or rhs is None:
                continue
            ok = {">=": lhs >= rhs, "<=": lhs <= rhs, ">": lhs > rhs, "<": lhs < rhs, "=": lhs == rhs}[op]
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


def profile_is_flat(iteration_profile: list[dict[str, Any]], tolerance: float = 0.10) -> Optional[bool]:
    """Test whether the P(Valid|k) iteration profile is approximately flat.

    Metric correlation: optional condition for the Stochastic Searcher profile.
    Rationale: a flat curve (max - min ≤ tolerance=0.10) indicates the probability
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
    - ``retry_gap``: SR, FASR, IWSR, and RG (= SR - FASR) sorted by RG.
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

    # Exclude instances where every attempt was a generation_error (infrastructure
    # failure, not a model failure) from all metric aggregations.  The rows remain
    # in df_metrics so they appear in row_metrics for diagnostics.
    _gen_err_mask = df_metrics.get("is_gen_error_instance", pd.Series(False, index=df_metrics.index)).fillna(False).astype(bool)
    df_for_agg = df_metrics[~_gen_err_mask].copy()

    gen_error_overall = df_metrics.groupby("Model", dropna=False).agg(
        gen_error_count=("is_gen_error_instance", "sum"),
    ).reset_index()
    gen_error_by_domain = df_metrics.groupby(["Model", "Domain"], dropna=False).agg(
        gen_error_count=("is_gen_error_instance", "sum"),
    ).reset_index()

    overall = df_for_agg.groupby("Model", dropna=False).agg(
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
    overall = overall.merge(gen_error_overall, on="Model", how="left")
    overall["gen_error_count"] = overall["gen_error_count"].fillna(0).astype(int)

    by_domain = df_for_agg.groupby(["Model", "Domain"], dropna=False).agg(
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
    by_domain = by_domain.merge(gen_error_by_domain, on=["Model", "Domain"], how="left")
    by_domain["gen_error_count"] = by_domain["gen_error_count"].fillna(0).astype(int)

    by_difficulty = df_for_agg.groupby(["Model", "Difficulty"], dropna=False).agg(
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

    cot_summary = df_for_agg.groupby(["Model", "Chain_of_Thought"], dropna=False).agg(
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

    failure_breakdown = df_for_agg.groupby("Model", dropna=False).agg(
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
    for model, model_df in df_for_agg.groupby("Model", dropna=False):
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


