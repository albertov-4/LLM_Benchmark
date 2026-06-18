"""Score PDDL planning-domain instances with general structural features.

The script reads a directory shaped as:

    Benchmark_Framework/tasks/
      domain-name/
        domain/domain.pddl
        easy/
          pfile1.pddl
        medium/
          pfile8.pddl
        hard/
          pfile15.pddl

and writes per-instance and per-domain complexity reports.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any


LOGICAL_OPERATORS = {
    "and",
    "or",
    "not",
    "imply",
    "forall",
    "exists",
    "when",
}
NUMERIC_COMPARISONS = {"=", "<", "<=", ">", ">="}
NUMERIC_EFFECTS = {"increase", "decrease", "assign", "scale-up", "scale-down"}
ARITHMETIC_OPERATORS = {"+", "-", "*", "/"}
TOPOLOGY_MARKERS = (
    "door",
    "adj",
    "connected",
    "can-drive",
    "can_traverse",
    "is_next",
    "visible",
    "before",
)


@dataclass(frozen=True)
class PredicateSchema:
    name: str
    parameters: list[tuple[str, str]]


@dataclass(frozen=True)
class FunctionSchema:
    name: str
    parameters: list[tuple[str, str]]


@dataclass(frozen=True)
class ActionSchema:
    name: str
    parameters: list[tuple[str, str]]
    precondition: Any | None
    effect: Any | None


@dataclass
class DomainModel:
    name: str
    type_parents: dict[str, str]
    predicates: dict[str, PredicateSchema]
    functions: dict[str, FunctionSchema]
    actions: list[ActionSchema]
    changed_predicates: set[str]
    numeric_preconditions: int
    numeric_effects: int


@dataclass
class ProblemModel:
    name: str
    domain_name: str
    object_types: dict[str, str]
    init_facts: set[tuple[str, ...]]
    numeric_init: dict[tuple[str, ...], float]
    goal: Any | None
    has_metric: bool


def remove_comments(text: str) -> str:
    return re.sub(r";.*", "", text)


def tokenize(text: str) -> list[str]:
    return re.findall(r"\(|\)|[^\s()]+", remove_comments(text))


def parse_sexpr(text: str) -> Any:
    tokens = tokenize(text)
    stack: list[list[Any]] = []
    root: list[Any] = []
    current = root

    for token in tokens:
        if token == "(":
            new_list: list[Any] = []
            current.append(new_list)
            stack.append(current)
            current = new_list
        elif token == ")":
            if not stack:
                raise ValueError("Unexpected closing parenthesis")
            current = stack.pop()
        else:
            current.append(token)

    if stack:
        raise ValueError("Unclosed parenthesis in PDDL")
    if len(root) != 1:
        raise ValueError("Expected one top-level PDDL expression")
    return root[0]


def atom_lower(value: Any) -> Any:
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, list):
        return [atom_lower(item) for item in value]
    return value


def section_name(section: Any) -> str | None:
    if isinstance(section, list) and section and isinstance(section[0], str):
        return section[0].lower()
    return None


def find_section(form: list[Any], name: str) -> list[Any] | None:
    target = name.lower()
    for section in form:
        if section_name(section) == target:
            return section
    return None


def find_all_sections(form: list[Any], name: str) -> list[list[Any]]:
    target = name.lower()
    return [section for section in form if section_name(section) == target]


def parse_typed_list(items: list[Any], default_type: str = "object") -> list[tuple[str, str]]:
    result: list[tuple[str, str]] = []
    pending: list[str] = []
    index = 0

    while index < len(items):
        token = items[index]
        if isinstance(token, list):
            index += 1
            continue

        if token == "-":
            assigned_type = str(items[index + 1]).lower() if index + 1 < len(items) else default_type
            result.extend((name, assigned_type) for name in pending)
            pending = []
            index += 2
        else:
            pending.append(str(token))
            index += 1

    result.extend((name, default_type) for name in pending)
    return result


def parse_type_parents(section: list[Any] | None) -> dict[str, str]:
    parents = {"object": ""}
    if not section:
        return parents

    for type_name, parent in parse_typed_list(section[1:], default_type="object"):
        parents[type_name] = parent
    return parents


def parse_predicates(section: list[Any] | None) -> dict[str, PredicateSchema]:
    predicates: dict[str, PredicateSchema] = {}
    if not section:
        return predicates

    for declaration in section[1:]:
        if not isinstance(declaration, list) or not declaration:
            continue
        name = str(declaration[0]).lower()
        parameters = parse_typed_list(declaration[1:])
        predicates[name] = PredicateSchema(name=name, parameters=parameters)
    return predicates


def parse_functions(section: list[Any] | None) -> dict[str, FunctionSchema]:
    functions: dict[str, FunctionSchema] = {}
    if not section:
        return functions

    for declaration in section[1:]:
        if not isinstance(declaration, list) or not declaration:
            continue
        name = str(declaration[0]).lower()
        parameters = parse_typed_list(declaration[1:])
        functions[name] = FunctionSchema(name=name, parameters=parameters)
    return functions


def parse_actions(form: list[Any]) -> list[ActionSchema]:
    actions: list[ActionSchema] = []
    action_sections = find_all_sections(form, ":action")

    for section in action_sections:
        if len(section) < 2:
            continue
        name = str(section[1]).lower()
        parameters: list[tuple[str, str]] = []
        precondition: Any | None = None
        effect: Any | None = None

        index = 2
        while index < len(section):
            token = section[index]
            if token == ":parameters" and index + 1 < len(section):
                parameters = parse_typed_list(section[index + 1])
                index += 2
            elif token == ":precondition" and index + 1 < len(section):
                precondition = section[index + 1]
                index += 2
            elif token == ":effect" and index + 1 < len(section):
                effect = section[index + 1]
                index += 2
            else:
                index += 1

        actions.append(
            ActionSchema(
                name=name,
                parameters=parameters,
                precondition=precondition,
                effect=effect,
            )
        )

    return actions


def is_number(token: Any) -> bool:
    if not isinstance(token, str):
        return False
    try:
        float(token)
    except ValueError:
        return False
    return True


def contains_numeric_expression(expr: Any, function_names: set[str]) -> bool:
    if is_number(expr):
        return True
    if not isinstance(expr, list) or not expr:
        return False

    head = str(expr[0]).lower()
    if head in function_names or head in ARITHMETIC_OPERATORS:
        return True
    return any(contains_numeric_expression(item, function_names) for item in expr[1:])


def count_numeric_preconditions(expr: Any, function_names: set[str]) -> int:
    if not isinstance(expr, list) or not expr:
        return 0

    head = str(expr[0]).lower()
    if head in NUMERIC_COMPARISONS:
        if any(contains_numeric_expression(item, function_names) for item in expr[1:]):
            return 1
        return 0

    return sum(count_numeric_preconditions(item, function_names) for item in expr[1:])


def count_numeric_effects(expr: Any) -> int:
    if not isinstance(expr, list) or not expr:
        return 0

    head = str(expr[0]).lower()
    current = 1 if head in NUMERIC_EFFECTS else 0
    return current + sum(count_numeric_effects(item) for item in expr[1:])


def is_predicate_atom(expr: Any) -> bool:
    if not isinstance(expr, list) or not expr or not isinstance(expr[0], str):
        return False
    head = expr[0].lower()
    return (
        not head.startswith(":")
        and head not in LOGICAL_OPERATORS
        and head not in NUMERIC_COMPARISONS
        and head not in NUMERIC_EFFECTS
        and head not in ARITHMETIC_OPERATORS
    )


def collect_effect_predicates(expr: Any) -> set[str]:
    if not isinstance(expr, list) or not expr:
        return set()

    head = str(expr[0]).lower()
    if head == "not" and len(expr) > 1 and is_predicate_atom(expr[1]):
        return {str(expr[1][0]).lower()}
    if is_predicate_atom(expr):
        return {head}

    changed: set[str] = set()
    for item in expr[1:]:
        changed.update(collect_effect_predicates(item))
    return changed


def collect_positive_precondition_atoms(expr: Any) -> list[tuple[str, ...]]:
    atoms: list[tuple[str, ...]] = []

    def visit(node: Any, negated: bool = False) -> None:
        if not isinstance(node, list) or not node:
            return

        head = str(node[0]).lower()
        if head == "not":
            if len(node) > 1:
                visit(node[1], negated=True)
            return

        if is_predicate_atom(node) and not negated:
            atoms.append(tuple(str(part).lower() for part in node))
            return

        for child in node[1:]:
            visit(child, negated=negated)

    visit(expr)
    return atoms


def parse_domain(path: Path) -> DomainModel:
    form = atom_lower(parse_sexpr(path.read_text(encoding="utf-8")))
    if not isinstance(form, list) or not form or form[0] != "define":
        raise ValueError(f"{path} is not a PDDL domain")

    domain_section = next(
        section for section in form if isinstance(section, list) and section and section[0] == "domain"
    )
    name = str(domain_section[1]).lower()
    type_parents = parse_type_parents(find_section(form, ":types"))
    predicates = parse_predicates(find_section(form, ":predicates"))
    functions = parse_functions(find_section(form, ":functions"))
    actions = parse_actions(form)
    function_names = set(functions)

    changed_predicates: set[str] = set()
    numeric_preconditions = 0
    numeric_effects = 0
    for action in actions:
        changed_predicates.update(collect_effect_predicates(action.effect))
        numeric_preconditions += count_numeric_preconditions(action.precondition, function_names)
        numeric_effects += count_numeric_effects(action.effect)

    return DomainModel(
        name=name,
        type_parents=type_parents,
        predicates=predicates,
        functions=functions,
        actions=actions,
        changed_predicates=changed_predicates,
        numeric_preconditions=numeric_preconditions,
        numeric_effects=numeric_effects,
    )


def numeric_value(value: Any) -> float | None:
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_init(section: list[Any] | None) -> tuple[set[tuple[str, ...]], dict[tuple[str, ...], float]]:
    facts: set[tuple[str, ...]] = set()
    numeric_init: dict[tuple[str, ...], float] = {}
    if not section:
        return facts, numeric_init

    for item in section[1:]:
        if not isinstance(item, list) or not item:
            continue

        head = str(item[0]).lower()
        if head == "=" and len(item) == 3 and isinstance(item[1], list):
            value = numeric_value(item[2])
            if value is not None:
                numeric_init[tuple(str(part).lower() for part in item[1])] = value
        elif is_predicate_atom(item):
            facts.add(tuple(str(part).lower() for part in item))

    return facts, numeric_init


def parse_problem(path: Path) -> ProblemModel:
    form = atom_lower(parse_sexpr(path.read_text(encoding="utf-8")))
    if not isinstance(form, list) or not form or form[0] != "define":
        raise ValueError(f"{path} is not a PDDL problem")

    problem_section = next(
        section for section in form if isinstance(section, list) and section and section[0] == "problem"
    )
    domain_section = find_section(form, ":domain")
    objects_section = find_section(form, ":objects")
    init_section = find_section(form, ":init")
    goal_section = find_section(form, ":goal")
    metric_section = find_section(form, ":metric")

    object_types = {
        name.lower(): type_name.lower()
        for name, type_name in parse_typed_list(objects_section[1:] if objects_section else [])
    }
    init_facts, numeric_init = parse_init(init_section)

    return ProblemModel(
        name=str(problem_section[1]).lower(),
        domain_name=str(domain_section[1]).lower() if domain_section and len(domain_section) > 1 else "",
        object_types=object_types,
        init_facts=init_facts,
        numeric_init=numeric_init,
        goal=goal_section[1] if goal_section and len(goal_section) > 1 else None,
        has_metric=metric_section is not None,
    )


def is_subtype(type_name: str, expected_type: str, type_parents: dict[str, str]) -> bool:
    current = type_name.lower()
    expected = expected_type.lower()
    while current:
        if current == expected:
            return True
        current = type_parents.get(current, "")
    return expected == "object"


def compatible_objects(problem: ProblemModel, domain: DomainModel, expected_type: str) -> list[str]:
    return sorted(
        obj
        for obj, obj_type in problem.object_types.items()
        if is_subtype(obj_type, expected_type, domain.type_parents)
    )


def static_init_facts(problem: ProblemModel, domain: DomainModel) -> set[tuple[str, ...]]:
    return {fact for fact in problem.init_facts if fact[0] not in domain.changed_predicates}


def condition_is_satisfied(atom: tuple[str, ...], assignment: dict[str, str], facts: set[tuple[str, ...]]) -> bool:
    grounded = []
    for part in atom:
        if part.startswith("?"):
            if part not in assignment:
                return True
            grounded.append(assignment[part])
        else:
            grounded.append(part)
    return tuple(grounded) in facts


def unify_static_atom(
    atom: tuple[str, ...],
    fact: tuple[str, ...],
    assignment: dict[str, str],
    allowed_values: dict[str, set[str]],
) -> dict[str, str] | None:
    if len(atom) != len(fact) or atom[0] != fact[0]:
        return None

    next_assignment = dict(assignment)
    for atom_part, fact_part in zip(atom[1:], fact[1:]):
        if atom_part.startswith("?"):
            if fact_part not in allowed_values.get(atom_part, set()):
                return None
            assigned_value = next_assignment.get(atom_part)
            if assigned_value is not None and assigned_value != fact_part:
                return None
            next_assignment[atom_part] = fact_part
        elif atom_part != fact_part:
            return None

    return next_assignment


def estimate_feasible_action_count(action: ActionSchema, problem: ProblemModel, domain: DomainModel) -> int:
    param_domains = [
        (variable.lower(), compatible_objects(problem, domain, type_name))
        for variable, type_name in action.parameters
    ]
    if any(not values for _, values in param_domains):
        return 0

    static_facts = static_init_facts(problem, domain)
    static_fact_predicates = {fact[0] for fact in static_facts}
    static_domain_predicates = set(domain.predicates) - domain.changed_predicates
    relevant_atoms = [
        atom
        for atom in collect_positive_precondition_atoms(action.precondition)
        if atom[0] in static_domain_predicates
    ]

    if not param_domains:
        return 1 if all(condition_is_satisfied(atom, {}, static_facts) for atom in relevant_atoms) else 0

    unconstrained_count = math.prod(len(values) for _, values in param_domains)
    if not relevant_atoms:
        return unconstrained_count

    if any(atom[0] not in static_fact_predicates for atom in relevant_atoms):
        return 0

    allowed_values = {variable: set(values) for variable, values in param_domains}
    facts_by_predicate: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for fact in static_facts:
        facts_by_predicate[fact[0]].append(fact)

    partial_assignments: set[tuple[tuple[str, str], ...]] = {tuple()}
    for atom in relevant_atoms:
        next_assignments: set[tuple[tuple[str, str], ...]] = set()
        for assignment_key in partial_assignments:
            assignment = dict(assignment_key)
            for fact in facts_by_predicate.get(atom[0], []):
                unified = unify_static_atom(atom, fact, assignment, allowed_values)
                if unified is not None:
                    next_assignments.add(tuple(sorted(unified.items())))
        partial_assignments = next_assignments
        if not partial_assignments:
            return 0

    count = 0
    all_variables = {variable for variable, _ in param_domains}
    for assignment_key in partial_assignments:
        assigned_variables = {variable for variable, _ in assignment_key}
        multiplier = math.prod(
            len(allowed_values[variable])
            for variable in all_variables - assigned_variables
        )
        count += multiplier

    return count


def count_number_of_actions(problem: ProblemModel, domain: DomainModel) -> int:
    return sum(estimate_feasible_action_count(action, problem, domain) for action in domain.actions)


def count_goal_conditions(expr: Any, function_names: set[str]) -> tuple[int, int]:
    total = 0
    numeric = 0

    def visit(node: Any) -> None:
        nonlocal total, numeric
        if not isinstance(node, list) or not node:
            return

        head = str(node[0]).lower()
        if head in LOGICAL_OPERATORS:
            for child in node[1:]:
                visit(child)
            return

        if head in NUMERIC_COMPARISONS and any(
            contains_numeric_expression(item, function_names) for item in node[1:]
        ):
            total += 1
            numeric += 1
            return

        if is_predicate_atom(node):
            total += 1
            return

        for child in node[1:]:
            visit(child)

    visit(expr)
    return total, numeric


def extract_topology_edges(problem: ProblemModel) -> set[tuple[str, str, str]]:
    edges: set[tuple[str, str, str]] = set()

    for fact in problem.init_facts:
        predicate = fact[0]
        if not any(marker in predicate for marker in TOPOLOGY_MARKERS):
            continue

        if len(fact) == 3:
            edges.add((predicate, fact[1], fact[2]))
        elif predicate == "can_traverse" and len(fact) == 4:
            edges.add((predicate, fact[2], fact[3]))

    return edges


def topology_features(problem: ProblemModel) -> tuple[int, int, float, float]:
    edges = extract_topology_edges(problem)
    nodes = {node for _, source, target in edges for node in (source, target)}
    node_count = len(nodes)
    edge_count = len(edges)
    possible_edges = node_count * (node_count - 1)
    density = min(1.0, edge_count / possible_edges) if possible_edges > 0 else 0.0
    topology_score = math.log1p(node_count) + math.log1p(edge_count) + density
    return node_count, edge_count, density, topology_score


def score_instance(domain_name: str, instance_path: Path, domain: DomainModel) -> dict[str, Any]:
    problem = parse_problem(instance_path)
    number_of_objects = len(problem.object_types)
    number_of_actions = count_number_of_actions(problem, domain)
    number_of_goals, numeric_goal_conditions = count_goal_conditions(problem.goal, set(domain.functions))
    topology_nodes, topology_edges, topology_density, topology_score = topology_features(problem)
    has_metric = 1 if problem.has_metric else 0
    numeric_score = (
        domain.numeric_effects
        + 2 * domain.numeric_preconditions
        + 3 * numeric_goal_conditions
        + has_metric
    )
    raw_score = (
        0.45 * math.log1p(number_of_actions)
        + 0.20 * math.log1p(number_of_goals)
        + 0.20 * math.log1p(numeric_score)
        + 0.15 * topology_score
    )

    return {
        "domain": domain_name,
        "instance": instance_path.stem,
        "problem_name": problem.name,
        "number_of_objects": number_of_objects,
        "number_of_actions": number_of_actions,
        "number_of_goals": number_of_goals,
        "numeric_goal_conditions": numeric_goal_conditions,
        "number_of_numeric_functions": len(domain.functions),
        "number_of_numeric_preconditions": domain.numeric_preconditions,
        "number_of_numeric_effects": domain.numeric_effects,
        "has_metric": bool(problem.has_metric),
        "topology_nodes": topology_nodes,
        "topology_edges": topology_edges,
        "topology_density": round(topology_density, 6),
        "raw_score": raw_score,
    }


def round_instance_scores(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        row["raw_score"] = round(row["raw_score"], 6)


def summarize_domains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_domain[row["domain"]].append(row)

    summaries = []
    for domain_name in sorted(rows_by_domain):
        domain_rows = rows_by_domain[domain_name]
        raw_scores = [float(row["raw_score"]) for row in domain_rows]
        summaries.append(
            {
                "domain": domain_name,
                "instances": len(domain_rows),
                "total_raw_score": round(sum(raw_scores), 6),
                "min_raw_score": round(min(raw_scores), 6),
                "median_raw_score": round(median(raw_scores), 6),
                "mean_raw_score": round(mean(raw_scores), 6),
                "max_raw_score": round(max(raw_scores), 6),
                "domain_difficulty_0_100": 0.0,
            }
        )

    if summaries:
        total_scores = [summary["total_raw_score"] for summary in summaries]
        min_total = min(total_scores)
        max_total = max(total_scores)
        span = max_total - min_total
        for summary in summaries:
            if span == 0:
                summary["domain_difficulty_0_100"] = 0.0
            else:
                summary["domain_difficulty_0_100"] = round(
                    100 * (summary["total_raw_score"] - min_total) / span,
                    6,
                )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def discover_domain_dirs(domains_dir: Path) -> list[Path]:
    if not domains_dir.exists():
        return []
    return sorted(
        path
        for path in domains_dir.iterdir()
        if path.is_dir() and (
            (path / "domain.pddl").exists() or 
            (path / "domain" / "domain.pddl").exists()
        )
    )


def score_domains(domains_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for domain_dir in discover_domain_dirs(domains_dir):
        # Cerca il file di dominio sia nella root che nella sottocartella 'domain'
        domain_pddl_path = domain_dir / "domain.pddl"
        if not domain_pddl_path.exists():
            domain_pddl_path = domain_dir / "domain" / "domain.pddl"
        
        if not domain_pddl_path.exists():
            continue
            
        domain = parse_domain(domain_pddl_path)

        all_instance_paths: list[Path] = []
        
        # Caso 1: Cartella 'instances' standard
        instances_dir = domain_dir / "instances"
        if instances_dir.exists():
            all_instance_paths.extend(sorted(instances_dir.glob("*.pddl")))
        
        # Caso 2: Sottocartelle divise per tier (easy, medium, hard)
        for subdir_name in ["easy", "medium", "hard"]:
            subdir = domain_dir / subdir_name
            if subdir.exists():
                all_instance_paths.extend(sorted(subdir.glob("*.pddl")))

        for instance_path in all_instance_paths:
            rows.append(score_instance(domain_dir.name, instance_path, domain))

    round_instance_scores(rows)
    return rows


def default_paths() -> tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    return (
        repo_root / "Benchmark_Framework" / "tasks",
        repo_root / "analysis" / "domain_complexity",
    )


def parse_args() -> argparse.Namespace:
    default_domains_dir, default_output_dir = default_paths()
    parser = argparse.ArgumentParser(description="Score PDDL domain instance complexity.")
    parser.add_argument(
        "--domains-dir",
        type=Path,
        default=default_domains_dir,
        help="Directory containing planning domain folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory where CSV and JSON reports are written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    domains_path = args.domains_dir.resolve()
    print(f"Sto cercando i domini in: {domains_path}")
    
    if not domains_path.exists():
        print(f"ERRORE: La cartella dei domini non esiste: {domains_path}")
        return

    rows = score_domains(args.domains_dir)
    if not rows:
        print(f"ATTENZIONE: Nessun dominio PDDL valido trovato in {domains_path}")
        print("Assicurati che le sottocartelle contengano un file 'domain.pddl' o 'domain/domain.pddl'.")
        return

    summaries = summarize_domains(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "complexity_scores.csv", rows)
    write_json(args.output_dir / "complexity_scores.json", rows)
    write_csv(args.output_dir / "domain_summary.csv", summaries)
    write_json(args.output_dir / "domain_summary.json", summaries)

    print(f"Scored {len(rows)} instances from {args.domains_dir}")
    print(f"Report generati con successo in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
