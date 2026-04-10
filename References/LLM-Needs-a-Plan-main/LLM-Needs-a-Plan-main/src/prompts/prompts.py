"""
PDDL Planning Prompts (reorganized)

This module provides a clear, domain-separated prompt pipeline for PDDL planning.
It contains two main sections:
  - Tetris domain prompts
  - CityCar domain prompts

Each domain exposes a function that builds a full prompt for that domain and
additional helpers for Chain-of-Thought (CoT) and validation-feedback prompts.

The file also preserves a few generic helpers used elsewhere in the codebase so
this can be swapped in without breaking imports.
"""

from typing import List, Optional

# -----------------------------
# System / shared definitions
# -----------------------------

# Base system prompt for PDDL planning tasks
system_prompt_pddl = (
    "You are an expert automated PDDL planning assistant specializing in complex domains "
    "like Tetris and Citycar, which involve explicit coordinate systems, "
    "multi-parameter actions, negative preconditions, and action costs.\n"
    "Your primary objective is to find the **shortest and/or lowest-cost** plan that achieves the goal.\n"
    "When asked to return a plan, follow these rules unless explicitly told otherwise:\n"
    "- Output only the action sequence, one action per line, using the exact PDDL action syntax: (action-name param1 param2 ...).\n"
    "- Do NOT repeat the domain or problem definition.\n"
    "- Do NOT output the initial state or goal state.\n"
    "- Do not include explanations, commentary, or extra metadata unless requested.\n"
    "- If you use <think> tags for reasoning, keep the reasoning concise and ensure the final plan is output clearly after the closing </think> tag.\n"
    "- Crucially: Verify that all action preconditions are respected in the current state "
    "and that the plan achieves all goal predicates while minimizing the total-cost.\n"
)

# Generic Chain-of-Thought helper (can be appended when CoT is requested)
def chain_of_thought_prompt(domain: str, problem: str) -> str:
    """
    Generic Chain-of-Thought prompt generator.
    Returns a short instruction guiding the model to reason step-by-step. Accepts
    domain and problem strings to allow future domain/problem-aware hints.
    """
    return (
        "Let's reason step-by-step, about the domain and problem descriptions provided, to construct the optimal plan.\n"
        "Output ONLY the corrected action sequence (one action per line)."
    )

# Utilities
def add_examples_to_prompt(base_prompt: str, examples: List[str]) -> str:
    examples_text = "\n\n".join(f"Example {i+1}:\n{ex}" for i, ex in enumerate(examples))
    return f"{base_prompt}\n\nExamples:\n{examples_text}"


def add_constraints_to_prompt(base_prompt: str, constraints: List[str]) -> str:
    constraints_text = "\n".join(f"- {c}" for c in constraints)
    return f"{base_prompt}\n\nAdditional constraints:\n{constraints_text}"


# -----------------------------
# TETRIS domain: prompts + examples
# -----------------------------

# Few-shot examples for Tetris (three small instances + solution plans).
# Keep examples concise: domain/problem summary and the expected plan (action lines).
_tetris_examples = [
    {
        # 1. Action: move_square
        # Scenario: Move a single square piece (squareA) from f0-0f to f0-1f (one step right).
        # Parameters: ?xy_initial, ?xy_final, ?element
        "instance": """
(:objects squareA - one_square f0-0f f0-1f - position)
(:init
(clear f0-1f)
(at_square squareA f0-0f)
(connected f0-0f f0-1f)
(connected f0-1f f0-0f)
)
(:goal (at_square squareA f0-1f))
""",
        "plan": "(move_square f0-0f f0-1f squareA)",
    },
    {
        # 2. Action: move_two (Shift/Rotate)
        # Scenario: Shift a two-square straight piece (straightB) down one row.
        # Position change: (f0-0f, f1-0f) -> (f1-0f, f2-0f).
        # Parameters: ?xy_initial1, ?xy_initial2, ?xy_final, ?element
        "instance": """
(:objects straightB - two_straight f0-0f f1-0f f2-0f - position)
(:init
(clear f2-0f)
(at_two straightB f0-0f f1-0f)
(connected f1-0f f2-0f)
)
(:goal (at_two straightB f1-0f f2-0f))
""",
        "plan": "(move_two f0-0f f1-0f f2-0f straightB)",
    },
    {
        # 3. Action: move_l_down
        # Scenario: Shift an L-piece (rightlC) down one row.
        # Position change: (f2-1f, f1-1f, f1-2f) -> (f1-1f, f0-1f, f0-2f).
        # Parameters: ?xy_initial1, ?xy_initial2, ?xy_initial3, ?xy_final, ?xy_final2, ?element
        # Note: The action requires 6 position parameters and 1 element parameter.
        "instance": """
(:objects rightlC - right_l f2-1f f1-1f f1-2f f0-1f f0-2f - position)
(:init
(clear f0-1f)
(clear f0-2f)
(at_right_l rightlC f2-1f f1-1f f1-2f)
(connected f1-1f f0-1f)
(connected f1-2f f0-2f)
(connected f0-2f f0-1f)
)
(:goal (at_right_l rightlC f1-1f f0-1f f0-2f))
""",
        # Note: The action move_l_down has 5 position parameters in the effect, but the PDDL signature requires 7 parameters total.
        "plan": "(move_l_down f2-1f f1-1f f1-2f f0-1f f0-2f rightlC)",
    },
]


def _format_tetris_examples() -> List[str]:
    return [f"Instance: {e['instance']}\nPlan:\n{e['plan']}" for e in _tetris_examples]


def tetris_problem_prompt(domain: str, problem: str, include_examples: bool = True, strict_plan_only: bool = True) -> str:
    """
    Build a Tetris-specific prompt.

    Args:
        domain: domain PDDL text
        problem: problem PDDL text
        include_examples: whether to include few-shot examples (default True)
        strict_plan_only: enforce plan-only output instructions (default True)

    Returns:
        A full prompt string ready to pass to the model as the user message.
    """
    header = (
        "TETRIS PLANNING TASK:\n"
        "You are solving a Tetris configuration planning problem.\n"
        "Now I provide you with all the information needed: the domain and the problem to be solved.\n"
    )

    body = f"=== DOMAIN DEFINITION ===\n{domain}\n\n=== PROBLEM DEFINITION ===\n{problem}"

    instructions = (
        "\nOUTPUT INSTRUCTIONS:\n"
        "- Provide ONLY the action sequence in PDDL action syntax, one action per line.\n"
        "- Do NOT include narrative, reasoning, or extraneous text.\n"
        "- Use exact token names from the domain/problem.\n"
    )

    prompt = f"{system_prompt_pddl}\n\n{header}\n\n{body}"

    # Add output instructions
    prompt += "\n\n" + instructions

    if include_examples:
        try:
            prompt = add_examples_to_prompt(prompt, _format_tetris_examples())
        except Exception:
            # Fallback: append raw examples
            prompt += "\n\n" + "\n\n".join(_format_tetris_examples())

    # End prompt with an explicit request to provide the plan now
    prompt += "\n\nPlease provide the plan (one action per line) now."

    return prompt


def tetris_chain_of_thought(domain: str, problem: str) -> str:
    # Reuse the generic CoT structure but tailor the preface for Tetris
    return (
        chain_of_thought_prompt(domain, problem)
    )


def tetris_validation_feedback(original_prompt: str, plan: str, validation_error: str) -> str:
    """
    Generate a feedback prompt for the Tetris domain to request a corrected plan after validation fails.
    """
    return (
        "Your previous Tetris plan did not validate correctly.\n\n"
        "ORIGINAL PROBLEM:\n" + original_prompt + "\n\n"
        "YOUR GENERATED PLAN:\n" + plan + "\n\n"
        "VALIDATION ERROR:\n" + validation_error + "\n\n"
        "Notes:\n- Check that moved pieces exist and preconditions (e.g., free target cells) are satisfied.\n"
        "- Ensure rotations are valid and do not collide with other pieces.\n\n"
        "Please provide a corrected plan that fixes the above error. Output ONLY the corrected action sequence (one action per line)."
    )


# -----------------------------
# CITYCAR domain: prompts + examples
# -----------------------------

_citycar_examples = [
    {
        # 1. Action Sequence: Move a car from Junction to Junction
        # Scenario: Car c0 at junction0-0 needs to move to junction1-0. A road (road0) must exist and be in place.
        # Action 1: move_car_in_road (?xy_initial, ?xy_final, ?machine, ?r1)
        # Action 2: move_car_out_road (?xy_initial, ?xy_final, ?machine, ?r1)
        "instance": """
(:objects car0 - car junction0-0 junction1-0 - junction road0 - road)
(:init
(at_car_jun car0 junction0-0)
(clear junction1-0)
(road_connect road0 junction0-0 junction1-0)
(in_place road0)
)
(:goal (at_car_jun car0 junction1-0))
""",
        "plan": "(move_car_in_road junction0-0 junction1-0 car0 road0)\n(move_car_out_road junction0-0 junction1-0 car0 road0)",
    },
    {
        # 2. Action: Car Start and Move Sequence
        # Scenario: Car c1 starts in garage0 at junction0-0 and moves to junction0-1. Requires building a road first.
        # Action 1: car_start (?xy_final, ?machine, ?g)
        # Action 2: build_straight_oneway (?xy_initial, ?xy_final, ?r1)
        # Action 3 & 4: move_car_in_road/move_car_out_road (to use the newly built road)
        "instance": """
(:objects car1 - car junction0-0 junction0-1 - junction garage0 - garage road1 - road)
(:init
(starting car1 garage0)
(at_garage garage0 junction0-0)
(clear junction0-0)
(clear junction0-1)
(same_line junction0-0 junction0-1)
(not (in_place road1))
)
(:goal (at_car_jun car1 junction0-1))
""",
        "plan": "(car_start junction0-0 car1 garage0)\n(build_straight_oneway junction0-0 junction0-1 road1)\n(move_car_in_road junction0-0 junction0-1 car1 road1)\n(move_car_out_road junction0-0 junction0-1 car1 road1)",
    },
    {
        # 3. Action Sequence: Road Destruction and Car Arrival
        # Scenario: Car c0 needs to arrive at junction2-0. Afterwards, a connected road (road2) is destroyed.
        # Action 1 & 2: Move car to the final junction (assumed)
        # Action 3: car_arrived (?xy_final, ?machine)
        # Action 4: destroy_road (?xy_initial, ?xy_final, ?r1)
        "instance": """
(:objects car0 - car junction1-0 junction2-0 - junction road2 - road)
(:init
(at_car_jun car0 junction2-0)
(road_connect road2 junction1-0 junction2-0)
(in_place road2)
)
(:goal
(and
(arrived car0 junction2-0)
(not (in_place road2))
)
)
""",
        "plan": "(car_arrived junction2-0 car0)\n(destroy_road junction1-0 junction2-0 road2)",
    },
]


def _format_citycar_examples() -> List[str]:
    return [f"Instance: {e['instance']}\nPlan:\n{e['plan']}" for e in _citycar_examples]


def citycar_problem_prompt(domain: str, problem: str, include_examples: bool = True, strict_plan_only: bool = True) -> str:
    """
    Build a CityCar-specific prompt.

    Args:
        domain: domain PDDL text
        problem: problem PDDL text
        include_examples: whether to include few-shot examples (default True)
        strict_plan_only: enforce plan-only output instructions (default True)

    Returns:
        A full prompt string ready to pass to the model as the user message.
    """
    header = (
        "CITYCAR PLANNING TASK:\n"
        "You are solving an urban traffic planning problem modeled in PDDL.\n"
        "Now I provide you with all the information needed: the domain and the problem to be solved.\n"
    )

    body = f"=== DOMAIN DEFINITION ===\n{domain}\n\n=== PROBLEM DEFINITION ===\n{problem}"

    instructions = (
        "\nOUTPUT INSTRUCTIONS:\n"
        "- Provide ONLY the action sequence in PDDL action syntax, one action per line.\n"
        "- Do NOT include narrative or explanations.\n"
        "- Respect traffic rules, signals, and collision avoidance encoded in the domain.\n"
    )

    prompt = f"{system_prompt_pddl}\n\n{header}\n\n{body}"

    # Add output instructions
    prompt += "\n\n" + instructions

    if include_examples:
        try:
            prompt = add_examples_to_prompt(prompt, _format_citycar_examples())
        except Exception:
            prompt += "\n\n" + "\n\n".join(_format_citycar_examples())

    prompt += "\n\nProvide the plan now (one action per line)."

    return prompt


def citycar_chain_of_thought(domain: str, problem: str) -> str:
    # Reuse the generic CoT structure but tailor the preface for CityCar
    return chain_of_thought_prompt(domain, problem)


def citycar_validation_feedback(original_prompt: str, plan: str, validation_error: str) -> str:
    """
    Generate a feedback prompt for CityCar to request a corrected plan after validation fails.
    """
    return (
        "Your previous CityCar plan failed validation.\n\n"
        "ORIGINAL PROBLEM:\n" + original_prompt + "\n\n"
        "YOUR PLAN:\n" + plan + "\n\n"
        "VALIDATION ERROR:\n" + validation_error + "\n\n"
        "Hints:\n- Check that each drive action uses connected intersections.\n"
        "- Ensure signals are obeyed and cars do not attempt conflicting moves.\n\n"
        "Please produce a corrected plan that fixes the validation errors. Output ONLY the corrected action sequence (one action per line)."
    )


# -----------------------------
# Generic prompts kept for compatibility
# -----------------------------

def generic_pddl_prompt(domain: str, problem: str) -> str:
    return f"{system_prompt_pddl}\n\n=== DOMAIN DEFINITION ===\n{domain}\n\n=== PROBLEM DEFINITION ===\n{problem}\n\nGenerate a valid plan (one action per line)."


def validation_feedback_prompt(original_prompt: str, plan: str, validation_error: str) -> str:
    # Generic fallback validation feedback
    return (
        "Your previous plan did not pass validation.\n\n"
        "ORIGINAL PROBLEM:\n" + original_prompt + "\n\n"
        "YOUR PLAN:\n" + plan + "\n\n"
        "VALIDATION ERROR:\n" + validation_error + "\n\n"
        "Please provide a corrected plan. Output only the corrected action sequence."
    )


def optimization_prompt(domain: str, problem: str) -> str:
    return (
        f"{system_prompt_pddl}\n\nOptimize the following problem for minimal actions.\n\n=== DOMAIN ===\n{domain}\n\n=== PROBLEM ===\n{problem}\n\n"
        "Return the most efficient action sequence (one action per line)."
    )


def incremental_planning_prompt(domain: str, problem: str) -> str:
    return (
        f"{system_prompt_pddl}\n\nProvide an incremental plan: propose one action at a time and wait for feedback before the next.\n\n=== DOMAIN ===\n{domain}\n\n=== PROBLEM ===\n{problem}"
    )

# End of file