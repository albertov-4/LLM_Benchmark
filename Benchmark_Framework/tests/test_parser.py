"""Unit tests for the shared benchmark parser."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


TOY_DOMAIN = """
(define (domain toy)
  (:predicates (at ?x))
  (:action move
    :parameters (?from ?to)
    :precondition (and (at ?from))
    :effect (and (at ?to))
  )
  (:action pick
    :parameters (?box ?room)
    :precondition (and)
    :effect (and)
  )
)
"""

BLOCK_DOMAIN = """
(define (domain blocks)
  (:action move_block_up :parameters (?b))
  (:action move_block_right :parameters (?b))
)
"""

SAILING_DOMAIN = """
(define (domain sailing)
  (:action accelerate :parameters (?b))
  (:action go_south :parameters (?b))
  (:action decelerate :parameters (?b))
  (:action save_person :parameters (?b ?p))
)
"""

ROVER_DOMAIN = """
(define (domain rover)
  (:action navigate :parameters (?r ?from ?to))
)
"""


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ParserTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        cls.parser_module = _load_module(
            "benchmark_framework_test_parser_module",
            framework_root / "evaluators" / "parser.py",
        )

    def test_parse_empty_output(self) -> None:
        result = self.parser_module.parse_plan_text("", domain_text=TOY_DOMAIN)

        self.assertEqual(result.raw["actions"], [])
        self.assertEqual(result.raw["format_issues"], ["empty_output"])
        self.assertEqual(result.reasoning["format_issues"], ["reasoning_text_empty"])

    def test_parse_extracts_raw_and_reasoning_actions(self) -> None:
        raw_text = "I will first think through the problem.\n(move a b)\n(pick box1 room2)"
        reasoning_text = "Try one option.\n(move c d)\nNo.\n\nFinal list:\n(move a b)\n(pick box1 room2)"

        result = self.parser_module.parse_plan_text(raw_text, reasoning_text=reasoning_text, domain_text=TOY_DOMAIN)

        self.assertEqual(result.raw["actions"], ["(move a b)", "(pick box1 room2)"])
        self.assertTrue(result.raw["contains_reasoning"])
        self.assertIn("raw_text_contains_reasoning_like_content", result.raw["format_issues"])
        self.assertEqual(result.reasoning["actions"], ["(move a b)", "(pick box1 room2)"])
        self.assertIn("multiple_reasoning_candidate_plans", result.reasoning["format_issues"])

    def test_parse_removes_markdown_fences_and_numbering(self) -> None:
        raw_text = "```text\n1. (move a b)\n2. (pick box1 room2)\n```"

        result = self.parser_module.parse_plan_text(raw_text, domain_text=TOY_DOMAIN)

        self.assertEqual(result.raw["actions"], ["(move a b)", "(pick box1 room2)"])
        self.assertIn("markdown_fences_removed", result.raw["format_issues"])

    def test_domain_filter_rejects_predicates_unknown_actions_and_wrong_arity(self) -> None:
        raw_text = "(at a)\n(fly a b)\n(move a)\n(move a b)"

        result = self.parser_module.parse_plan_text(raw_text, domain_text=TOY_DOMAIN)

        self.assertEqual(result.raw["actions"], ["(move a b)"])
        self.assertIn("unknown_action_names_found", result.raw["format_issues"])
        self.assertIn("wrong_action_arity", result.raw["format_issues"])

    def test_parse_reports_missing_actions(self) -> None:
        result = self.parser_module.parse_plan_text("This is a detailed explanation, but not a plan.", domain_text=TOY_DOMAIN)

        self.assertEqual(result.raw["actions"], [])
        self.assertIn("no_valid_domain_actions_found", result.raw["format_issues"])

    def test_parse_expands_explicit_repeat(self) -> None:
        result = self.parser_module.parse_plan_text("repeat 3 times (move a b)", domain_text=TOY_DOMAIN)

        self.assertEqual(result.raw["actions"], ["(move a b)", "(move a b)", "(move a b)"])
        self.assertIn("compressed_actions_expanded", result.raw["format_issues"])

    def test_parse_expands_repeat_after_action(self) -> None:
        for text in (
            "(move a b) repeated 3 times",
            "(move a b) for 3 times",
            "repeated 3 times (move a b)",
        ):
            with self.subTest(text=text):
                result = self.parser_module.parse_plan_text(text, domain_text=TOY_DOMAIN)

                self.assertEqual(result.raw["actions"], ["(move a b)", "(move a b)", "(move a b)"])
                self.assertIn("compressed_actions_expanded", result.raw["format_issues"])

    def test_parse_expands_repeat_after_action_in_reasoning(self) -> None:
        reasoning_text = "Final plan:\n(move a b) repeated 3 times\n(pick box1 room2)"

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=TOY_DOMAIN)

        self.assertEqual(
            result.reasoning["actions"],
            ["(move a b)", "(move a b)", "(move a b)", "(pick box1 room2)"],
        )
        self.assertIn("compressed_actions_expanded", result.reasoning["format_issues"])

    def test_reasoning_expands_parenthesized_repeat_with_current_object(self) -> None:
        reasoning_text = """(accelerate b0)
(accelerate b0)
(repeat go_south 3 times)
(decelerate b0)
(save_person b0 p0)
"""

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=SAILING_DOMAIN)

        self.assertEqual(
            result.reasoning["actions"],
            [
                "(accelerate b0)",
                "(accelerate b0)",
                "(go_south b0)",
                "(go_south b0)",
                "(go_south b0)",
                "(decelerate b0)",
                "(save_person b0 p0)",
            ],
        )
        self.assertIn("compressed_actions_expanded", result.reasoning["format_issues"])

    def test_parenthesized_repeat_does_not_invent_missing_object(self) -> None:
        result = self.parser_module.parse_plan_text("", reasoning_text="(repeat go_south 3 times)", domain_text=SAILING_DOMAIN)

        self.assertEqual(result.reasoning["actions"], [])
        self.assertIn("ambiguous_compressed_action", result.reasoning["format_issues"])

    def test_parenthesized_repeat_uses_safe_alias_with_current_object(self) -> None:
        reasoning_text = """(move_block_right b1)
(repeat up 2 times)
"""

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=BLOCK_DOMAIN)

        self.assertEqual(
            result.reasoning["actions"],
            ["(move_block_right b1)", "(move_block_up b1)", "(move_block_up b1)"],
        )
        self.assertIn("compressed_actions_expanded", result.reasoning["format_issues"])

    def test_repeat_after_action_still_rejects_invalid_domain_actions(self) -> None:
        unknown = self.parser_module.parse_plan_text("(fly a b) repeated 3 times", domain_text=TOY_DOMAIN)
        wrong_arity = self.parser_module.parse_plan_text("(move a) repeated 3 times", domain_text=TOY_DOMAIN)

        self.assertEqual(unknown.raw["actions"], [])
        self.assertIn("unknown_action_names_found", unknown.raw["format_issues"])
        self.assertEqual(wrong_arity.raw["actions"], [])
        self.assertIn("wrong_action_arity", wrong_arity.raw["format_issues"])

    def test_reasoning_prefers_progressive_numbered_action_list(self) -> None:
        reasoning_text = """Summary: b1 up 2, b2 up 2.

Actions:
1 (move_block_up b1)
2 (move_block_up b1)

b2 moves:
3 (move_block_right b2)
4 (move_block_right b2)
5 (move_block_up b2)
"""

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=BLOCK_DOMAIN)

        self.assertEqual(
            result.reasoning["actions"],
            [
                "(move_block_up b1)",
                "(move_block_up b1)",
                "(move_block_right b2)",
                "(move_block_right b2)",
                "(move_block_up b2)",
            ],
        )

    def test_reasoning_expands_numbered_compressed_list(self) -> None:
        reasoning_text = """Actions:
1 b1 up 2
2 right 1
3 b2 right 2
"""

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=BLOCK_DOMAIN)

        self.assertEqual(
            result.reasoning["actions"],
            [
                "(move_block_up b1)",
                "(move_block_up b1)",
                "(move_block_right b1)",
                "(move_block_right b2)",
                "(move_block_right b2)",
            ],
        )
        self.assertIn("compressed_actions_expanded", result.reasoning["format_issues"])

    def test_reasoning_numbered_compressed_list_survives_headings(self) -> None:
        reasoning_text = """1 b1 up 2

Continue b1:
2 right 1

Move b2:
3 b2 up 1
"""

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=BLOCK_DOMAIN)

        self.assertEqual(
            result.reasoning["actions"],
            [
                "(move_block_up b1)",
                "(move_block_up b1)",
                "(move_block_right b1)",
                "(move_block_up b2)",
            ],
        )

    def test_numbered_compression_does_not_invent_unknown_or_ambiguous_aliases(self) -> None:
        ambiguous_domain = """
(define (domain ambiguous)
  (:action move_left :parameters (?b))
  (:action slide_left :parameters (?b))
)
"""

        unknown = self.parser_module.parse_plan_text("", reasoning_text="1 b1 sideways 2", domain_text=BLOCK_DOMAIN)
        ambiguous = self.parser_module.parse_plan_text("", reasoning_text="1 b1 left 2", domain_text=ambiguous_domain)

        self.assertEqual(unknown.reasoning["actions"], [])
        self.assertIn("ambiguous_compressed_action", unknown.reasoning["format_issues"])
        self.assertEqual(ambiguous.reasoning["actions"], [])
        self.assertIn("ambiguous_compressed_action", ambiguous.reasoning["format_issues"])

    def test_numbered_compression_does_not_reconstruct_multi_argument_actions(self) -> None:
        result = self.parser_module.parse_plan_text("", reasoning_text="1 rover1 navigate 2", domain_text=ROVER_DOMAIN)

        self.assertEqual(result.reasoning["actions"], [])
        self.assertIn("no_valid_domain_actions_found", result.reasoning["format_issues"])

    def test_parse_expands_safe_one_argument_compression(self) -> None:
        result = self.parser_module.parse_plan_text("b4 up 2\nb4 right 1", domain_text=BLOCK_DOMAIN)

        self.assertEqual(
            result.raw["actions"],
            ["(move_block_up b4)", "(move_block_up b4)", "(move_block_right b4)"],
        )
        self.assertIn("compressed_actions_expanded", result.raw["format_issues"])

    def test_parse_does_not_invent_missing_arguments_for_other_domains(self) -> None:
        result = self.parser_module.parse_plan_text("rover1 goes to wp2", domain_text=ROVER_DOMAIN)

        self.assertEqual(result.raw["actions"], [])
        self.assertIn("no_valid_domain_actions_found", result.raw["format_issues"])

    def test_reasoning_candidates_include_multiple_sailing_plans(self) -> None:
        reasoning_text = """First candidate:
(go_south b0) x174
(save_person b0 p0)

Final answer:
(accelerate b0) twice
58 times (go_south b0)
(decelerate b0) twice
(save_person b0 p0)
"""

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=SAILING_DOMAIN)
        candidates = self.parser_module.extract_reasoning_candidates(reasoning_text, domain_text=SAILING_DOMAIN)

        self.assertGreaterEqual(len(candidates), 2)
        self.assertEqual(result.reasoning["actions"].count("(go_south b0)"), 58)
        self.assertEqual(result.reasoning["actions"][0:2], ["(accelerate b0)", "(accelerate b0)"])
        self.assertEqual(result.reasoning["actions"][-3:], ["(decelerate b0)", "(decelerate b0)", "(save_person b0 p0)"])

    def test_parenthesized_repeat_count_before_action(self) -> None:
        result = self.parser_module.parse_plan_text("", reasoning_text="(repeat 3 times) (go_south b0)", domain_text=SAILING_DOMAIN)

        self.assertEqual(result.reasoning["actions"], ["(go_south b0)", "(go_south b0)", "(go_south b0)"])
        self.assertIn("compressed_actions_expanded", result.reasoning["format_issues"])

    def test_reasoning_candidate_marks_truncated_action_fragment(self) -> None:
        reasoning_text = """Final answer:
(accelerate b0)
(go_south"""

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=SAILING_DOMAIN)
        candidates = self.parser_module.extract_reasoning_candidates(reasoning_text, domain_text=SAILING_DOMAIN)

        self.assertTrue(candidates[0]["truncated"])
        self.assertIn("truncated_reasoning_candidate", result.reasoning["format_issues"])


    def test_reasoning_composes_nearby_compressed_fragments(self) -> None:
        block_domain = """
(define (domain blocks)
  (:action move_block_left :parameters (?b))
  (:action move_block_up :parameters (?b))
)
"""
        reasoning_text = """b1 left x3:
(move_block_left b1) three times

b2 left x2:
(move_block_left b2) twice

b2 up x9:
(move_block_up b2) nine times

b4 up x6:
(move_block_up b4) six times
"""

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=block_domain)
        candidates = self.parser_module.extract_reasoning_candidates(reasoning_text, domain_text=block_domain)

        self.assertEqual(len(result.reasoning["actions"]), 20)
        self.assertTrue(any(len(candidate["actions"]) == 20 for candidate in candidates))
        self.assertEqual(result.reasoning["actions"].count("(move_block_left b1)"), 3)
        self.assertEqual(result.reasoning["actions"].count("(move_block_left b2)"), 2)
        self.assertEqual(result.reasoning["actions"].count("(move_block_up b2)"), 9)
        self.assertEqual(result.reasoning["actions"].count("(move_block_up b4)"), 6)

    def test_non_parenthesized_repeat_expands_domain_valid_actions(self) -> None:
        for text in ("58 times go_south b0", "repeat 58 times go_south b0", "go_south b0 repeated 58 times", "go_south b0 58 times"):
            with self.subTest(text=text):
                result = self.parser_module.parse_plan_text("", reasoning_text=text, domain_text=SAILING_DOMAIN)
                self.assertEqual(result.reasoning["actions"], ["(go_south b0)"] * 58)

    def test_composite_candidates_stop_at_alternative_boundary(self) -> None:
        reasoning_text = """b1 up 2
alternative:
b2 right 3
"""

        candidates = self.parser_module.extract_reasoning_candidates(reasoning_text, domain_text=BLOCK_DOMAIN)

        self.assertFalse(any(len(candidate["actions"]) == 5 for candidate in candidates))
        self.assertTrue(any(candidate["actions"] == ["(move_block_up b1)", "(move_block_up b1)"] for candidate in candidates))
        self.assertTrue(any(candidate["actions"] == ["(move_block_right b2)"] * 3 for candidate in candidates))

    def test_gpt_oss_block_grouping_reasoning_composes_pddl_steps(self) -> None:
        block_domain = """
(define (domain blocks)
  (:action move_block_left :parameters (?b - block))
  (:action move_block_right :parameters (?b - block))
  (:action move_block_up :parameters (?b - block))
  (:action move_block_down :parameters (?b - block))
)
"""
        reasoning_text = """We'll produce actions:

First move b1 left twice, down three times.

Sequence:

(move_block_left b1) -> (19,14)
(move_block_left b1) -> (18,14)
(move_block_down b1) -> (18,13)
(move_block_down b1) -> (18,12)
(move_block_down b1) -> (18,11) done.

Now b1 at goal.

b2: need left1, up6.

(move_block_left b2) -> (18,5)
Then six ups:

(move_block_up b2) -> (18,6)
(move_block_up b2) -> (18,7)
(move_block_up b2) -> (18,8)
(move_block_up b2) -> (18,9)
(move_block_up b2) -> (18,10)
(move_block_up b2) -> (18,11) done.

b3: right1, down3.

(move_block_right b3) -> (18,14)
(move_block_down b3) -> (18,13)
(move_block_down b3) -> (18,12)
(move_block_down b3) -> (18,11) done.

b4: right1, up3.

(move_block_right b4) -> (18,8)
(move_block_up b4) -> (18,9)
(move_block_up b4) -> (18,10)
(move_block_up b4) -> (18,11) done.
"""

        result = self.parser_module.parse_plan_text("", reasoning_text=reasoning_text, domain_text=block_domain)
        candidates = self.parser_module.extract_reasoning_candidates(reasoning_text, domain_text=block_domain)

        self.assertEqual(len(result.reasoning["actions"]), 20)
        self.assertTrue(any(candidate.get("composite") and len(candidate["actions"]) == 20 for candidate in candidates))
        self.assertEqual(result.reasoning["actions"].count("(move_block_left b1)"), 2)
        self.assertEqual(result.reasoning["actions"].count("(move_block_down b1)"), 3)
        self.assertEqual(result.reasoning["actions"].count("(move_block_up b2)"), 6)
        self.assertEqual(result.reasoning["actions"].count("(move_block_down b3)"), 3)
        self.assertEqual(result.reasoning["actions"].count("(move_block_up b4)"), 3)

    def test_non_parenthesized_repeat_infers_unique_problem_object(self) -> None:
        domain_text = """
(define (domain sailing)
  (:types boat person)
  (:action go_south :parameters (?b - boat))
)
"""
        problem_text = """
(define (problem p1)
  (:domain sailing)
  (:objects b0 - boat p0 - person)
)
"""

        result = self.parser_module.parse_plan_text(
            "",
            reasoning_text="58 times go_south",
            domain_text=domain_text,
            problem_text=problem_text,
        )

        self.assertEqual(result.reasoning["actions"], ["(go_south b0)"] * 58)
        self.assertIn("compressed_actions_expanded", result.reasoning["format_issues"])

    def test_non_parenthesized_repeat_does_not_infer_ambiguous_problem_object(self) -> None:
        domain_text = """
(define (domain sailing)
  (:types boat)
  (:action go_south :parameters (?b - boat))
)
"""
        problem_text = """
(define (problem p1)
  (:domain sailing)
  (:objects b0 b1 - boat)
)
"""

        result = self.parser_module.parse_plan_text(
            "",
            reasoning_text="58 times go_south",
            domain_text=domain_text,
            problem_text=problem_text,
        )

        self.assertEqual(result.reasoning["actions"], [])
        self.assertIn("ambiguous_compressed_action", result.reasoning["format_issues"])


if __name__ == "__main__":
    unittest.main()
