"""Unit tests for the shared benchmark parser."""

import importlib.util
import unittest
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
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
        result = self.parser_module.parse_plan_text("")

        self.assertEqual(result.actions, [])
        self.assertEqual(result.reasoning, "")
        self.assertEqual(result.format_issues, ["empty_output"])

    def test_parse_extracts_reasoning_and_actions(self) -> None:
        raw_text = "I will first think through the problem.\n(move a b)\n(pick box1 room2)"

        result = self.parser_module.parse_plan_text(raw_text)

        self.assertEqual(result.reasoning, "I will first think through the problem.")
        self.assertEqual(result.actions, ["(move a b)", "(pick box1 room2)"])
        self.assertIn("reasoning_before_plan_removed", result.format_issues)

    def test_parse_removes_markdown_fences_and_numbering(self) -> None:
        raw_text = "```text\n1. (move a b)\n2. (pick box1 room2)\n```"

        result = self.parser_module.parse_plan_text(raw_text)

        self.assertEqual(result.actions, ["(move a b)", "(pick box1 room2)"])
        self.assertEqual(result.reasoning, "")
        self.assertIn("markdown_fences_removed", result.format_issues)

    def test_parse_extracts_actions_embedded_in_text(self) -> None:
        raw_text = "Plan: first do (move a b), then do (pick box1 room2)."

        result = self.parser_module.parse_plan_text(raw_text)

        self.assertEqual(result.actions, ["(move a b)", "(pick box1 room2)"])
        self.assertIn("plan_section_marker_removed", result.format_issues)
        self.assertIn("actions_embedded_in_text", result.format_issues)

    def test_parse_reports_missing_actions(self) -> None:
        result = self.parser_module.parse_plan_text("This is a detailed explanation, but not a plan.")

        self.assertEqual(result.actions, [])
        self.assertEqual(result.reasoning, "This is a detailed explanation, but not a plan.")
        self.assertIn("no_parenthesized_actions_found", result.format_issues)


if __name__ == "__main__":
    unittest.main()
