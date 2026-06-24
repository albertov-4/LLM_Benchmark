"""Tests for iterative repair feedback text generation."""

from __future__ import annotations

import pathlib
import sys
import unittest


sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from evaluators.repair_feedback import RepairFeedbackConfig, build_repair_feedback


def _attempt(
    *,
    raw_actions: list[str] | None = None,
    reasoning_actions: list[str] | None = None,
    raw_text: str = "(move a b)",
    reasoning_text: str = "",
    valid: bool = False,
    error_type: str = "invalid_precondition",
    feedback_text: str = "The plan contains an action whose preconditions are not satisfied.",
    reasoning_valid: bool = False,
    format_issues: list[str] | None = None,
) -> dict[str, object]:
    return {
        "generation": {"raw_text": raw_text, "reasoning_text": reasoning_text},
        "parsed_plan": {
            "raw": {"actions": raw_actions or ["(move a b)"], "format_issues": format_issues or []},
            "reasoning": {"actions": reasoning_actions or []},
        },
        "validation_result": {
            "valid": valid,
            "status": "valid" if valid else "invalid",
            "error_type": error_type,
            "feedback_text": feedback_text,
            "failed_step": 1,
            "failed_action": "(move a b)",
            "goal_satisfied": error_type != "unsatisfied_goal",
            "details": {"raw_format_issues": format_issues or []},
        },
        "final_plan_valid": valid,
        "reasoning_validation_result": {"valid": reasoning_valid},
        "reasoning_final_plan_valid": reasoning_valid,
    }


class RepairFeedbackTest(unittest.TestCase):
    def test_includes_previous_final_and_reasoning_when_available(self) -> None:
        feedback = build_repair_feedback(
            attempt_record=_attempt(
                raw_text="FINAL",
                reasoning_text="REASONING",
                reasoning_actions=["(move a b)"],
            )
        )

        self.assertIn("[PREVIOUS FINAL ANSWER]", feedback)
        self.assertIn("FINAL", feedback)
        self.assertIn("[PREVIOUS REASONING TEXT]", feedback)
        self.assertIn("REASONING", feedback)

    def test_omits_reasoning_block_when_absent(self) -> None:
        feedback = build_repair_feedback(attempt_record=_attempt(reasoning_text=""))

        self.assertNotIn("[PREVIOUS REASONING TEXT]", feedback)

    def test_visible_feedback_does_not_dump_internal_action_metrics(self) -> None:
        feedback = build_repair_feedback(
            attempt_record=_attempt(
                raw_actions=["(move a b)"],
                reasoning_actions=["(move b c)"],
                reasoning_text="I considered a different move.",
            )
        )

        self.assertNotIn("Raw actions:", feedback)
        self.assertNotIn("Reasoning actions:", feedback)
        self.assertNotIn("LCS", feedback)
        self.assertNotIn("action_bag_overlap", feedback)

    def test_reasoning_to_final_transfer_diagnosis(self) -> None:
        feedback = build_repair_feedback(
            attempt_record=_attempt(
                reasoning_text="The valid plan is here.",
                reasoning_actions=["(move a b)"],
                reasoning_valid=True,
            )
        )

        self.assertIn("valid action sequence was decoded from the reasoning text", feedback)
        self.assertIn("transfer problem", feedback)
        self.assertIn("Return only the complete PDDL action sequence", feedback)

    def test_validator_only_without_reasoning(self) -> None:
        feedback = build_repair_feedback(attempt_record=_attempt(reasoning_actions=[]))

        self.assertIn("No separate reasoning plan was available", feedback)
        self.assertIn("validator result", feedback)

    def test_coherent_but_invalid_plan_diagnosis(self) -> None:
        feedback = build_repair_feedback(
            attempt_record=_attempt(
                raw_actions=["(move a b)", "(move b c)"],
                reasoning_actions=["(move a b)", "(move b c)"],
                reasoning_text="(move a b)\n(move b c)",
            )
        )

        self.assertIn("coherent with each other", feedback)
        self.assertIn("plan itself", feedback)

    def test_raw_reasoning_divergence_diagnosis(self) -> None:
        feedback = build_repair_feedback(
            attempt_record=_attempt(
                raw_actions=["(move a b)", "(move b c)", "(move c d)"],
                reasoning_actions=["(pick box room)", "(place box target)"],
                reasoning_text="(pick box room)\n(place box target)",
            )
        )

        self.assertIn("did not closely reflect", feedback)
        self.assertIn("one coherent plan", feedback)

    def test_ordering_precondition_diagnosis(self) -> None:
        feedback = build_repair_feedback(
            attempt_record=_attempt(
                raw_actions=["(move a b)", "(move b c)"],
                reasoning_actions=["(move b c)", "(move a b)"],
                reasoning_text="(move b c)\n(move a b)",
            )
        )

        self.assertIn("mostly the same actions", feedback)
        self.assertIn("order", feedback)

    def test_truncates_long_raw_and_reasoning_text(self) -> None:
        feedback = build_repair_feedback(
            attempt_record=_attempt(
                raw_text="R" * 80,
                reasoning_text="T" * 80,
                reasoning_actions=["(move a b)"],
            ),
            config=RepairFeedbackConfig(max_raw_chars=30, max_reasoning_chars=30),
        )

        self.assertIn("[truncated]", feedback)


if __name__ == "__main__":
    unittest.main()
