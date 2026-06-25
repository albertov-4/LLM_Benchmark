"""Smoke test for `runner/run_case.py`."""

from __future__ import annotations

import importlib.util
import sys
import json
import tempfile
import unittest
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class PrefixLengthValidator:
    """Validator test double that accepts a specific action-prefix length."""

    def __init__(self, valid_prefix_length: int | None = None) -> None:
        self.valid_prefix_length = valid_prefix_length
        self.seen_plan_texts: list[str] = []

    def validate(self, domain_file: str, problem_file: str, plan_text: str) -> dict[str, object]:
        self.seen_plan_texts.append(plan_text)
        actions = [line.strip() for line in plan_text.splitlines() if line.strip()]
        plan_length = len(actions)
        valid = self.valid_prefix_length is not None and plan_length == self.valid_prefix_length
        return {
            "valid": valid,
            "status": "valid" if valid else "invalid",
            "error_type": None if valid else "invalid_precondition",
            "feedback_text": (
                "Plan validated successfully."
                if valid
                else "Prefix did not validate."
            ),
            "failed_step": None if valid else plan_length,
            "failed_action": None if valid or not actions else actions[-1],
            "goal_satisfied": True if valid else False,
            "plan_length": plan_length,
            "validation_time_ms": 1,
            "raw_validator_output": f"mock-prefix-length-{plan_length}",
            "details": {
                "domain_file": domain_file,
                "problem_file": problem_file,
                "validator_kind": "prefix_length_mock",
            },
        }


class RunCaseSmokeTest(unittest.TestCase):
    def _build_task_and_protocol(self, run_case_module, max_iterations: int = 1):
        tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)

        tmp_path = Path(tmp_dir.name)
        domain_file = tmp_path / "domain.pddl"
        problem_file = tmp_path / "problem.pddl"
        domain_file.write_text(
            """
(define (domain toy)
  (:action move
    :parameters (?from ?to)
    :precondition (and)
    :effect (and)
  )
)
""",
            encoding="utf-8",
        )
        problem_file.write_text("(define (problem toy-problem))", encoding="utf-8")

        task_spec = run_case_module.TaskSpec(
            task_family="toy",
            tier="easy",
            instance_id="instance-01",
            domain_file=str(domain_file),
            problem_file=str(problem_file),
        )
        protocol_spec = run_case_module.ProtocolSpec(
            protocol_id="direct_plan" if max_iterations == 1 else "iterative_repair",
            max_iterations=max_iterations,
            require_final_plan_only=True,
        )
        return task_spec, protocol_spec

    def test_run_case_solves_with_mock_components(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_module",
            framework_root / "runner" / "run_case.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_mock_adapter_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_mock_validator_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        adapter = mock_adapter_module.MockAdapter(scripted_outputs=["(move a b)"])
        validator = mock_validator_module.MockValidator(valid_on_attempt=1)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)

        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=adapter,
            validator=validator,
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        self.assertTrue(result.solved)
        self.assertEqual(result.iterations_used, 1)
        self.assertEqual(result.metrics["validity_at_1"], True)
        self.assertEqual(result.metrics["plan_length"], 1)
        self.assertEqual(result.metrics["repair_success"], False)
        self.assertEqual(result.metrics["hit_iteration_limit"], False)
        self.assertGreaterEqual(result.generation_time_seconds, 0.0)

    def test_run_case_succeeds_after_repair(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_repair_module",
            framework_root / "runner" / "run_case.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_mock_adapter_repair_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_mock_validator_repair_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        adapter = mock_adapter_module.MockAdapter(
            scripted_outputs=["(move wrong target)", "(move a b)"]
        )
        validator = mock_validator_module.MockValidator(valid_on_attempt=2)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=3)

        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=adapter,
            validator=validator,
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        self.assertTrue(result.solved)
        self.assertEqual(result.iterations_used, 2)
        self.assertEqual(result.metrics["validity_at_1"], False)
        self.assertEqual(result.metrics["repair_success"], True)
        self.assertEqual(result.metrics["iterations_to_valid"], 2)
        self.assertEqual(result.metrics["hit_iteration_limit"], False)

    def test_run_case_includes_feedback_prompt_during_repair(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_feedback_module",
            framework_root / "runner" / "run_case.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_mock_adapter_feedback_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_mock_validator_feedback_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        adapter = mock_adapter_module.MockAdapter(
            scripted_outputs=["(move wrong target)", "(move a b)"]
        )
        validator = mock_validator_module.MockValidator(valid_on_attempt=2)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=3)
        protocol_spec.include_external_feedback = True

        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=adapter,
            validator=validator,
            task_spec=task_spec,
            protocol_spec=protocol_spec,
            feedback_prompt="USE THE VALIDATOR FEEDBACK TEMPLATE.",
        )

        self.assertTrue(result.solved)
        self.assertGreaterEqual(len(adapter.last_messages), 2)
        self.assertIn("USE THE VALIDATOR FEEDBACK TEMPLATE.", adapter.last_messages[-1]["content"])

    def test_run_case_stops_after_iteration_budget(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_budget_module",
            framework_root / "runner" / "run_case.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_mock_adapter_budget_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_mock_validator_budget_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        adapter = mock_adapter_module.MockAdapter(
            scripted_outputs=["(move wrong target)", "(move wrong target)"]
        )
        validator = mock_validator_module.MockValidator(valid_on_attempt=99)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=2)

        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=adapter,
            validator=validator,
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        self.assertFalse(result.solved)
        self.assertEqual(result.iterations_used, 2)
        self.assertEqual(result.stopped_by_iteration_limit, True)
        self.assertEqual(result.metrics["validity_at_k"], False)
        self.assertEqual(result.metrics["hit_iteration_limit"], True)
        self.assertEqual(result.validation_result["error_type"], "invalid_precondition")

    def test_run_case_reports_parse_error(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_parse_module",
            framework_root / "runner" / "run_case.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_mock_adapter_parse_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_mock_validator_parse_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        adapter = mock_adapter_module.MockAdapter(scripted_outputs=["this is not a plan"])
        validator = mock_validator_module.MockValidator(valid_on_attempt=1)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)

        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=adapter,
            validator=validator,
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        self.assertFalse(result.solved)
        self.assertEqual(result.validation_result["status"], "parse_error")
        self.assertEqual(result.validation_result["error_type"], "syntax_error")
        self.assertEqual(result.metrics["validity_at_k"], False)
        self.assertEqual(result.metrics["error_type"], "syntax_error")

    def test_run_case_persists_completed_attempts_when_later_generation_fails(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_generation_error_module",
            framework_root / "runner" / "run_case.py",
        )

        class FailingSecondGenerationAdapter:
            def __init__(self):
                self.call_count = 0

            def generate(self, messages):
                self.call_count += 1
                if self.call_count == 1:
                    return {
                        "model_id": "mock_model",
                        "raw_text": "(move wrong target)",
                        "usage": {},
                        "latency_s": 0.0,
                    }
                raise TimeoutError("Request timed out.")

        adapter = FailingSecondGenerationAdapter()
        validator = PrefixLengthValidator(valid_prefix_length=None)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=3)
        protocol_spec.include_external_feedback = True

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_case_module.run_case(
                model_id="mock_model",
                adapter=adapter,
                validator=validator,
                task_spec=task_spec,
                protocol_spec=protocol_spec,
                output_root=tmp_dir,
                run_id="test-generation-error",
            )

            self.assertFalse(result.solved)
            self.assertEqual(result.iterations_used, 2)
            self.assertFalse(result.stopped_by_iteration_limit)
            self.assertEqual(result.validation_result["status"], "generation_error")
            self.assertEqual(result.validation_result["error_type"], "TimeoutError")
            self.assertEqual(result.metrics["error_type"], "TimeoutError")

            raw_payload = json.loads(Path(result.raw_output_path or "").read_text(encoding="utf-8"))
            parsed_payload = json.loads(Path(result.parsed_output_path or "").read_text(encoding="utf-8"))
            scored_payload = json.loads(Path(result.scored_output_path or "").read_text(encoding="utf-8"))

            self.assertEqual(len(raw_payload["attempts"]), 2)
            self.assertEqual(raw_payload["attempts"][0]["generation"]["raw_text"], "(move wrong target)")
            self.assertEqual(raw_payload["attempts"][1]["generation"]["raw_text"], "")
            self.assertIn("TimeoutError", raw_payload["attempts"][1]["generation"]["stream_error"])
            self.assertEqual(parsed_payload["attempts"][0]["parsed_plan"]["raw"]["actions"], ["(move wrong target)"])
            self.assertIsNone(parsed_payload["attempts"][1]["parsed_plan"])
            self.assertEqual(scored_payload["attempts"][1]["validation_result"]["status"], "generation_error")
            self.assertEqual(scored_payload["attempts"][1]["validation_result"]["error_type"], "TimeoutError")

    def test_run_case_solves_when_action_prefix_validates(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_prefix_valid_module",
            framework_root / "runner" / "run_case.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_mock_adapter_prefix_valid_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )

        adapter = mock_adapter_module.MockAdapter(
            scripted_outputs=[
                "\n".join(
                    [
                        "(move a b)",
                        "(move b c)",
                        "(move c d)",
                        "(move d e)",
                    ]
                )
            ]
        )
        validator = PrefixLengthValidator(valid_prefix_length=2)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)

        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=adapter,
            validator=validator,
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        first_attempt = result.attempts[0]
        self.assertTrue(result.solved)
        self.assertEqual(result.validation_result["plan_length"], 2)
        self.assertEqual(result.metrics["plan_length"], 2)
        self.assertEqual(first_attempt["first_valid_prefix_length"], 2)
        self.assertEqual(first_attempt["extra_actions_after_first_valid"], 2)
        self.assertFalse(first_attempt["final_plan_valid"])
        self.assertNotIn("prefix_validation_results", first_attempt)
        self.assertEqual(len(validator.seen_plan_texts), 4)
        self.assertEqual(
            first_attempt["first_valid_plan_text"],
            "(move a b)\n(move b c)",
        )

    def test_run_case_uses_full_plan_result_when_no_prefix_validates(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_prefix_invalid_module",
            framework_root / "runner" / "run_case.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_mock_adapter_prefix_invalid_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )

        adapter = mock_adapter_module.MockAdapter(
            scripted_outputs=[
                "\n".join(
                    [
                        "(move a b)",
                        "(move b c)",
                        "(move c d)",
                    ]
                )
            ]
        )
        validator = PrefixLengthValidator(valid_prefix_length=None)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)
        protocol_spec.include_external_feedback = True

        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=adapter,
            validator=validator,
            task_spec=task_spec,
            protocol_spec=protocol_spec,
            feedback_prompt="USE FEEDBACK.",
        )

        first_attempt = result.attempts[0]
        self.assertFalse(result.solved)
        self.assertEqual(result.validation_result["plan_length"], 3)
        self.assertEqual(result.validation_result["error_type"], "invalid_precondition")
        self.assertEqual(first_attempt["first_valid_prefix_length"], None)
        self.assertEqual(first_attempt["first_valid_plan_text"], None)
        self.assertEqual(first_attempt["extra_actions_after_first_valid"], None)
        self.assertFalse(first_attempt["final_plan_valid"])
        self.assertNotIn("prefix_validation_results", first_attempt)
        self.assertEqual(len(validator.seen_plan_texts), 3)
        self.assertIn("USE FEEDBACK.", first_attempt["feedback_to_next_iteration"])
        self.assertIn("invalid_precondition", first_attempt["feedback_to_next_iteration"])

    def test_repair_feedback_uses_valid_reasoning_plan_as_hint(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_reasoning_feedback_module",
            framework_root / "runner" / "run_case.py",
        )

        feedback = run_case_module.build_repair_feedback(
            {
                "valid": False,
                "status": "invalid",
                "error_type": "unsatisfied_goal",
                "feedback_text": "Goal not satisfied.",
                "details": {},
            },
            reasoning_validation_result={"valid": True},
            reasoning_plan_text="(move a b)\n(pick box1 room2)",
        )

        self.assertIn("valid action sequence was decoded from the reasoning text", feedback)
        self.assertIn("final answer/raw plan did not validate", feedback)
        self.assertNotIn("(move a b)\n(pick box1 room2)", feedback)

    def test_run_case_stores_reasoning_plan_separately(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_reasoning_module",
            framework_root / "runner" / "run_case.py",
        )

        class ReasoningAdapter:
            model_id = "reasoning-model"

            def generate(self, messages):
                return {
                    "model_id": self.model_id,
                    "raw_text": "Visible parser reasoning\n(move a b)",
                    "reasoning_text": "Thinking.\n(move a b)",
                    "usage": {},
                    "latency_s": 0.0,
                }

        adapter = ReasoningAdapter()
        validator = PrefixLengthValidator(valid_prefix_length=1)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_case_module.run_case(
                model_id="mock_model",
                adapter=adapter,
                validator=validator,
                task_spec=task_spec,
                protocol_spec=protocol_spec,
                output_root=tmp_dir,
                run_id="test-reasoning",
            )

            parsed_payload = json.loads(Path(result.parsed_output_path or "").read_text(encoding="utf-8"))
            parsed_plan = parsed_payload["attempts"][0]["parsed_plan"]

            self.assertTrue(result.solved)
            scored_payload = json.loads(Path(result.scored_output_path or "").read_text(encoding="utf-8"))
            scored_attempt = scored_payload["attempts"][0]

            self.assertEqual(result.attempts[0]["parsed_plan"]["reasoning"]["actions"], ["(move a b)"])
            self.assertEqual(parsed_plan["reasoning"]["actions"], ["(move a b)"])
            self.assertEqual(parsed_plan["reasoning"]["source_ref"], {"artifact": "raw", "field": "generation.reasoning_text"})
            self.assertTrue(parsed_plan["raw"]["contains_reasoning"])
            self.assertTrue(scored_attempt["validation_result"]["valid"])
            self.assertTrue(scored_attempt["reasoning_validation_result"]["valid"])
            self.assertEqual(scored_attempt["reasoning_first_valid_prefix_length"], 1)
            self.assertTrue(scored_attempt["reasoning_final_plan_valid"])


    def test_run_case_selects_valid_composite_reasoning_candidate_for_feedback(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_composite_reasoning_module",
            framework_root / "runner" / "run_case.py",
        )

        class CompositeReasoningAdapter:
            model_id = "reasoning-model"

            def generate(self, messages):
                return {
                    "model_id": self.model_id,
                    "raw_text": "(move_block_up b9)",
                    "reasoning_text": "b1 up x2:\n(move_block_up b1) twice\n\nb2 up x3:\n(move_block_up b2) three times",
                    "usage": {},
                    "latency_s": 0.0,
                }

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            domain_file = tmp_path / "domain.pddl"
            problem_file = tmp_path / "problem.pddl"
            domain_file.write_text(
                """
(define (domain blocks)
  (:action move_block_up :parameters (?b))
)
""",
                encoding="utf-8",
            )
            problem_file.write_text("(define (problem blocks-problem))", encoding="utf-8")
            task_spec = run_case_module.TaskSpec(
                task_family="blocks",
                tier="easy",
                instance_id="pfile1",
                domain_file=str(domain_file),
                problem_file=str(problem_file),
            )
            protocol_spec = run_case_module.ProtocolSpec(
                protocol_id="iterative_repair",
                max_iterations=1,
                require_final_plan_only=True,
                include_external_feedback=True,
            )

            result = run_case_module.run_case(
                model_id="mock_model",
                adapter=CompositeReasoningAdapter(),
                validator=PrefixLengthValidator(valid_prefix_length=5),
                task_spec=task_spec,
                protocol_spec=protocol_spec,
            )

        attempt = result.attempts[0]
        self.assertFalse(result.solved)
        self.assertTrue(attempt["reasoning_final_plan_valid"])
        self.assertEqual(attempt["reasoning_first_valid_prefix_length"], 5)
        self.assertEqual(len(attempt["parsed_plan"]["reasoning"]["actions"]), 5)
        self.assertIn("valid action sequence was decoded from the reasoning text", attempt["feedback_to_next_iteration"])
        self.assertIn("(move_block_up b2)", attempt["feedback_to_next_iteration"])

    def test_run_case_persists_raw_parsed_and_scored_outputs(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_persist_module",
            framework_root / "runner" / "run_case.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_mock_adapter_persist_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_mock_validator_persist_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        adapter = mock_adapter_module.MockAdapter(scripted_outputs=["(move a b)"])
        validator = mock_validator_module.MockValidator(valid_on_attempt=1)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = run_case_module.run_case(
                model_id="mock_model",
                adapter=adapter,
                validator=validator,
                task_spec=task_spec,
                protocol_spec=protocol_spec,
                output_root=tmp_dir,
                run_id="test-run",
            )

            self.assertIsNotNone(result.raw_output_path)
            self.assertIsNotNone(result.parsed_output_path)
            self.assertIsNotNone(result.scored_output_path)

            raw_path = Path(result.raw_output_path or "")
            parsed_path = Path(result.parsed_output_path or "")
            scored_path = Path(result.scored_output_path or "")

            self.assertTrue(raw_path.exists())
            self.assertTrue(parsed_path.exists())
            self.assertTrue(scored_path.exists())
            self.assertIn("test-run", raw_path.parts)
            self.assertIn("test-run", parsed_path.parts)
            self.assertIn("test-run", scored_path.parts)
            self.assertIn("easy", raw_path.parts)
            self.assertIn("easy", parsed_path.parts)
            self.assertIn("easy", scored_path.parts)

            raw_payload = json.loads(raw_path.read_text(encoding="utf-8"))
            parsed_payload = json.loads(parsed_path.read_text(encoding="utf-8"))
            scored_payload = json.loads(scored_path.read_text(encoding="utf-8"))

            self.assertGreaterEqual(result.generation_time_seconds, 0.0)
            self.assertIn("generation_time_seconds", raw_payload)
            self.assertIn("generation_time_seconds", parsed_payload)
            self.assertIn("generation_time_seconds", scored_payload)
            self.assertGreaterEqual(raw_payload["generation_time_seconds"], 0.0)
            self.assertGreaterEqual(parsed_payload["generation_time_seconds"], 0.0)
            self.assertGreaterEqual(scored_payload["generation_time_seconds"], 0.0)
            self.assertNotIn("raw_output", raw_payload)
            self.assertNotIn("raw_generations", raw_payload)
            self.assertEqual(raw_payload["attempts"][0]["iteration"], 1)
            self.assertIn("generation_time_seconds", raw_payload["attempts"][0])
            self.assertIn("messages", raw_payload["attempts"][0])
            self.assertIn("generation", raw_payload["attempts"][0])
            self.assertNotIn("raw_output", raw_payload["attempts"][0])
            self.assertNotIn("model_id", raw_payload["attempts"][0]["generation"])
            self.assertIn("generation_time_seconds", raw_payload["attempts"][0]["generation"])
            self.assertEqual(raw_payload["attempts"][0]["generation"]["raw_text"], "(move a b)")
            self.assertNotIn("parsed_plan", raw_payload["attempts"][0])
            self.assertNotIn("validation_result", raw_payload["attempts"][0])
            self.assertNotIn("parsed_plan", parsed_payload)
            self.assertIn("generation_time_seconds", parsed_payload["attempts"][0])
            self.assertEqual(parsed_payload["attempts"][0]["parsed_plan"]["raw"]["actions"], ["(move a b)"])
            self.assertNotIn("validation_result", parsed_payload["attempts"][0])
            self.assertTrue(scored_payload["solved"])
            self.assertIn("metrics", scored_payload)
            self.assertIn("raw_output_path", scored_payload)
            self.assertIn("parsed_output_path", scored_payload)
            self.assertNotIn("raw_output", scored_payload)
            self.assertNotIn("parsed_plan", scored_payload)
            self.assertNotIn("validation_result", scored_payload)
            self.assertNotIn("scored_output_path", scored_payload)
            self.assertEqual(scored_payload["attempts"][0]["iteration"], 1)
            self.assertIn("generation_time_seconds", scored_payload["attempts"][0])
            self.assertEqual(scored_payload["attempts"][0]["validation_result"]["valid"], True)
            self.assertNotIn("prefix_validation_results", scored_payload["attempts"][0])
            self.assertEqual(scored_payload["attempts"][0]["first_valid_prefix_length"], 1)
            self.assertEqual(scored_payload["attempts"][0]["first_valid_plan_text"], "(move a b)")
            self.assertEqual(scored_payload["attempts"][0]["final_plan_valid"], True)
            self.assertEqual(scored_payload["attempts"][0]["extra_actions_after_first_valid"], 0)
            self.assertIsNone(scored_payload["attempts"][0]["reasoning_validation_result"])
            self.assertIsNone(scored_payload["attempts"][0]["reasoning_first_valid_prefix_length"])
            self.assertNotIn("messages", scored_payload["attempts"][0])

    def test_reasoning_candidate_selection_prefers_valid_candidate(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_reasoning_candidate_valid_module",
            framework_root / "runner" / "run_case.py",
        )

        class ReasoningAdapter:
            def generate(self, messages):
                return {
                    "raw_text": "(move raw bad)",
                    "reasoning_text": """Try:
(move a b)

Final answer:
(move a b)
(move b c)""",
                    "usage": {},
                    "latency_s": 0.0,
                }

        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)
        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=ReasoningAdapter(),
            validator=PrefixLengthValidator(valid_prefix_length=2),
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        attempt = result.attempts[0]
        self.assertFalse(result.solved)
        self.assertTrue(attempt["reasoning_final_plan_valid"])
        self.assertEqual(attempt["reasoning_valid_candidate_count"], 1)
        self.assertEqual(attempt["parsed_plan"]["reasoning"]["actions"], ["(move a b)", "(move b c)"])

    def test_reasoning_candidate_selection_prefers_final_marker_between_valid_candidates(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_reasoning_candidate_final_module",
            framework_root / "runner" / "run_case.py",
        )

        class ReasoningAdapter:
            def generate(self, messages):
                return {
                    "raw_text": "this is not a plan",
                    "reasoning_text": """Candidate:
(move x y)

Final answer:
(move a b)""",
                    "usage": {},
                    "latency_s": 0.0,
                }

        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)
        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=ReasoningAdapter(),
            validator=PrefixLengthValidator(valid_prefix_length=1),
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        attempt = result.attempts[0]
        self.assertEqual(attempt["reasoning_valid_candidate_count"], 2)
        self.assertEqual(attempt["parsed_plan"]["reasoning"]["actions"], ["(move a b)"])

    def test_reasoning_candidate_selection_penalizes_truncated_candidate(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_reasoning_candidate_truncated_module",
            framework_root / "runner" / "run_case.py",
        )

        class ReasoningAdapter:
            def generate(self, messages):
                return {
                    "raw_text": "this is not a plan",
                    "reasoning_text": """Final answer:
(move x y)
(move

Alternative:
(move a b)""",
                    "usage": {},
                    "latency_s": 0.0,
                }

        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)
        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=ReasoningAdapter(),
            validator=PrefixLengthValidator(valid_prefix_length=1),
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        attempt = result.attempts[0]
        self.assertFalse(attempt["reasoning_selected_candidate_truncated"])
        self.assertEqual(attempt["parsed_plan"]["reasoning"]["actions"], ["(move a b)"])


    def test_reasoning_candidate_selection_promotes_valid_prefix(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_reasoning_prefix_module",
            framework_root / "runner" / "run_case.py",
        )

        class ReasoningAdapter:
            def generate(self, messages):
                return {
                    "raw_text": "(move raw bad)",
                    "reasoning_text": """Final answer:
(move a b)
(move b c)
(move c d)""",
                    "usage": {},
                    "latency_s": 0.0,
                }

        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)
        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=ReasoningAdapter(),
            validator=PrefixLengthValidator(valid_prefix_length=2),
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        attempt = result.attempts[0]
        self.assertFalse(result.solved)
        self.assertFalse(attempt["reasoning_final_plan_valid"])
        self.assertEqual(attempt["reasoning_first_valid_prefix_length"], 2)
        self.assertEqual(attempt["reasoning_extra_actions_after_first_valid"], 1)
        self.assertTrue(attempt["reasoning_validation_result"]["valid"])
        self.assertEqual(attempt["parsed_plan"]["reasoning"]["actions"], ["(move a b)", "(move b c)"])

    def test_reasoning_candidate_selection_keeps_best_invalid_candidate_diagnostic(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_reasoning_candidate_invalid_module",
            framework_root / "runner" / "run_case.py",
        )

        class ReasoningAdapter:
            def generate(self, messages):
                return {
                    "raw_text": "this is not a plan",
                    "reasoning_text": """Candidate:
(move x y)

Final answer:
(move a b)
(move b c)""",
                    "usage": {},
                    "latency_s": 0.0,
                }

        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=1)
        result = run_case_module.run_case(
            model_id="mock_model",
            adapter=ReasoningAdapter(),
            validator=PrefixLengthValidator(valid_prefix_length=None),
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        attempt = result.attempts[0]
        self.assertFalse(result.solved)
        self.assertEqual(attempt["reasoning_valid_candidate_count"], 0)
        self.assertFalse(attempt["reasoning_validation_result"]["valid"])
        self.assertEqual(attempt["parsed_plan"]["reasoning"]["actions"], ["(move a b)", "(move b c)"])



    def test_repair_feedback_starts_on_second_iteration_and_defaults_last_only(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_case_module = _load_module(
            "benchmark_framework_test_run_case_repair_history_module",
            framework_root / "runner" / "run_case.py",
        )

        class CapturingAdapter:
            model_id = "capture-model"

            def __init__(self) -> None:
                self.outputs = ["(move a b)", "(move b c)", "(move c d)"]
                self.messages_by_call = []

            def generate(self, messages):
                self.messages_by_call.append([dict(message) for message in messages])
                raw_text = self.outputs[len(self.messages_by_call) - 1]
                return {"raw_text": raw_text, "usage": {}, "latency_s": 0.0}

        adapter = CapturingAdapter()
        validator = PrefixLengthValidator(valid_prefix_length=None)
        task_spec, protocol_spec = self._build_task_and_protocol(run_case_module, max_iterations=3)
        protocol_spec.include_external_feedback = True

        run_case_module.run_case(
            model_id="mock_model",
            adapter=adapter,
            validator=validator,
            task_spec=task_spec,
            protocol_spec=protocol_spec,
        )

        self.assertEqual(len(adapter.messages_by_call), 3)
        self.assertEqual(len(adapter.messages_by_call[0]), 1)
        second_feedback = adapter.messages_by_call[1][-1]["content"]
        third_feedback = adapter.messages_by_call[2][-1]["content"]
        self.assertIn("[PREVIOUS FINAL ANSWER]", second_feedback)
        self.assertIn("(move a b)", second_feedback)
        self.assertIn("(move b c)", third_feedback)
        self.assertNotIn("(move a b)", third_feedback)

if __name__ == "__main__":
    unittest.main()
