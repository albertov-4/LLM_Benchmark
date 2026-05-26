"""Smoke test for `runner/run_case.py`."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
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
        domain_file.write_text("(define (domain toy))", encoding="utf-8")
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
            self.assertEqual(parsed_payload["attempts"][0]["parsed_plan"]["actions"], ["(move wrong target)"])
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
            self.assertEqual(parsed_payload["attempts"][0]["parsed_plan"]["actions"], ["(move a b)"])
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
            self.assertNotIn("messages", scored_payload["attempts"][0])


if __name__ == "__main__":
    unittest.main()
