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

            self.assertEqual(raw_payload["raw_output"], "(move a b)")
            self.assertEqual(raw_payload["attempts"][0]["iteration"], 1)
            self.assertIn("messages", raw_payload["attempts"][0])
            self.assertIn("generation", raw_payload["attempts"][0])
            self.assertEqual(raw_payload["attempts"][0]["raw_output"], "(move a b)")
            self.assertNotIn("parsed_plan", raw_payload["attempts"][0])
            self.assertNotIn("validation_result", raw_payload["attempts"][0])
            self.assertEqual(parsed_payload["parsed_plan"]["actions"], ["(move a b)"])
            self.assertEqual(parsed_payload["attempts"][0]["parsed_plan"]["actions"], ["(move a b)"])
            self.assertNotIn("validation_result", parsed_payload["attempts"][0])
            self.assertTrue(scored_payload["solved"])
            self.assertEqual(scored_payload["attempts"][0]["iteration"], 1)
            self.assertEqual(scored_payload["attempts"][0]["validation_result"]["valid"], True)
            self.assertNotIn("messages", scored_payload["attempts"][0])


if __name__ == "__main__":
    unittest.main()
