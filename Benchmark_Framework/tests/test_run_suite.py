"""Smoke test for `runner/run_suite.py`."""

import importlib.util
import shutil
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


def _resolve_validate_command(framework_root: Path) -> str | None:
    path_command = shutil.which("Validate")
    if path_command:
        return path_command

    workspace_root = framework_root.parent
    candidate_paths = [
        workspace_root / "VAL" / "build" / "win64" / "mingw" / "Debug" / "bin" / "Validate.exe",
        workspace_root / "VAL" / "build" / "win64" / "mingw" / "Debug" / "install" / "bin" / "Validate.exe",
        workspace_root.parent / "VAL" / "build" / "win64" / "mingw" / "Debug" / "bin" / "Validate.exe",
        workspace_root.parent / "VAL" / "build" / "win64" / "mingw" / "Debug" / "install" / "bin" / "Validate.exe",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    return None


class RunSuiteSmokeTest(unittest.TestCase):
    def test_run_suite_aggregates_mock_results(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_suite_mock_adapter_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_suite_mock_validator_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
        result = run_suite_module.run_suite(
            tasks_root=fixtures_root / "tasks",
            protocols_root=fixtures_root / "protocols",
            model_registry_path=fixtures_root / "models" / "model_registry.yaml",
            adapter_factory=mock_adapter_module.build_mock_adapter_for_suite,
            validator_factory=mock_validator_module.build_mock_validator_for_suite,
        )

        self.assertEqual(result["summary"]["num_jobs"], 1)
        self.assertEqual(len(result["suite_results"]), 1)
        self.assertIn("generation_time_seconds", result["suite_results"][0])
        self.assertGreaterEqual(result["suite_results"][0]["generation_time_seconds"], 0.0)
        self.assertEqual(result["aggregate_results"]["overall"]["num_solved"], 1)
        self.assertEqual(result["aggregate_results"]["by_model"]["mock_model"]["solve_rate"], 1.0)

    def test_run_suite_reports_orchestration_errors_as_empty_for_mock_flow(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_no_error_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_suite_mock_adapter_no_error_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_suite_mock_validator_no_error_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
        result = run_suite_module.run_suite(
            tasks_root=fixtures_root / "tasks",
            protocols_root=fixtures_root / "protocols",
            model_registry_path=fixtures_root / "models" / "model_registry.yaml",
            adapter_factory=mock_adapter_module.build_mock_adapter_for_suite,
            validator_factory=mock_validator_module.build_mock_validator_for_suite,
        )

        self.assertEqual(result["orchestration_errors"], [])

    def test_run_suite_loads_prompt_bundle_for_one_job(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_prompt_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_suite_prompt_adapter_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_suite_prompt_validator_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        created_adapters = []

        def adapter_factory(model_entry, protocol_config):
            adapter = mock_adapter_module.MockAdapter(
                scripted_outputs=["(move a b)"],
                model_id=str(model_entry.get("model_id", "mock_model")),
            )
            created_adapters.append(adapter)
            return adapter

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts_root = Path(tmp_dir)
            (prompts_root / "system.txt").write_text("SYSTEM PROMPT", encoding="utf-8")
            (prompts_root / "toy.txt").write_text("TOY DOMAIN PROMPT", encoding="utf-8")
            (prompts_root / "feedback.txt").write_text("FEEDBACK PROMPT", encoding="utf-8")
            protocols_root = prompts_root / "protocols"
            protocols_root.mkdir()
            (protocols_root / "direct_plan.yaml").write_text(
                "\n".join(
                    [
                        "protocol_id: direct_plan",
                        "prompting:",
                        "  use_system_prompt: true",
                        "  include_domain_prompt: true",
                        "  include_examples: false",
                        "  include_chain_of_thought: false",
                        "  include_external_feedback: false",
                        "evaluation:",
                        "  max_iterations: 1",
                        "  require_final_plan_only: true",
                    ]
                ),
                encoding="utf-8",
            )

            fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
            result = run_suite_module.run_suite(
                tasks_root=fixtures_root / "tasks",
                protocols_root=protocols_root,
                prompts_root=prompts_root,
                model_registry_path=fixtures_root / "models" / "model_registry.yaml",
                adapter_factory=adapter_factory,
                validator_factory=mock_validator_module.build_mock_validator_for_suite,
            )

        self.assertEqual(result["orchestration_errors"], [])
        self.assertEqual(len(created_adapters), 1)
        messages = created_adapters[0].last_messages
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("SYSTEM PROMPT", messages[0]["content"])
        self.assertIn("TOY DOMAIN PROMPT", messages[1]["content"])
        self.assertIn("FINAL ANSWER FORMAT", messages[1]["content"])

    def test_run_suite_filters_one_protocol_id(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_protocol_filter_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_suite_protocol_filter_adapter_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_suite_protocol_filter_validator_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        def adapter_factory(model_entry, protocol_config):
            return mock_adapter_module.MockAdapter(
                scripted_outputs=["(move a b)"],
                model_id=str(model_entry.get("model_id", "mock_model")),
            )

        protocol_body = "\n".join(
            [
                "prompting:",
                "  use_system_prompt: true",
                "  include_domain_prompt: true",
                "  include_examples: false",
                "  include_chain_of_thought: false",
                "  include_external_feedback: false",
                "evaluation:",
                "  max_iterations: 1",
                "  require_final_plan_only: true",
            ]
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            prompts_root = Path(tmp_dir)
            (prompts_root / "system.txt").write_text("SYSTEM PROMPT", encoding="utf-8")
            (prompts_root / "toy.txt").write_text("TOY DOMAIN PROMPT", encoding="utf-8")
            (prompts_root / "feedback.txt").write_text("FEEDBACK PROMPT", encoding="utf-8")
            protocols_root = prompts_root / "protocols"
            protocols_root.mkdir()
            (protocols_root / "direct_plan.yaml").write_text(
                "protocol_id: direct_plan\n" + protocol_body,
                encoding="utf-8",
            )
            (protocols_root / "iterative_repair.yaml").write_text(
                "protocol_id: iterative_repair\n" + protocol_body,
                encoding="utf-8",
            )

            fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
            result = run_suite_module.run_suite(
                tasks_root=fixtures_root / "tasks",
                protocols_root=protocols_root,
                prompts_root=prompts_root,
                model_registry_path=fixtures_root / "models" / "model_registry.yaml",
                protocol_id="direct_plan",
                adapter_factory=adapter_factory,
                validator_factory=mock_validator_module.build_mock_validator_for_suite,
            )

        self.assertEqual(result["protocol_ids"], ["direct_plan"])
        self.assertEqual(result["summary"]["num_protocols"], 1)
        self.assertEqual(result["summary"]["num_jobs"], 1)
        self.assertEqual({job["protocol_id"] for job in result["run_matrix"]}, {"direct_plan"})

    def test_run_suite_filters_task_family_tier_and_instance(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_task_filter_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_suite_task_filter_adapter_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_suite_task_filter_validator_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        def adapter_factory(model_entry, protocol_config):
            return mock_adapter_module.MockAdapter(
                scripted_outputs=["(move a b)"],
                model_id=str(model_entry.get("model_id", "mock_model")),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            tasks_root = tmp_root / "tasks"
            for task_family in ("toy", "other"):
                for tier in ("easy", "hard"):
                    tier_root = tasks_root / task_family / tier
                    tier_root.mkdir(parents=True)
                    domain_root = tasks_root / task_family / "domain"
                    domain_root.mkdir(exist_ok=True)
                    (domain_root / "domain.pddl").write_text("(define (domain toy))", encoding="utf-8")
                    (tier_root / "instance-01.pddl").write_text("(define (problem p1))", encoding="utf-8")

            fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
            result = run_suite_module.run_suite(
                tasks_root=tasks_root,
                protocols_root=fixtures_root / "protocols",
                model_registry_path=fixtures_root / "models" / "model_registry.yaml",
                task_family="toy",
                tier="hard",
                instance_id="instance-01",
                adapter_factory=adapter_factory,
                validator_factory=mock_validator_module.build_mock_validator_for_suite,
            )

        self.assertEqual(result["summary"]["num_task_cases"], 1)
        self.assertEqual(result["summary"]["num_jobs"], 1)
        self.assertEqual(result["task_cases"][0]["task_family"], "toy")
        self.assertEqual(result["task_cases"][0]["tier"], "hard")
        self.assertEqual(result["task_cases"][0]["instance_id"], "instance-01")

    def test_run_suite_can_use_real_validator(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        validate_command = _resolve_validate_command(framework_root)
        if validate_command is None:
            self.skipTest("Validate executable not found for real-validator suite integration test.")

        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_real_validator_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_suite_real_validator_adapter_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )

        def adapter_factory(model_entry, protocol_config):
            return mock_adapter_module.MockAdapter(
                scripted_outputs=["(move)"],
                model_id=str(model_entry.get("model_id", "mock_model")),
            )

        fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
        result = run_suite_module.run_suite(
            tasks_root=fixtures_root / "tasks",
            protocols_root=fixtures_root / "protocols",
            model_registry_path=fixtures_root / "models" / "model_registry.yaml",
            adapter_factory=adapter_factory,
            use_real_validator=True,
            validator_command=validate_command,
        )

        self.assertEqual(result["summary"]["num_jobs"], 1)
        self.assertEqual(len(result["suite_results"]), 1)
        self.assertEqual(result["aggregate_results"]["overall"]["num_solved"], 1)
        self.assertTrue(result["suite_results"][0]["solved"])
        self.assertTrue(result["suite_results"][0]["metrics"]["validity_at_1"])
        self.assertIn("scored_output_path", result["suite_results"][0])
        self.assertEqual(result["orchestration_errors"], [])

    def test_run_suite_preflight_uses_real_validator_before_jobs(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        validate_command = _resolve_validate_command(framework_root)
        if validate_command is None:
            self.skipTest("Validate executable not found for preflight integration test.")

        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_preflight_real_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_suite_preflight_adapter_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )

        def adapter_factory(model_entry, protocol_config):
            return mock_adapter_module.MockAdapter(
                scripted_outputs=["(move)"],
                model_id=str(model_entry.get("model_id", "mock_model")),
            )

        fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
        result = run_suite_module.run_suite(
            tasks_root=fixtures_root / "tasks",
            protocols_root=fixtures_root / "protocols",
            model_registry_path=fixtures_root / "models" / "model_registry.yaml",
            adapter_factory=adapter_factory,
            use_real_validator=True,
            validator_command=validate_command,
            preflight_tasks=True,
        )

        self.assertEqual(len(result["preflight_results"]), 1)
        self.assertTrue(result["preflight_results"][0]["ok"])
        self.assertEqual(result["preflight_results"][0]["status"], "ok")
        self.assertEqual(len(result["suite_results"]), 1)
        self.assertEqual(result["orchestration_errors"], [])

    def test_run_suite_preflight_failure_stops_before_model_jobs(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_preflight_unavailable_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_suite_preflight_unavailable_validator_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        adapter_calls = []

        def adapter_factory(model_entry, protocol_config):
            adapter_calls.append(model_entry)
            raise AssertionError("Adapter should not be built after preflight failure.")

        fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
        result = run_suite_module.run_suite(
            tasks_root=fixtures_root / "tasks",
            protocols_root=fixtures_root / "protocols",
            model_registry_path=fixtures_root / "models" / "model_registry.yaml",
            adapter_factory=adapter_factory,
            validator_factory=mock_validator_module.build_mock_validator_for_suite,
            validator_command="definitely-missing-validate-command",
            preflight_tasks=True,
        )

        self.assertEqual(adapter_calls, [])
        self.assertEqual(len(result["preflight_results"]), 1)
        self.assertFalse(result["preflight_results"][0]["ok"])
        self.assertEqual(result["preflight_results"][0]["status"], "validator_unavailable")
        self.assertEqual(result["suite_results"], [])
        self.assertEqual(result["orchestration_errors"][0]["error_type"], "validator_unavailable")

    def test_adapter_override_applies_to_enabled_models(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_adapter_override_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_suite_adapter_override_adapter_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_suite_adapter_override_validator_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        captured_entries = []

        def adapter_factory(model_entry, protocol_config):
            captured_entries.append(dict(model_entry))
            return mock_adapter_module.MockAdapter(
                scripted_outputs=["(move a b)"],
                model_id=str(model_entry.get("model_id", "mock_model")),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            registry_path = tmp_root / "model_registry.yaml"
            registry_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - model_id: model_a",
                        "    adapter: hf_local",
                        "    provider: huggingface_local",
                        "    enabled: true",
                        "  - model_id: model_b",
                        "    adapter: ollama",
                        "    provider: ollama_local",
                        "    enabled: true",
                        "  - model_id: model_c",
                        "    adapter: hf_local",
                        "    provider: huggingface_local",
                        "    enabled: false",
                    ]
                ),
                encoding="utf-8",
            )
            fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
            result = run_suite_module.run_suite(
                tasks_root=fixtures_root / "tasks",
                protocols_root=fixtures_root / "protocols",
                model_registry_path=registry_path,
                adapter_override="nvidia_api",
                adapter_factory=adapter_factory,
                validator_factory=mock_validator_module.build_mock_validator_for_suite,
            )

        self.assertEqual(result["model_ids"], ["model_a", "model_b"])
        self.assertEqual([entry["adapter"] for entry in captured_entries], ["nvidia_api", "nvidia_api"])

    def test_model_id_selection_uses_yaml_adapter_even_with_override(self) -> None:
        framework_root = Path(__file__).resolve().parents[1]
        run_suite_module = _load_module(
            "benchmark_framework_test_run_suite_model_id_module",
            framework_root / "runner" / "run_suite.py",
        )
        mock_adapter_module = _load_module(
            "benchmark_framework_test_suite_model_id_adapter_module",
            framework_root / "tests" / "mocks" / "mock_adapter.py",
        )
        mock_validator_module = _load_module(
            "benchmark_framework_test_suite_model_id_validator_module",
            framework_root / "tests" / "mocks" / "mock_validator.py",
        )

        captured_entries = []

        def adapter_factory(model_entry, protocol_config):
            captured_entries.append(dict(model_entry))
            return mock_adapter_module.MockAdapter(
                scripted_outputs=["(move a b)"],
                model_id=str(model_entry.get("model_id", "mock_model")),
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            registry_path = tmp_root / "model_registry.yaml"
            registry_path.write_text(
                "\n".join(
                    [
                        "models:",
                        "  - model_id: model_a",
                        "    adapter: hf_local",
                        "    provider: huggingface_local",
                        "    enabled: true",
                        "  - model_id: model_b",
                        "    adapter: llama_cpp_cli",
                        "    provider: llama_cpp_local",
                        "    enabled: false",
                    ]
                ),
                encoding="utf-8",
            )
            fixtures_root = framework_root / "tests" / "fixtures" / "benchmark_suite"
            result = run_suite_module.run_suite(
                tasks_root=fixtures_root / "tasks",
                protocols_root=fixtures_root / "protocols",
                model_registry_path=registry_path,
                model_id="model_b",
                adapter_override="nvidia_api",
                adapter_factory=adapter_factory,
                validator_factory=mock_validator_module.build_mock_validator_for_suite,
            )

        self.assertEqual(result["model_ids"], ["model_b"])
        self.assertEqual(captured_entries[0]["adapter"], "llama_cpp_cli")


if __name__ == "__main__":
    unittest.main()
