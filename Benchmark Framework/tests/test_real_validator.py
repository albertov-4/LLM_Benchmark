"""Integration test for the real VAL-based validator adapter."""

import importlib.util
import shutil
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
    """Resolve the real VAL executable from PATH or common local build paths."""
    path_command = shutil.which("Validate")
    if path_command:
        return path_command

    workspace_root = framework_root.parent
    candidate_paths = [
        workspace_root / "VAL" / "build" / "win64" / "mingw" / "Debug" / "bin" / "Validate.exe",
        workspace_root / "VAL" / "build" / "win64" / "mingw" / "Debug" / "install" / "bin" / "Validate.exe",
        workspace_root / "VAL" / "build" / "win64" / "mingw" / "Release" / "bin" / "Validate.exe",
        workspace_root.parent / "VAL" / "build" / "win64" / "mingw" / "Debug" / "bin" / "Validate.exe",
        workspace_root.parent / "VAL" / "build" / "win64" / "mingw" / "Debug" / "install" / "bin" / "Validate.exe",
        workspace_root.parent / "VAL" / "build" / "win64" / "mingw" / "Release" / "bin" / "Validate.exe",
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return str(candidate)

    return None


class RealValidatorIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.framework_root = Path(__file__).resolve().parents[1]
        cls.validator_module = _load_module(
            "benchmark_framework_test_real_validator_module",
            cls.framework_root / "evaluators" / "validator.py",
        )
        cls.validate_command = _resolve_validate_command(cls.framework_root)
        if cls.validate_command is None:
            raise unittest.SkipTest("Validate executable not found in PATH or local VAL build folders.")

        cls.domain_file = cls.framework_root / "tests" / "fixtures" / "benchmark_suite" / "tasks" / "toy" / "domain" / "domain.pddl"
        cls.problem_file = cls.framework_root / "tests" / "fixtures" / "benchmark_suite" / "tasks" / "toy" / "easy" / "instance-01.pddl"

    def _build_validator(self):
        config = self.validator_module.VALValidatorConfig(
            validator_command=self.validate_command,
            timeout_seconds=10,
        )
        return self.validator_module.VALValidatorAdapter(config)

    def test_real_validator_accepts_valid_toy_plan(self) -> None:
        validator = self._build_validator()
        result = validator.validate(
            domain_file=str(self.domain_file),
            problem_file=str(self.problem_file),
            plan_text="(move)",
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.status, "valid")
        self.assertIsNone(result.error_type)
        self.assertEqual(result.plan_length, 1)
        self.assertIsNotNone(result.raw_validator_output)
        self.assertIn("Plan valid", result.raw_validator_output)

    def test_real_validator_rejects_invalid_toy_plan(self) -> None:
        validator = self._build_validator()
        result = validator.validate(
            domain_file=str(self.domain_file),
            problem_file=str(self.problem_file),
            plan_text="(wrong-action)",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "invalid")
        self.assertIn(result.error_type, {"unknown", "unknown_action", "syntax_error"})
        self.assertEqual(result.plan_length, 1)
        self.assertIsNotNone(result.raw_validator_output)
        self.assertIn("operator", result.raw_validator_output.lower())

    def test_real_validator_accepts_citycar_starter_plan(self) -> None:
        validator = self._build_validator()
        domain_file = self.framework_root / "tasks" / "citycar" / "domain" / "domain.pddl"
        problem_file = self.framework_root / "tasks" / "citycar" / "easy" / "instance-01.pddl"
        result = validator.validate(
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            plan_text="(move car1 j1 j2)\n(move car1 j2 j3)",
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.status, "valid")
        self.assertIsNone(result.error_type)
        self.assertEqual(result.plan_length, 2)

    def test_real_validator_accepts_tetris_starter_plan(self) -> None:
        validator = self._build_validator()
        domain_file = self.framework_root / "tasks" / "tetris" / "domain" / "domain.pddl"
        problem_file = self.framework_root / "tasks" / "tetris" / "medium" / "instance-01.pddl"
        result = validator.validate(
            domain_file=str(domain_file),
            problem_file=str(problem_file),
            plan_text="(slide piece2 c3 c4)\n(slide piece1 c1 c2)\n(slide piece1 c2 c3)\n(slide piece2 c4 c5)",
        )

        self.assertTrue(result.valid)
        self.assertEqual(result.status, "valid")
        self.assertIsNone(result.error_type)
        self.assertEqual(result.plan_length, 4)


if __name__ == "__main__":
    unittest.main()
