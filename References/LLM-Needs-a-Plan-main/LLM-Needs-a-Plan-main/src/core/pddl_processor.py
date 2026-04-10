"""Processing pipeline orchestrating prompt creation, generation, and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .file_manager import DomainBundle, FileManager
from .model_manager import ModelManager
from prompts.prompts import (
    add_constraints_to_prompt,
    add_examples_to_prompt,
    chain_of_thought_prompt,
    citycar_problem_prompt,
    citycar_validation_feedback,
    generic_pddl_prompt,
    tetris_problem_prompt,
    tetris_validation_feedback,
)
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PDDLProcessor:
    """Coordinates domain processing, prompt creation, and plan validation."""

    def __init__(self, model_manager: ModelManager, output_dir: str):
        self.model_manager = model_manager
        self.output_dir = Path(output_dir)
        self.file_manager = FileManager()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("PDDLProcessor initialized (output=%s)", self.output_dir)

    # ------------------------------------------------------------------
    # Domain-level processing
    # ------------------------------------------------------------------

    def process_domain_with_validation(
        self,
        domain_data: DomainBundle,
        *,
        max_iterations: int = 3,
        enable_cot: bool = False,
        add_system_prompt: bool = True,
        sampling: bool = False,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        domain_output_dir = self.output_dir / domain_data.domain_name
        domain_output_dir.mkdir(exist_ok=True)
        logger.info(
            "Processing domain %s (%d problems)",
            domain_data.domain_name,
            len(domain_data.problem_paths),
        )

        results = {
            "domain_name": domain_data.domain_name,
            "total_problems": len(domain_data.problem_paths),
            "successful_plans": 0,
            "failed_plans": 0,
            "problem_results": [],
            "output_directory": str(domain_output_dir),
        }

        for idx, problem_path in enumerate(domain_data.problem_paths, start=1):
            logger.info(
                "[%d/%d] %s",
                idx,
                len(domain_data.problem_paths),
                problem_path.name,
            )
            try:
                problem_result = self._process_single_problem(
                    domain=domain_data,
                    problem_path=problem_path,
                    output_dir=domain_output_dir,
                    max_iterations=max_iterations,
                    enable_cot=enable_cot,
                    add_system_prompt=add_system_prompt,
                    sampling=sampling,
                    **generation_kwargs,
                )
                results["problem_results"].append(problem_result)
                if problem_result["plan_valid"]:
                    results["successful_plans"] += 1
                else:
                    results["failed_plans"] += 1
            except Exception as exc:
                logger.exception(
                    "Problem %s failed with error", problem_path.name
                )
                results["failed_plans"] += 1
                results["problem_results"].append(
                    {
                        "problem_path": str(problem_path),
                        "problem_name": problem_path.stem,
                        "plan_valid": False,
                        "iterations": 0,
                        "error": str(exc),
                    }
                )

        success_rate = (
            (results["successful_plans"] / results["total_problems"]) * 100
            if results["total_problems"]
            else 0
        )
        logger.info(
            "Domain %s done: %d/%d valid (%.1f%%)",
            domain_data.domain_name,
            results["successful_plans"],
            results["total_problems"],
            success_rate,
        )

        return results

    def batch_process_domains(
        self,
        domains_data: List[DomainBundle],
        *,
        max_iterations: int = 3,
        enable_cot: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        logger.info("Starting batch processing of %d domain(s)", len(domains_data))
        batch_results = {
            "total_domains": len(domains_data),
            "domain_results": [],
            "overall_stats": {
                "total_problems": 0,
                "total_successful": 0,
                "total_failed": 0,
            },
        }

        for domain_bundle in domains_data:
            result = self.process_domain_with_validation(
                domain_bundle,
                max_iterations=max_iterations,
                enable_cot=enable_cot,
                **kwargs,
            )
            batch_results["domain_results"].append(result)
            stats = batch_results["overall_stats"]
            stats["total_problems"] += result["total_problems"]
            stats["total_successful"] += result["successful_plans"]
            stats["total_failed"] += result["failed_plans"]

        stats = batch_results["overall_stats"]
        success_rate = (
            (stats["total_successful"] / stats["total_problems"]) * 100
            if stats["total_problems"]
            else 0
        )
        logger.info(
            "Batch complete: %d problems, %.1f%% success",
            stats["total_problems"],
            success_rate,
        )
        return batch_results

    # ------------------------------------------------------------------
    # Problem-level processing
    # ------------------------------------------------------------------

    def _process_single_problem(
        self,
        *,
        domain: DomainBundle,
        problem_path: Path,
        output_dir: Path,
        max_iterations: int,
        enable_cot: bool,
        add_system_prompt: bool,
        sampling: bool,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        problem_text = self.file_manager.read_file(problem_path)
        if problem_text is None:
            raise ValueError(f"Unable to read problem file {problem_path}")

        prompt = self._build_prompt(
            domain_name=domain.domain_name,
            domain_text=domain.domain_text,
            problem_text=problem_text,
            enable_cot=enable_cot,
        )

        validation_feedback_fn = self._get_validation_feedback_fn(domain.domain_name)

        response_text, iterations, is_valid = self.model_manager.iterative_planning_with_validation(
            domain_path=str(domain.domain_path),
            problem_path=str(problem_path),
            initial_prompt=prompt,
            max_iterations=max_iterations,
            add_system_prompt=add_system_prompt,
            validation_feedback_fn=validation_feedback_fn,
            sampling=sampling,
            **generation_kwargs,
        )

        plan_path = output_dir / f"{problem_path.stem}_plan.txt"
        metadata = (
            "\n\n--- Processing Metadata ---\n"
            f"Domain: {domain.domain_name}\n"
            f"Problem: {problem_path.stem}\n"
            f"Iterations: {iterations}\n"
            f"Plan Valid: {is_valid}\n"
            f"Chain of Thought: {enable_cot}\n"
        )
        self.file_manager.save_file(plan_path, response_text + metadata)

        return {
            "problem_path": str(problem_path),
            "problem_name": problem_path.stem,
            "plan_path": str(plan_path),
            "plan_valid": is_valid,
            "iterations": iterations,
            "response_length": len(response_text),
            "cot_enabled": enable_cot,
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        *,
        domain_name: str,
        domain_text: str,
        problem_text: str,
        enable_cot: bool,
        examples: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
    ) -> str:
        top_instruction = (
            "OUTPUT ONLY: Provide the final PDDL action sequence, one action per line. "
            "Do NOT ask clarifying questions; rely solely on the DOMAIN and PROBLEM provided."
        )

        base_prompt = self._create_domain_prompt(
            domain_name, domain_text, problem_text, include_examples=False
        )
        composed = f"{top_instruction}\n\n{base_prompt}"

        if not examples:
            examples = self._load_domain_examples(domain_name)
        if examples:
            try:
                composed = add_examples_to_prompt(composed, examples)
            except Exception:
                composed += "\n\n" + "\n\n".join(examples)

        if constraints:
            try:
                composed = add_constraints_to_prompt(composed, constraints)
            except Exception:
                composed += "\n\nAdditional constraints:\n" + "\n".join(constraints)

        if enable_cot:
            try:
                cot_text = self._chain_of_thought(domain_name, domain_text, problem_text)
                composed += "\n\n" + cot_text
            except Exception:
                composed += "\n\nThink step by step before finalizing the plan."

        return composed

    def _create_domain_prompt(
        self,
        domain_name: str,
        domain_text: str,
        problem_text: str,
        include_examples: bool,
    ) -> str:
        domain_lower = domain_name.lower()
        if "tetris" in domain_lower:
            logger.debug("Using Tetris prompt template")
            return tetris_problem_prompt(domain_text, problem_text, include_examples)
        if "citycar" in domain_lower:
            logger.debug("Using CityCar prompt template")
            return citycar_problem_prompt(domain_text, problem_text, include_examples)
        logger.debug("Using generic prompt template for %s", domain_name)
        return generic_pddl_prompt(domain_text, problem_text)

    def _chain_of_thought(self, domain_name: str, domain_text: str, problem_text: str) -> str:
        domain_lower = domain_name.lower()
        if "tetris" in domain_lower:
            from prompts.prompts import tetris_chain_of_thought

            return tetris_chain_of_thought(domain_text, problem_text)
        if "citycar" in domain_lower:
            from prompts.prompts import citycar_chain_of_thought

            return citycar_chain_of_thought(domain_text, problem_text)
        return chain_of_thought_prompt(domain_text, problem_text)

    def _load_domain_examples(self, domain_name: str) -> Optional[List[str]]:
        try:
            from prompts import prompts as prompts_module

            domain_lower = domain_name.lower()
            if "tetris" in domain_lower and hasattr(
                prompts_module, "_format_tetris_examples"
            ):
                return prompts_module._format_tetris_examples()
            if "citycar" in domain_lower and hasattr(
                prompts_module, "_format_citycar_examples"
            ):
                return prompts_module._format_citycar_examples()
        except Exception:
            logger.debug("Example loading failed for %s", domain_name)
        return None

    # ------------------------------------------------------------------
    # Validation feedback helpers
    # ------------------------------------------------------------------

    def _get_validation_feedback_fn(self, domain_name: str):
        domain_lower = domain_name.lower()
        if "tetris" in domain_lower:
            return tetris_validation_feedback
        if "citycar" in domain_lower:
            return citycar_validation_feedback
        return None

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_processor_info(self) -> Dict[str, Any]:
        return {
            "output_directory": str(self.output_dir),
            "model_info": self.model_manager.get_model_info(),
            "file_manager_available": self.file_manager is not None,
        }


__all__ = ["PDDLProcessor"]