"""High-level orchestrator tying together file discovery, models, and processing."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .file_manager import DomainBundle, FileManager
from .model_manager import ModelManager
from .pddl_processor import PDDLProcessor
from utils.logging_utils import get_logger

logger = get_logger(__name__)


class PDDLPlanner:
    """Coordinates discovery, model loading, and processing workflows."""

    def __init__(self, args, config: Optional[Dict] = None):
        self.args = args
        self.config = config or {}
        self.file_manager: Optional[FileManager] = None
        self.model_manager: Optional[ModelManager] = None
        self.processor: Optional[PDDLProcessor] = None
        self.domains_data: List[DomainBundle] = []
        self.results: Dict[str, Any] = {}
        logger.info(
            "Planner initialized (problems=%s, weights=%s, output=%s)",
            args.problems_path,
            args.weights_path,
            args.output_dir,
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def setup(self) -> None:
        logger.info("Setting up planner components")
        self.file_manager = FileManager()
        self.domains_data = self._discover_domains()
        self._filter_domains_if_requested()

        model_path = self._resolve_model_path()
        self.model_manager = ModelManager(model_path)
        self.model_manager.load()

        self.processor = PDDLProcessor(
            model_manager=self.model_manager, output_dir=self.args.output_dir
        )
        logger.info("Planner setup complete")

    def run(self) -> None:
        if not self.domains_data or not self.processor:
            raise RuntimeError("Planner setup() must be executed before run().")

        processing_kwargs = self._build_processing_kwargs()

        if self.args.batch:
            logger.info("Running batch mode across all domains")
            self.results = self.processor.batch_process_domains(
                self.domains_data, **processing_kwargs
            )
        else:
            logger.info("Running sequential mode")
            overall_stats = {
                "total_problems": 0,
                "total_successful": 0,
                "total_failed": 0,
            }
            domain_results = []
            for domain_bundle in self.domains_data:
                result = self.processor.process_domain_with_validation(
                    domain_bundle, **processing_kwargs
                )
                domain_results.append(result)
                overall_stats["total_problems"] += result["total_problems"]
                overall_stats["total_successful"] += result["successful_plans"]
                overall_stats["total_failed"] += result["failed_plans"]

            self.results = {
                "domain_results": domain_results,
                "overall_stats": overall_stats,
            }

        self._log_summary()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _discover_domains(self) -> List[DomainBundle]:
        bundles = self.file_manager.find_pddl_files(self.args.problems_path)
        if not bundles:
            raise ValueError(
                f"No PDDL domains found in {self.args.problems_path}."
            )
        return bundles

    def _filter_domains_if_requested(self) -> None:
        if not self.args.domain:
            return
        filtered = [
            bundle
            for bundle in self.domains_data
            if bundle.domain_name.lower() == self.args.domain.lower()
        ]
        if not filtered:
            raise ValueError(f"Domain '{self.args.domain}' not found")
        self.domains_data = filtered

    def _resolve_model_path(self) -> str:
        return str(Path(self.args.weights_path))

    def _build_processing_kwargs(self) -> Dict[str, Any]:
        kwargs = {
            "max_iterations": self.args.max_iterations,
            "enable_cot": self.args.cot,
            "add_system_prompt": self.args.add_system_prompt,
            "sampling": self.args.sampling,
            "max_tokens": self.args.max_tokens,
            "include_prompt": self.args.include_prompt,
            "skip_special_tokens": self.args.skip_special_tokens,
            "temperature": self.args.temperature,
            "top_k": getattr(self.args, "top_k", 10),
        }
        return kwargs

    def _log_summary(self) -> None:
        stats = self.results.get("overall_stats")
        if not stats:
            return
        total = stats.get("total_problems", 0)
        success = stats.get("total_successful", 0)
        failed = stats.get("total_failed", 0)
        rate = (success / total) * 100 if total else 0
        logger.info(
            "Summary: problems=%d success=%d failed=%d (%.1f%%)",
            total,
            success,
            failed,
            rate,
        )
        for domain in self.results.get("domain_results", []):
            if "error" in domain:
                logger.error("Domain %s failed: %s", domain["domain_name"], domain["error"])
            else:
                domain_total = domain["total_problems"]
                domain_success = domain["successful_plans"]
                domain_rate = (
                    (domain_success / domain_total) * 100 if domain_total else 0
                )
                logger.info(
                    "Domain %s: %d/%d (%.1f%%)",
                    domain["domain_name"],
                    domain_success,
                    domain_total,
                    domain_rate,
                )
        logger.info("Results stored in %s", self.args.output_dir)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_results(self) -> Dict[str, Any]:
        return self.results

    def get_planner_info(self) -> Dict[str, Any]:
        return {
            "arguments": vars(self.args),
            "config": self.config,
            "domains_available": len(self.domains_data),
            "model_info": self.model_manager.get_model_info()
            if self.model_manager
            else None,
            "processor_info": self.processor.get_processor_info()
            if self.processor
            else None,
        }


__all__ = ["PDDLPlanner"]