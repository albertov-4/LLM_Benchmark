#!/usr/bin/env python3
"""CLI entry point for the PDDL Planning Framework."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

from core.pddl_planner import PDDLPlanner
from utils.configuration import load_config
from utils.logging_utils import configure_logging, get_logger


def _build_parser(config: Dict) -> argparse.ArgumentParser:
    """Create the CLI parser with defaults pulled from config.yml."""
    parser = argparse.ArgumentParser(
        description="PDDL Planning with Large Language Models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--problems_path",
        default=config.get("PROBLEMS_PATH", "src/data"),
        help="Directory containing PDDL domains and problems",
    )
    parser.add_argument(
        "--weights_path",
        default=config.get("MODEL_PATH", "src/models"),
        help="Directory with model weights",
    )
    parser.add_argument(
        "--output_dir",
        default=config.get("MODEL_OUTPUT", "src/results"),
        help="Destination directory for generated plans",
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process all discovered domains in batch mode",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Process only the specified domain name",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=config.get("DEFAULT_ITERATIONS", 1),
        help="Maximum validation iterations per problem",
    )

    parser.add_argument(
        "--sampling",
        action="store_true",
        help="Enable sampling (temperature / top-k) instead of greedy decoding",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=config.get("TEMPERATURE", 0.1),
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=config.get("TOP_K", 10),
        help="Top-k for sampling",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=config.get("MAX_TOKENS", 1024),
        help="Maximum tokens to generate per response",
    )

    boolean_action = argparse.BooleanOptionalAction
    parser.add_argument(
        "--add_system_prompt",
        action=boolean_action,
        default=True,
        help="Include the system prompt when talking to the model",
    )
    parser.add_argument(
        "--cot",
        action=boolean_action,
        default=True,
        help="Enable chain-of-thought prompting",
    )
    parser.add_argument(
        "--include_prompt",
        action=boolean_action,
        default=True,
        help="Persist the prompt alongside the model output",
    )
    parser.add_argument(
        "--skip_special_tokens",
        action=boolean_action,
        default=True,
        help="Strip tokenizer special tokens from the output",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable additional debug logging",
    )

    parser.add_argument(
        "--model",
        choices=["llama3", "phi4", "gemma3", "kimi", "auto"],
        default="auto",
        help="Explicit model type override",
    )
    parser.add_argument(
        "--log-level",
        default=config.get("LOG_LEVEL", "INFO"),
        help="Root logging level",
    )
    parser.add_argument(
        "--log-file",
        default=config.get("LOG_FILE"),
        help="Optional path to also write logs",
    )

    return parser


def main():
    """Main function to run the PDDL planning pipeline."""

    try:
        config = load_config()
    except Exception as exc:  # pragma: no cover - fatal configuration
        print(f"Failed to load config.yml: {exc}", file=sys.stderr)
        sys.exit(1)

    parser = _build_parser(config)
    args = parser.parse_args()

    configure_logging(level=args.log_level, log_file=args.log_file)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger = get_logger(__name__)
    logger.info("Starting PDDL Planning Framework")

    args.output_dir = _resolve_output_dir(args)
    _validate_paths(args.problems_path, args.weights_path, logger)

    logger.debug("CLI arguments: %s", vars(args))

    try:
        planner = PDDLPlanner(args, config)
        planner.setup()
        planner.run()
        logger.info("PDDL Planning Framework completed successfully")
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        sys.exit(1)
    except Exception:
        logger.exception("Fatal error during planning run")
        sys.exit(1)


def _validate_paths(problems_path: str, weights_path: str, logger: logging.Logger) -> None:
    """Ensure the provided directories exist before running the planner."""
    if not Path(problems_path).exists():
        logger.error("Problems path does not exist: %s", problems_path)
        sys.exit(1)
    if not Path(weights_path).exists():
        logger.error("Model weights path does not exist: %s", weights_path)
        sys.exit(1)


def _resolve_output_dir(args: argparse.Namespace) -> str:
    """Append the model alias to the output directory when available."""
    output_dir = Path(args.output_dir)
    model_alias = None

    if args.model and args.model.lower() != "auto":
        model_alias = args.model.lower()
    else:
        model_alias = Path(args.weights_path).name.lower()

    if model_alias:
        output_dir = output_dir / model_alias

    try:
        os.makedirs(output_dir, exist_ok=True)
    except OSError:
        pass

    return str(output_dir)


if __name__ == "__main__":
    main()