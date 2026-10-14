"""Small command-line interface for local CellState analyses."""

from __future__ import annotations

import argparse
import logging

from .config import load_config
from .pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cellstate", description="Cross-dataset single-cell RNA-seq analysis"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run analysis on one or more local AnnData/10x datasets")
    run.add_argument("--config", required=True, help="YAML configuration path")
    run.add_argument(
        "--input",
        required=True,
        action="append",
        help="input dataset path; repeat for multiple datasets",
    )
    run.add_argument(
        "--combat", action="store_true", help="also calculate optional ComBat representation"
    )
    run.add_argument(
        "--sensitivity",
        action="store_true",
        help="run sensitivity grid specified in config/CLI defaults",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    config = load_config(args.config)
    result = run_pipeline(
        args.input, config, run_combat=args.combat, run_sensitivity=args.sensitivity
    )
    print(f"CellState {result['status']}; outputs: {config.output_dir}")


if __name__ == "__main__":
    main()
