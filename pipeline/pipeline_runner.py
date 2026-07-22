# pipeline/pipeline_runner.py
"""Runner script for data downloading and preprocessing pipeline."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.run_pipeline import run_data_pipeline
from src.utils.configs import load_and_resolve_config


def main():
    parser = argparse.ArgumentParser(
        description="Run the Bird Intelligence System data downloading and preprocessing pipeline."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to config file relative to project root (default: configs/config.yaml)",
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Process the entire dataset without class balancing/filtering",
    )

    args = parser.parse_args()

    # Load and resolve paths
    resolved_config = load_and_resolve_config(ROOT_DIR, args.config)

    # Execute data processing pipeline
    run_data_pipeline(resolved_config, use_full_dataset=args.full_dataset)


if __name__ == "__main__":
    main()
