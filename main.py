# main.py

import argparse
from pathlib import Path

from src.data.run_pipeline import run_data_pipeline
from src.training.train import train_model
from src.evaluation.evaluate import evaluate_model
from src.utils.configs import load_and_resolve_config  # use the centralized config loader

# Determine the absolute path of the project root (where main.py is located) and define the absolute path to the config file
ROOT_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = "configs/config.yaml"

def main():
    parser = argparse.ArgumentParser(description="Bird Intelligence System Orchestrator")
    parser.add_argument("--config", type=str, default=CONFIG_FILE_PATH, help="Path to the config file")
    parser.add_argument("--pipeline", action="store_true", help="Run the data downloading and preprocessing pipeline")
    parser.add_argument("--train", action="store_true", help="Run the end-to-end training script")
    parser.add_argument("--evaluate", action="store_true", help="Run the evaluation and analysis script")
    parser.add_argument("--all", action="store_true", help="Run the entire workflow sequentially")

    args = parser.parse_args()

    # Load and resolve all paths using the centralized loader
    config = load_and_resolve_config(ROOT_DIR, args.config)

    # 1. Data Pipeline
    if args.pipeline or args.all:
        print(">>> Starting Data Pipeline...")
        run_data_pipeline(config)

    # 2. Training Loop
    if args.train or args.all:
        print(">>> Starting Model Training...")
        train_model(config)

    # 3. Evaluation
    if args.evaluate or args.all:
        print(">>> Starting Evaluation and Analysis...")
        evaluate_model(config)

    if not (args.pipeline or args.train or args.evaluate or args.all):
        print("No action specified. Use --help to see available arguments.")

if __name__ == "__main__":
    main()
