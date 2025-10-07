"""Main entry point for training and evaluating the NanoSocrates model.

This script orchestrates the entire machine learning pipeline, including:
- Setting up logging and experiment tracking (wandb).
- Ensuring reproducibility by setting random seeds.
- Loading datasets and the tokenizer.
- Building the model, optimizer, and learning rate scheduler.
- Initiating the training loop.

Configuration is managed via YAML files, with the option to override any
parameter directly from the command line for maximum flexibility.

Example usage:
    # Run training with a specific config file
    python src/run.py --config-path configs/train/baseline.yaml

    # Override a parameter from the command line
    python src/run.py --config-path configs/train/baseline.yaml --training.batch_size 16
"""

import argparse
import logging
import random
import string
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import wandb
import yaml
from tokenizers import Tokenizer

from .data.builders import build_and_cache_datasets
from .model.transformer import TinySeq2Seq
from .training.dataloaders import create_multitask_dataloader
from .training.loop import TrainingLoop
from .training.scheduler import create_scheduler

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility across all relevant libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def setup_wandb(config: Dict[str, Any]) -> "wandb.sdk.wandb_run.Run":
    """Initializes and configures a Weights & Biases run."""
    run_id = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"].get("entity"),
        config=config,
        name=f"{config['wandb']['name']}-{run_id}",
        notes=config["wandb"].get("notes"),
    )
    logger.info("Weights & Biases run initialized.")
    return run


def load_essentials(config: Dict[str, Any]) -> tuple[Dict[str, Any], Tokenizer]:
    """Loads the datasets and tokenizer."""
    logger.info("Loading tokenizer...")
    tokenizer = Tokenizer.from_file(config["data"]["tokenizer_path"])

    logger.info("Building or loading datasets from cache...")
    datasets = build_and_cache_datasets(config["data"], tokenizer)
    logger.info(f"Datasets loaded. Splits: {list(datasets.keys())}")
    
    return datasets, tokenizer


def build_components(
    config: Dict[str, Any], model_vocab_size: int
) -> tuple[TinySeq2Seq, torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """Builds the model, optimizer, and scheduler from the configuration."""
    logger.info(f"Building model with architecture: {config['model']['architecture']}...")
    model = TinySeq2Seq(vocab_size=model_vocab_size, **config["model"])
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters.")

    logger.info("Creating optimizer...")
    optimizer = torch.optim.AdamW(model.parameters(), **config["optimizer"])

    logger.info("Creating learning rate scheduler...")
    # Calculate total steps for scheduler
    num_train_samples = len(config["data"]["train_path"]) # Approximation
    steps_per_epoch = num_train_samples // config["training"]["batch_size"]
    total_steps = steps_per_epoch * config["training"]["num_epochs"]
    
    scheduler = create_scheduler(
        optimizer=optimizer, total_steps=total_steps, **config["scheduler"]
    )
    return model, optimizer, scheduler


def main(config: Dict[str, Any]) -> None:
    """The main function to run the entire pipeline."""
    set_seed(config["training"]["seed"])

    run = setup_wandb(config) if config.get("wandb") else None
    
    datasets, tokenizer = load_essentials(config)

    model, optimizer, scheduler = build_components(config, tokenizer.get_vocab_size())
    
    logger.info("Creating dataloaders...")
    train_loader = create_multitask_dataloader(
        datasets["train"],
        tokenizer=tokenizer,
        **config["data"]["train_loader"],
    )
    val_loader = create_multitask_dataloader(
        datasets["validation"],
        tokenizer=tokenizer,
        shuffle=False, # No need to shuffle validation data
        **config["data"]["val_loader"],
    )

    logger.info("Initializing training loop...")
    training_loop = TrainingLoop(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        wandb_run=run,
        **config["training_loop"],
    )

    training_loop.run(num_epochs=config["training"]["num_epochs"])

    if run:
        run.finish()
    logger.info("Pipeline finished successfully.")


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments for configuration."""
    parser = argparse.ArgumentParser(description="Run NanoSocrates Training")
    parser.add_argument(
        "--config-path", type=Path, required=True, help="Path to the YAML config file."
    )
    # Allows overriding config values, e.g., --training.batch_size 32
    parser.add_argument(
        "overrides",
        nargs="*",
        help="<key>=<value> pairs to override config values.",
    )
    return parser.parse_args()


def load_config_with_overrides(args: argparse.Namespace) -> Dict[str, Any]:
    """Loads a YAML config and applies command-line overrides."""
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    for override in args.overrides:
        key_str, value = override.split("=", 1)
        keys = key_str.split(".")
        sub_config = config
        for key in keys[:-1]:
            sub_config = sub_config.setdefault(key, {})
        
        # Attempt to cast value to its correct type
        try:
            value = eval(value)
        except (NameError, SyntaxError):
            pass # Keep as string if it's not a basic type
        sub_config[keys[-1]] = value
        
    return config


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    args = parse_args()
    config = load_config_with_overrides(args)
    main(config)