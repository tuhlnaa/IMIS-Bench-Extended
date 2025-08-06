"""
Configuration management for PyTorch training using OmegaConf.
Based on: https://github.com/huggingface/pytorch-image-models/blob/main/train.py
"""
import sys
import argparse

from pathlib import Path
from typing import Dict, Any, Optional, Union
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.utils.logging_utils import LoggingManager

def parse_args(args=None) -> OmegaConf:
    """Parse command line arguments and merge with config file.
    
    Args:
        args: List of arguments to parse. If None, uses sys.argv.
    """
    parser = argparse.ArgumentParser(description="Train RSNA Trauma Detection model")
   
    # Only keep frequently modified arguments in argparse
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--output_dir", type=str, help="Path to save model and results")
    parser.add_argument("--data_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--csv_path", type=str, help="Path to the CSV file containing class mapping")
    parser.add_argument("--batch_size", type=int, help="Input batch size for training")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train")
    parser.add_argument("--learning_rate", type=float, help="Learning rate for the optimizer")
    parser.add_argument("--device", type=str, help="Device (accelerator) to use")
    parser.add_argument("--experiment_name", type=str, help="Name for the experiment")
    parser.add_argument("--seed", type=int, help="Random seed")
   
    parsed_args = parser.parse_args(args)
   
    # Create the configuration object
    config = create_config(parsed_args)

    # Print configuration using the logging utility
    LoggingManager.print_config(config, "Configuration")

    return config


def create_config(args: argparse.Namespace) -> OmegaConf:
    """Create a unified configuration by merging default, file and CLI configs."""
    # 1. Load default configuration
    default_conf = OmegaConf.load(Path(__file__).parent / "default_config.yaml")
    
    # 2. Load config from file if specified
    file_conf = OmegaConf.create({})
    if args.config:
        config_path = Path(args.config)
        if config_path.exists():
            file_conf = OmegaConf.load(config_path)
        else:
            print(f"Warning: Config file {args.config} not found. Using defaults.")
    
    # 3. Create config from command line arguments (only non-None values)
    cli_conf = OmegaConf.create({k: v for k, v in vars(args).items() if v is not None})
    
    # 4. Merge configurations with priority: CLI > file > default
    config = OmegaConf.merge(default_conf, file_conf, cli_conf)
    
    # 5. Validate and process paths
    process_paths(config)
    
    return config


def process_paths(config: OmegaConf) -> None:
    """Process and validate paths in the configuration."""
    # Convert string paths to Path objects
    path_keys = ["output_dir", "data_dir", "mask_dir", "category_weights_path", "checkpoint_path"]
    
    for key in path_keys:
        if key in config and config[key]:
            # Convert to Path object
            config[key] = Path(config[key])
            
            # Validate existence for input paths
            if key != "output_dir" and not config[key].exists():
                print(f"Warning: {key} at {config[key]} does not exist.")
    
    # Create output directory if it doesn"t exist
    if "output_dir" in config and config["output_dir"]:
        config.output_dir.mkdir(parents=True, exist_ok=True)


def save_config(config: OmegaConf, path: Union[str, Path]) -> None:
    """Save configuration to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to container before saving to ensure proper serialization
    config_dict = OmegaConf.to_container(config, resolve=True)
    
    with open(path, "w") as f:
        OmegaConf.save(config=config_dict, f=f)

    
if __name__ == "__main__":
    parse_args()