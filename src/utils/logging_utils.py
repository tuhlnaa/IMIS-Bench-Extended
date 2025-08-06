import torch
import logging

from pathlib import Path
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from typing import Any, Dict, List, Union, Optional
from rich import box, print
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from omegaconf import OmegaConf

from .logging_base import BaseLogger

# ClearML logger implementation
try:
    from clearml import Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False



class LoggingManager:
    """Manages multiple loggers and provides unified logging interface."""
    def __init__(
            self, 
            output_dir: str, 
            logger_type: str = 'tensorboard',
            neptune_config: Optional[Dict[str, str]] = None,
            clearml_config: Optional[Dict[str, str]] = None
        ):
        self.output_dir = Path(output_dir)
        self.loggers = []
        
        if logger_type == 'tensorboard':
            self.loggers.append(TensorBoardLogger(output_dir))

        elif logger_type == 'clearml':
            if not clearml_config:
                raise ValueError("ClearML configuration required when using ClearML loggers")
            self.loggers.append(ClearMLLogger(output_dir, **clearml_config))


    def log_metrics(self, *args, **kwargs) -> None:
        """Forward metrics to all active loggers."""
        for logger in self.loggers:
            logger.log_metrics(*args, **kwargs)


    def log_hyperparameters(self, *args, **kwargs) -> None:
        """Forward hyperparameters to all active loggers."""
        for logger in self.loggers:
            logger.log_hyperparameters(*args, **kwargs)


    def log_model_summary(self, *args, **kwargs) -> None:
        """Forward model summary to all active loggers."""
        for logger in self.loggers:
            logger.log_model_summary(*args, **kwargs)


    def log_artifact(self, *args, **kwargs) -> None:
        """Forward artifacts to all active loggers."""
        for logger in self.loggers:
            logger.log_artifact(*args, **kwargs)


    def close(self) -> None:
        """Close all loggers."""
        for logger in self.loggers:
            logger.close()


    @staticmethod
    def print_transforms(transform: Any, title: str = "Transform Steps") -> None:
        """Print transformation pipeline details."""
        transforms = transform.transforms if hasattr(transform, 'transforms') else [transform]
        pretty_transforms = Pretty(transforms)
        panel = Panel(pretty_transforms, title=title, subtitle="Detailed Transformations")
        print(panel, "\n")


    @staticmethod
    def print_config(config: Any, title: str = "Configuration") -> None:
        """Print configuration details in a structured table."""
        table = Table(title=title, box=box.ROUNDED)
        table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        # Check if the config is an OmegaConf object
        if OmegaConf.is_config(config):
            # Convert OmegaConf to a dictionary
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            # Add rows recursively for nested config
            def add_dict_to_table(d, prefix=""):
                for key, value in d.items():
                    param_name = f"{prefix}{key}"
                    if isinstance(value, dict):
                        add_dict_to_table(value, f"{param_name}.")
                    else:
                        pretty_value = Pretty(value, indent_guides=False)
                        table.add_row(param_name, pretty_value)
            
            add_dict_to_table(config_dict)
        else:
            # Handle argparse or other config types
            for key, value in vars(config).items():
                pretty_value = Pretty(value, indent_guides=False)
                table.add_row(key, pretty_value)
        
        print(table, "\n")


    @staticmethod
    def print_training_config(
            args: Any,
            train_loader: DataLoader, 
            val_loader: DataLoader,
            loss_functions: Union[Dict, List], 
            scheduler: Union[Dict, List], 
            title: str = "Training Configuration"
        ) -> None:
        """Print training configuration details in a structured table."""

        # Calculate training steps with validation frequency
        num_training_steps_per_epoch = len(train_loader)
        total_steps = num_training_steps_per_epoch * args.lr_scheduler.epochs

        table = Table(title=title, box=box.SIMPLE_HEAVY)
        table.add_column("Parameter", justify="right", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        params = {
            "Learning rate": args.lr_scheduler.learning_rate,
            "Epochs": args.lr_scheduler.epochs,
            "Batch size": args.model.batch_size,
            "Total steps (Iteration)": total_steps,
            "Steps per epoch": num_training_steps_per_epoch,
            "Training examples": len(train_loader.dataset),
            "Validation examples": len(val_loader.dataset),
            "Criterion": loss_functions,
            "Scheduler": scheduler
        }
        
        for param_name, param_value in params.items():
            table.add_row(param_name, Pretty(param_value, indent_guides=False))
        
        print(table, "\n")


    @staticmethod
    def print_validation_summary(
            epoch: int,
            num_epochs: int,
            metrics: Dict[str, float],
            is_best: bool
        ) -> None:
        """Print validation summary using Rich table.
        
        Args:
            epoch: Current epoch number
            num_epochs: Total number of epochs
            metrics: Combined training and validation metrics
            is_best: Whether this is the best model so far
        """
        table = Table(title=f"Epoch {epoch}/{num_epochs} Summary", box=box.ROUNDED)
        table.add_column("Metric", justify="right", style="cyan")
        table.add_column("Training", style="green")
        table.add_column("Validation", style="blue")

        # Separate training and validation metrics
        train_metrics = {k.replace('train_', ''): v for k, v in metrics.items() if k.startswith('train_')}
        val_metrics = {k.replace('val_', ''): v for k, v in metrics.items() if k.startswith('val_')}
        
        # Get all unique metric names without prefixes
        metric_names = set(train_metrics.keys()) | set(val_metrics.keys())
        
        # Add learning rate separately if present
        if 'lr' in metrics:
            table.add_row('Learning Rate', f"{metrics['learning_rate']:.6f}", "N/A")
            
        # Add other metrics
        for metric in sorted(metric_names):
            if metric != 'lr':  # Skip lr as it's already added
                train_value = f"{train_metrics.get(metric, 'N/A'):.4f}" if metric in train_metrics else "N/A"
                val_value = f"{val_metrics.get(metric, 'N/A'):.4f}" if metric in val_metrics else "N/A"
                table.add_row(metric, train_value, val_value)

        if is_best:
            table.caption = "🏆 New Best Model"
            
        print(table, "\n")


class TensorBoardLogger(BaseLogger):
    """TensorBoard logging implementation."""
    def __init__(self, output_dir: Union[str, Path]):
        super().__init__(output_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir=str(self.log_dir / f"runs_{timestamp}"))


    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log metrics to TensorBoard with organized grouping by metric type and organ.
        Args:
            metrics: Dictionary of metrics to log with train/val prefixes
            step: Current training step
        """
        # Organize metrics by type and phase (train/val)
        train_metrics = {}
        val_metrics = {}
        
        for name, value in metrics.items():
            # Handle learning rate separately
            if name == 'learning_rate':
                self.writer.add_scalar('training/learning_rate', value, step)
                continue
                
            # Split into train/val metrics
            if name.startswith('train_'):
                train_metrics[name[6:]] = value  # Remove 'train_' prefix
            elif name.startswith('val_'):
                val_metrics[name[4:]] = value    # Remove 'val_' prefix
                
        # Process each phase's metrics
        for phase, phase_metrics in [('train', train_metrics), ('val', val_metrics)]:
            organ_accuracies = {}
            
            for name, value in phase_metrics.items():
                # Handle mean accuracy separately
                if name == 'mean_accuracy':
                    self.writer.add_scalar(f'{phase}/mean_accuracy', value, step)
                    continue
                    
                # Handle organ-specific accuracies
                if name.endswith('_accuracy'):
                    organ = name.replace('_accuracy', '')
                    organ_accuracies[organ] = value
                # Handle loss
                elif name == 'loss':
                    self.writer.add_scalar(f'{phase}/loss', value, step)
                    
            # Log organ accuracies together
            if organ_accuracies:
                self.writer.add_scalars(f'{phase}/organ_accuracies', organ_accuracies, step)


    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to TensorBoard."""
        pass


    def log_model_summary(self, model: Any) -> None:
        """Log model architecture summary."""
        pass  # TensorBoard doesn't support direct model summary logging


    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Log artifacts (not supported in TensorBoard)."""
        pass


    def close(self) -> None:
        """Close the TensorBoard writer."""
        self.writer.close()


class ClearMLLogger(BaseLogger):
    """ClearML logging implementation."""
    
    def __init__(
            self, 
            output_dir: Union[str, Path], 
            project: str, 
            task_name: Optional[str] = None, 
            task_type: Optional[str] = "training",
            reuse_last_task_id: Optional[str] = None, 
            tags: Optional[list] = None
        ):
        super().__init__(output_dir)
        if not CLEARML_AVAILABLE:
            raise ImportError("ClearML package is not installed. Install it with 'pip install clearml'")
        
        # Initialize ClearML task
        self.task = Task.init(
            project_name=project,
            task_name=task_name,
            task_type=task_type,
            continue_last_task=reuse_last_task_id,
            tags=tags,
            output_uri=True,
            auto_connect_frameworks={'pytorch': False}
        )
        self.task.set_initial_iteration(offset=0)

        # Store the logger for easy access
        self.logger = self.task.get_logger()

        # Force iteration-based reporting with a dummy metric
        # Do this early in your script, before any time-consuming operations
        self.logger.report_scalar(
            title="dummy", 
            series="force_iteration_reporting", 
            iteration=0, 
            value=0.0
        )

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)


    def log_metrics(self, metrics: Dict[str, float], step: int, mode: str = 'train') -> None:
        """
        Log metrics to ClearML with organized grouping.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            mode: Mode of operation ('train' or 'val')
        """
        for name, value in metrics.items():
            # Handle learning rate specially
            if name == 'learning_rate':
                self.logger.report_scalar(
                    title="Learning Rate", 
                    series="LR", 
                    value=value,
                    iteration=step
                )
                continue
                
            # Split into appropriate paths based on metric name
            if name.startswith(('train_', 'val_')):
                phase = 'Training' if name.startswith('train_') else 'Validation'
                metric_name = name.replace('train_', '').replace('val_', '')
                
                # Organize metrics into subgroups
                if metric_name == 'loss':
                    title = metric_name
                    series = phase
                elif metric_name == "grad_clip_ratio" or metric_name == "exploding_grad_ratio" or metric_name == "vanishing_grad_ratio":
                    title = "grad_ratio"
                    series = metric_name
                elif "grad" in metric_name:
                    title = "grad"
                    series = metric_name
                elif metric_name == 'dice':
                    title = metric_name
                    series = phase  
                elif metric_name == 'mean_auroc':
                    title = metric_name
                    series = phase  
                elif metric_name == 'mean_auprc':
                    title = metric_name
                    series = phase  
                elif "auroc_class" in metric_name:
                    title = f"{phase}/auroc_class"
                    series = metric_name.replace('auroc_', '')
                elif "auprc_class" in metric_name:
                    title = f"{phase}/auprc_class"
                    series = metric_name.replace('auprc_', '')
                else:
                    title = phase
                    series = metric_name
                    
                self.logger.report_scalar(
                    title=title,
                    series=series,
                    value=value,
                    iteration=step
                )
            else:
                # If no prefix, use the provided mode
                title = 'Training' if mode == 'train' else 'Validation'
                self.logger.report_scalar(
                    title=title,
                    series=name,
                    value=value,
                    iteration=step
                )


    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters to ClearML."""
        # Convert params to ClearML-compatible format
        clearml_params = {}
        
        def convert_value(v: Any) -> Union[float, int, str, bool]:
            if isinstance(v, (float, int, str, bool)):
                return v
            elif isinstance(v, (list, tuple)):
                return str(v)  # Convert lists/tuples to strings
            elif isinstance(v, torch.Tensor):
                return v.item() if v.numel() == 1 else str(v.tolist())
            else:
                return str(v)
        
        def process_dict(d: Dict, parent_key: str = '') -> None:
            for k, v in d.items():
                new_key = f"{parent_key}/{k}" if parent_key else k
                
                if isinstance(v, dict):
                    process_dict(v, new_key)
                else:
                    clearml_params[new_key] = convert_value(v)
        
        process_dict(params)
        
        # Connect parameters to the task
        for key, value in clearml_params.items():
            self.task.set_parameter(key, value)


    def log_model_summary(self, model: Any) -> None:
        """Log model architecture summary."""
        pass
        # model_summary = str(model)
        # self.task.set_model_metadata(summary=model_summary)
        
        # # Optional: log as text artifact
        # self.task.get_logger().report_text(model_summary, "model_summary")


    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """
        Log artifacts to ClearML.
        
        Args:
            local_path: Path to the file to upload
            artifact_path: Name for the artifact in ClearML
        """
        artifact_name = artifact_path or Path(local_path).name
        self.task.upload_artifact(artifact_name, local_path)


    def close(self) -> None:
        """Close ClearML task."""
        # ClearML will automatically close the task when the program exits,
        # but we can explicitly mark it as completed here
        self.task.mark_completed()
        

    def print_validation_summary(self, iteration: int, train_metrics: Dict[str, float],
                               val_metrics: Dict[str, float], is_best: bool) -> None:
        """Print validation summary."""
        summary = f"Iteration {iteration}: Validation results"
        if is_best:
            summary += " (BEST)"
        
        self.logger.report_text(summary, "validation_summary")
        
        # Log text summary for easier review in ClearML UI
        details = "Train metrics:\n"
        for key, value in train_metrics.items():
            details += f"  {key}: {value:.4f}\n"
        
        details += "\nValidation metrics:\n"
        for key, value in val_metrics.items():
            details += f"  {key}: {value:.4f}\n"
            
        self.logger.report_text(details, "validation_details", iteration=iteration)

