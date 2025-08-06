from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from pathlib import Path

class BaseLogger(ABC):
    """Abstract base class for logging implementations."""
    
    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.log_dir = self.output_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, float], step: int, mode: str = 'train') -> None:
        """Log metrics during training/validation."""
        pass

    @abstractmethod
    def log_hyperparameters(self, params: Dict[str, Any]) -> None:
        """Log hyperparameters."""
        pass

    @abstractmethod
    def log_model_summary(self, model: Any) -> None:
        """Log model architecture summary."""
        pass

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str = None) -> None:
        """Log artifacts (e.g., model checkpoints)."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Cleanup and close logger."""
        pass

    def print_validation_summary(self, iteration: int, train_metrics: Dict[str, float],
                               val_metrics: Dict[str, float], is_best: bool) -> None:
        """Print validation summary (implementation optional)."""
        pass