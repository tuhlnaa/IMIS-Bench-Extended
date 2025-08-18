import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Tuple

def compute_segmentation_metrics(self, pred: torch.Tensor, label: torch.Tensor) -> Tuple[float, float]:
    """
    Compute IoU and Dice metrics for predictions and labels.
    
    Args:
        pred: Predicted masks [B, C, H, W]
        label: Ground truth labels [B, C, H, W]
        
    Returns:
        Tuple of (IoU, Dice) scores
    """
    assert pred.shape == label.shape, f"Shape mismatch: {pred.shape} vs {label.shape}"
    
    # Apply sigmoid and threshold for predictions
    pred_binary = (torch.sigmoid(pred) > 0.5)
    label_binary = (label > 0)
    
    # Calculate intersection and union
    intersection = torch.logical_and(pred_binary, label_binary).sum(dim=(1, 2, 3))
    union = torch.logical_or(pred_binary, label_binary).sum(dim=(1, 2, 3))
    
    # Calculate metrics with epsilon for numerical stability
    eps = 1e-8
    iou = intersection.float() / (union.float() + eps)
    
    pred_sum = pred_binary.sum(dim=(1, 2, 3))
    label_sum = label_binary.sum(dim=(1, 2, 3))
    dice = (2 * intersection.float()) / (pred_sum + label_sum + eps)
    
    return iou.mean().item(), dice.mean().item()


class SegmentationMetrics(nn.Module):
    """A PyTorch module for computing segmentation metrics including IoU and Dice scores."""

    def __init__(self, split: str = 'val'):
        """
        Initialize the segmentation metrics tracker.

        Args:
            split: The data split ('train', 'val', or 'test')
        """
        super().__init__()
        self.split = split
        
        # Reset to initialize states
        self.reset()


    def reset(self):
        """Reset all accumulated states for a new computation cycle."""
        self.dice_scores = []
        self.iou_scores = []
        self.loss_sum = 0.0
        self.num_samples = 0


    def _compute_single_metrics(self, pred: torch.Tensor, label: torch.Tensor) -> Tuple[float, float]:
        """
        Compute IoU and Dice metrics for predictions and labels.
        
        Args:
            pred: Predicted masks [B, C, H, W]
            label: Ground truth labels [B, C, H, W]
            
        Returns:
            Tuple of (IoU, Dice) scores
        """
        assert pred.shape == label.shape, f"Shape mismatch: {pred.shape} vs {label.shape}"
        
        # Apply sigmoid and threshold for predictions
        pred_binary = (torch.sigmoid(pred) > 0.5)
        label_binary = (label > 0)
        
        # Calculate intersection and union
        intersection = torch.logical_and(pred_binary, label_binary).sum(dim=(1, 2, 3))
        union = torch.logical_or(pred_binary, label_binary).sum(dim=(1, 2, 3))
        
        # Calculate metrics with epsilon for numerical stability
        eps = 1e-8
        iou = intersection.float() / (union.float() + eps)
        
        pred_sum = pred_binary.sum(dim=(1, 2, 3))
        label_sum = label_binary.sum(dim=(1, 2, 3))
        dice = (2 * intersection.float()) / (pred_sum + label_sum + eps)
        
        return iou.mean().item(), dice.mean().item()


    def update(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            loss: Optional[torch.Tensor] = None
        ) -> None:
        """
        Update states with predictions and targets from a new batch.
        
        Args:
            preds: Model predictions/logits [B, C, H, W]
            targets: Ground truth labels [B, C, H, W]
            loss: Optional loss value for this batch
        """
        # Compute metrics for this batch
        iou_score, dice_score = self._compute_single_metrics(preds, targets)
        
        # Accumulate metrics
        self.iou_scores.append(iou_score)
        self.dice_scores.append(dice_score)
        
        # Update loss tracking if provided
        if loss is not None:
            self.loss_sum += loss.item()
        
        self.num_samples += 1


    def update_multi_class(
            self,
            preds_list: List[torch.Tensor],
            targets_list: List[torch.Tensor],
            loss_list: Optional[List[torch.Tensor]] = None
        ) -> None:
        """
        Update states with multi-class predictions (for processing multiple classes per sample).
        
        Args:
            preds_list: List of predictions for each class
            targets_list: List of targets for each class
            loss_list: Optional list of loss values for each class
        """
        # Compute metrics for each class and average
        class_ious = []
        class_dices = []
        
        for i, (pred, target) in enumerate(zip(preds_list, targets_list)):
            iou_score, dice_score = self._compute_single_metrics(pred, target)
            class_ious.append(iou_score)
            class_dices.append(dice_score)
        
        # Store average metrics across classes for this sample
        avg_iou = np.mean(class_ious)
        avg_dice = np.mean(class_dices)
        
        self.iou_scores.append(avg_iou)
        self.dice_scores.append(avg_dice)
        
        # Update loss tracking if provided
        if loss_list is not None:
            avg_loss = np.mean([loss.item() for loss in loss_list])
            self.loss_sum += avg_loss
        
        self.num_samples += 1


    def compute(self) -> Dict[str, float]:
        """Compute final metrics from accumulated data."""

        if not self.dice_scores:
            print(f"Warning: No data accumulated")
            return {f"{self.split}_loss": 0.0}
        
        # Calculate mean metrics, filtering out NaN values
        mean_dice = float(np.mean([x for x in self.dice_scores if not np.isnan(x)]))
        mean_iou = float(np.mean([x for x in self.iou_scores if not np.isnan(x)]))
        mean_loss = self.loss_sum / self.num_samples if self.num_samples > 0 else 0.0
        
        # Create metrics dictionary
        metrics = {
            f"{self.split}_loss": mean_loss,
            f"{self.split}_mean_iou": mean_iou,
            f"{self.split}_mean_dice": mean_dice,
            f"{self.split}_total_samples": self.num_samples
        }
        
        # Print metrics summary
        print(f"{self.split.capitalize()} metrics - Loss: {mean_loss:.4f}, "
              f"IoU: {mean_iou:.4f}, Dice: {mean_dice:.4f}, "
              f"Samples: {self.num_samples}")
        
        return metrics


    def get_current_scores(self) -> Tuple[List[float], List[float]]:
        """
        Get current accumulated scores without computing final metrics.
        
        Returns:
            Tuple of (IoU scores list, Dice scores list)
        """
        return self.iou_scores.copy(), self.dice_scores.copy()


    def forward(
            self, 
            preds: torch.Tensor, 
            targets: torch.Tensor, 
            loss: Optional[torch.Tensor] = None, 
            compute_metrics: bool = False
        ) -> Optional[Dict[str, float]]:
        """
        Forward pass that updates metrics and optionally computes them.

        Args:
            preds: Model predictions/logits [B, C, H, W]
            targets: Ground truth labels [B, C, H, W]
            loss: Optional loss value for this batch
            compute_metrics: Whether to compute and return metrics after update

        Returns:
            Optional[Dict[str, float]]: Dictionary of metrics if compute_metrics is True, else None
        """
        self.update(preds, targets, loss)
        
        if compute_metrics:
            return self.compute()
        return None

