import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path
from typing import Optional

class VisualizationUtils:
    """Helper class to manage demo execution with integrated visualization utilities."""
    
    def __init__(self, output_path: Path, filename_stem: str):
        self.output_path = output_path
        self.filename_stem = filename_stem

        
    @staticmethod
    def show_mask(
            mask: np.ndarray,
            ax: plt.Axes,
            random_color: bool = False,
            alpha: float = 0.6
        ) -> None:
        """
        Display segmentation mask on matplotlib axes.
    
        Args:
            mask: Binary mask (H, W) or (1, H, W) or (N, 1, H, W) to display
            ax: Matplotlib axes object
            random_color: Whether to use random color
            alpha: Transparency level
        """
        # Handle multiple masks by combining them (logical OR)
        if mask.ndim == 4:
            mask = np.any(mask, axis=0)
        
        if mask.ndim == 3 and mask.shape[0] == 1:
            mask = mask[0]
        
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([alpha])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, alpha])
        
        height, width = mask.shape[-2:]
        mask_image = mask.reshape(height, width, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    @staticmethod
    def show_points(
            coords: np.ndarray, 
            labels: np.ndarray, 
            ax: plt.Axes, 
            marker_size: int = 96
        ) -> None:
        """
        Display interaction points on matplotlib axes.
        
        Args:
            coords: Point coordinates (N, 2)
            labels: Point labels (N,)
            ax: Matplotlib axes object
            marker_size: Size of point markers
        """
        positive_points = coords[labels == 1]
        negative_points = coords[labels == 0]
        
        if len(positive_points) > 0:
            ax.scatter(
                positive_points[:, 0], positive_points[:, 1], 
                color='green', marker='P', s=marker_size, 
                edgecolor='white', linewidth=1.25
            )
            
        if len(negative_points) > 0:
            ax.scatter(
                negative_points[:, 0], negative_points[:, 1], 
                color='red', marker='P', s=marker_size, 
                edgecolor='white', linewidth=1.25
            )
    

    @staticmethod
    def show_box(box: np.ndarray, ax: plt.Axes) -> None:
        """
        Display bounding box on matplotlib axes.
        
        Args:
            box: Bounding box coordinates [x0, y0, x1, y1]
            ax: Matplotlib axes object
        """
        x0, y0 = box[0], box[1]
        width, height = box[2] - box[0], box[3] - box[1]
        
        ax.add_patch(plt.Rectangle(
            (x0, y0), width, height, 
            edgecolor='green', facecolor=(0, 0, 0, 0), 
            linewidth=2
        ))
    

    def save_plot(self, title: str, suffix: str) -> None:
        """Save current plot with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        plt.title(title)
        output_file = self.output_path / f'{self.filename_stem}_{suffix}_{timestamp}.png'
        plt.savefig(output_file, dpi=200, bbox_inches='tight')
        plt.close()
    

    def visualize_result(
            self, 
            image: np.ndarray, 
            masks: np.ndarray, 
            title: str, 
            suffix: str, 
            labels: Optional[np.ndarray] = None,
            points: Optional[np.ndarray] = None,
            boxes: Optional[np.ndarray] = None
        ) -> None:
        """Create and save visualization with mask and points."""
        plt.figure()
        plt.imshow(image)
        self.show_mask(masks, plt.gca())

        if points is not None:
            self.show_points(points, labels, plt.gca())
        elif boxes is not None:
            self.show_box(boxes, plt.gca())

        self.save_plot(title, suffix)
