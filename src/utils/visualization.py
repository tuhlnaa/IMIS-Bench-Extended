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
            colormap: str = 'tab20',
            alpha: float = 0.6
        ) -> None:
        """
        Display segmentation mask using matplotlib colormap.
        
        Args:
            mask: Binary mask (N, 1, H, W) to display
            ax: Matplotlib axes object
            colormap: Matplotlib colormap name
            alpha: Transparency level
        """
        if mask.ndim != 4:
            raise ValueError("This function expects 4D input (N, 1, H, W)")
        
        N, _, H, W = mask.shape
        cmap = plt.cm.get_cmap(colormap)
        
        # Create instance mask (each pixel gets the index of the first mask that covers it)
        instance_mask = np.zeros((H, W), dtype=int)
        
        for i in range(N):
            if np.any(mask[i, 0]):
                # Assign class index (i+1) to avoid confusion with background (0)
                instance_mask = np.where(mask[i, 0] > 0, i + 1, instance_mask)
        
        # Apply colormap
        colored_mask = cmap(instance_mask / N)
        colored_mask[:, :, 3] = np.where(instance_mask > 0, alpha, 0)  # Set alpha
        
        ax.imshow(colored_mask)


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
        Display bounding box(es) on matplotlib axes.
    
        Args:
            box: Bounding box coordinates. Can be:
                - Single box: [x0, y0, x1, y1] with shape (4,)
                - Multiple boxes: [[x0, y0, x1, y1], ...] with shape (N, 4)
            ax: Matplotlib axes object
        """
        # Ensure box is 2D for consistent processing
        if box.ndim == 1:
            boxes = box.reshape(1, -1)  # (4,) -> (1, 4)
        else:
            boxes = box  # (N, 4)
        
        # Draw each bounding box
        for i, single_box in enumerate(boxes):
            x0, y0, x1, y1 = single_box
            width, height = x1 - x0, y1 - y0
            
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
