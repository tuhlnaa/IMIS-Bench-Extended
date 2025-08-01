# examples/medical_segmentation_demo.py

import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from argparse import Namespace
from datetime import datetime
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, Union, List

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from segment_anything import sam_model_registry
from segment_anything.predictor import IMISPredictor
from model import IMISNet


class MedicalImageSegmentationDemo:
    """
    Interactive Medical Image Segmentation demonstration class.
    
    This class provides functionality for loading models, processing images,
    and performing interactive segmentation with clicks, boxes, and text prompts.
    """
    
    def __init__(
        self,
        model_checkpoint: str = 'output/checkpoint/IMISNet-B.pth',
        category_weights: str = 'dataloaders/categories_weight.pkl',
        image_size: int = 1024,
        device: Optional[str] = None
    ):
        """
        Initialize the segmentation demo.
        
        Args:
            model_checkpoint: Path to the trained model checkpoint
            category_weights: Path to category weights file
            image_size: Input image size for the model
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = image_size
        
        # Initialize model arguments
        self.args = Namespace()
        self.args.image_size = image_size
        self.args.sam_checkpoint = model_checkpoint
        
        # Load model
        self._load_model(category_weights)
        

    def _load_model(self, category_weights: str) -> None:
        """Load the SAM model and IMISNet."""
        try:
            sam = sam_model_registry["vit_b"](self.args).to(self.device)
            self.imis_net = IMISNet(
                sam, 
                test_mode=True, 
                category_weights=category_weights
            ).to(self.device)
            
            self.predictor = IMISPredictor(self.imis_net)
            print(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess image for segmentation."""
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Set image for predictor
            self.predictor.set_image(image_array)
            
            return image_array
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
    

    def predict(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: Optional[np.ndarray] = None,
        bounding_box: Optional[np.ndarray] = None,
        text_prompt: Optional[str] = None,
        multimask_output: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Perform segmentation prediction with point interactions.
        
        Args:
            point_coords: Coordinates of interaction points (N, 2)
            point_labels: Labels for points (1 for positive, 0 for negative)
            mask_input: Previous mask logits for refinement
            text_prompt: Optional text description
            multimask_output: Whether to output multiple masks
            
        Returns:
            Tuple of (masks, logits, predicted_category)
        """
        try:
            masks, logits, category_pred = self.predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                box=bounding_box,
                text=text_prompt,
                multimask_output=multimask_output,
            )
            
            return masks, logits, category_pred
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")


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
            mask: Binary mask to display
            ax: Matplotlib axes object
            random_color: Whether to use random color
            alpha: Transparency level
        """
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
            labels: np.ndarray, 
            title: str, 
            suffix: str, 
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


def run_interactive_point_demo(image_path: str = 'demo_image/train_177_51.png', output_dir: str = "output") -> None:
    """Run interactive segmentation demonstration with reduced redundancy."""
    image_path = Path(image_path)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    filename_stem = image_path.stem

    # Initialize demo components
    demo = MedicalImageSegmentationDemo()
    vis_utils = VisualizationUtils(output_path, filename_stem)

    # Load and display original image
    image = demo.load_image(image_path)
    plt.figure()
    plt.imshow(image)
    vis_utils.save_plot("Original Image", "original")
    print(f"Image shape: {image.shape}")

    # Point segmentation
    examples = [
        {
            'name': 'Single click segmentation',
            'points': np.array([[188, 205]]),
            'labels': np.array([1]),
        },
        {
            'name': 'Different region segmentation',
            'points': np.array([[346, 211]]),
            'labels': np.array([1]),
        }
    ]
    
    # Run simple examples
    for example in examples:
        print(f"\n=== {example['name']} ===")
        masks, logits, category_pred = demo.predict(
            point_coords=example['points'],
            point_labels=example['labels'],
            mask_input=None,
            multimask_output=False
        )
        print(f"Predicted category: {category_pred}")
        vis_utils.visualize_result(image, masks, example['labels'], points=example['points'], 
            title=f"{example['name']} Result", suffix="segmentation")
    
    # Multi-step refinement segmentation
    print(f"\n=== Multiple click correction ===")
    
    # Initial prediction
    initial_point = np.array([[378, 258]])
    initial_label = np.array([1])
    
    print(f"\n=== Initial Prediction ===")
    masks, logits, category_pred = demo.predict(
        point_coords=initial_point,
        point_labels=initial_label,
        mask_input=None,
        multimask_output=False
    )
    print(f"Predicted category: {category_pred}")
    vis_utils.visualize_result(image, masks, initial_label, points=initial_point,
        title="Initial Prediction", suffix="segmentation")

    # Refinement prediction using previous logits
    refined_point = np.array([[311, 287]])
    refined_label = np.array([1])

    print(f"\n=== Refined Prediction ===")
    masks, logits, category_pred = demo.predict(
        point_coords=refined_point,
        point_labels=refined_label,
        mask_input=logits,
        multimask_output=False
    )
    print(f"Predicted category: {category_pred}")
    vis_utils.visualize_result(image, masks, refined_label, points=refined_point,
        title="Refined Prediction", suffix="segmentation")

    # Bounding box segmentation
    bounding_box = np.array([215, 118, 304, 232])

    masks, logits, category_pred = demo.predict(
        point_coords=None,
        point_labels=None,
        mask_input=None,
        bounding_box=bounding_box,
        text_prompt = None,
        multimask_output=False,
    )

    print(f"Predicted category: {category_pred}")
    vis_utils.visualize_result(image, masks, refined_label, boxes=bounding_box,
        title="Bounding box Prediction", suffix="segmentation")
    
if __name__ == "__main__":
    # Run the interactive demonstration
    run_interactive_point_demo()