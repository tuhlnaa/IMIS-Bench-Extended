# 001.py
"""
Interactive Medical Image Segmentation: A Benchmark Dataset and Baseline

Interactive Medical Image Segmentation (IMIS) has long been constrained by the limited 
availability of large-scale, diverse, and densely annotated datasets, which hinders model 
generalization and consistent evaluation across different models. This module demonstrates 
the IMed-361M benchmark dataset and provides a baseline network for interactive segmentation.

The IMed-361M dataset spans 14 modalities and 204 segmentation targets, totaling 361 million 
masks—an average of 56 masks per image.
"""

import warnings
from argparse import Namespace
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# Suppress flash attention warning
warnings.filterwarnings("ignore", message="1Torch was not compiled with flash attention.")

# Import custom modules (assuming they exist in the project)
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
        """
        Load and preprocess image for segmentation.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            image = Image.open(image_path).convert('RGB')
            image_array = np.array(image)
            
            # Set image for predictor
            self.predictor.set_image(image_array)
            
            return image_array
            
        except Exception as e:
            raise RuntimeError(f"Failed to load image {image_path}: {e}")
    
    def predict_with_points(
        self,
        point_coords: np.ndarray,
        point_labels: np.ndarray,
        mask_input: Optional[np.ndarray] = None,
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
                text=text_prompt,
                multimask_output=multimask_output,
            )
            
            return masks, logits, category_pred
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")


class VisualizationUtils:
    """Utility class for visualization functions."""
    
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
        marker_size: int = 375
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
                color='green', marker='*', s=marker_size, 
                edgecolor='white', linewidth=1.25
            )
            
        if len(negative_points) > 0:
            ax.scatter(
                negative_points[:, 0], negative_points[:, 1], 
                color='red', marker='*', s=marker_size, 
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


def run_interactive_demo(image_path: str = 'demo_image/train_177_51.png') -> None:
    """
    Run interactive segmentation demonstration.
    
    Args:
        image_path: Path to demonstration image
    """
    # Initialize demo
    demo = MedicalImageSegmentationDemo()
    vis_utils = VisualizationUtils()
    
    # Load and display image
    try:
        image = demo.load_image(image_path)
        print(f"Image shape: {image.shape}")
        
        # Display original image
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('on')
        plt.show()
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Example 1: Single click segmentation (kidney)
    print("\n=== Example 1: Single click segmentation ===")
    input_point = np.array([[188, 205]])
    input_label = np.array([1])
    
    # Show input points
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    vis_utils.show_points(input_point, input_label, plt.gca())
    plt.title("Input Points")
    plt.axis('on')
    plt.show()
    
    # Predict and display result
    try:
        masks, logits, category_pred = demo.predict_with_points(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        
        print(f"Predicted category: {category_pred}")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        vis_utils.show_mask(masks, plt.gca())
        vis_utils.show_points(input_point, input_label, plt.gca())
        plt.title("Segmentation Result")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error in prediction: {e}")
    
    # Example 2: Different region
    print("\n=== Example 2: Different region segmentation ===")
    input_point = np.array([[346, 211]])
    input_label = np.array([1])
    
    try:
        masks, logits, category_pred = demo.predict_with_points(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        
        print(f"Predicted category: {category_pred}")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        vis_utils.show_mask(masks, plt.gca())
        vis_utils.show_points(input_point, input_label, plt.gca())
        plt.title("Second Region Segmentation")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error in prediction: {e}")
    
    # Example 3: Multiple click correction
    print("\n=== Example 3: Multiple click correction ===")
    
    # Initial click
    input_point = np.array([[378, 258]])
    input_label = np.array([1])
    
    try:
        masks, logits, category_pred = demo.predict_with_points(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False
        )
        
        print(f"Initial prediction category: {category_pred}")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        vis_utils.show_mask(masks, plt.gca())
        vis_utils.show_points(input_point, input_label, plt.gca())
        plt.title("Initial Prediction")
        plt.axis('off')
        plt.show()
        
        # Refinement click
        input_point_refined = np.array([[311, 287]])
        input_label_refined = np.array([1])
        
        masks_refined, logits_refined, category_pred_refined = demo.predict_with_points(
            point_coords=input_point_refined,
            point_labels=input_label_refined,
            mask_input=logits,  # Use previous logits for refinement
            multimask_output=False
        )
        
        print(f"Refined prediction category: {category_pred_refined}")
        
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        vis_utils.show_mask(masks_refined, plt.gca())
        vis_utils.show_points(input_point_refined, input_label_refined, plt.gca())
        plt.title("Refined Prediction")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error in multi-click prediction: {e}")


if __name__ == "__main__":
    # Run the interactive demonstration
    run_interactive_demo()