# examples/medical_segmentation_demo.py
import torch
import numpy as np

from argparse import Namespace
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, Union

from segment_anything import sam_model_registry
from segment_anything.predictor import IMISPredictor
from model import IMISNet


class MedicalImageSegmentation:
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