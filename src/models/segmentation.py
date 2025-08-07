# examples/segmentation.py
import torch
import numpy as np

from pathlib import Path
from PIL import Image
from rich import print
from typing import Optional, Tuple, Union
from omegaconf import OmegaConf

# Import custom modules
from segment_anything.build_sam import get_sam_model
from segment_anything.predictor import IMISPredictor
from model import IMISNet


def determine_device(config_device: Optional[str] = None, verbose: bool = True) -> torch.device:
    """
    Determine the appropriate PyTorch device based on configuration and hardware availability.
    """
    # Determine device
    if config_device:
        device = torch.device(config_device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if verbose and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[bold blue]GPU: {gpu_name}[/bold blue]")
        
        # Check compute capability
        major, minor = torch.cuda.get_device_capability(0)
        print(f"[bold blue]Compute capability: {major}.{minor}[/bold blue]")
        
        # FlashAttention V2 needs compute capability >= 8.0 (Ampere+)
        if major >= 8:
            print("[green]✅ Compatible with FlashAttention V2[/green]")
        else:
            print("[red]❌ Not compatible with FlashAttention V2[/red]")
    
    return device


class MedicalImageSegmentation:
    """
    Interactive Medical Image Segmentation demonstration class.
    
    This class provides functionality for loading models, processing images,
    and performing interactive segmentation with clicks, boxes, and text prompts.
    """
    
    def __init__(self, config: OmegaConf):
        """Initialize the segmentation demo."""
        self.config = config
        self.device = determine_device(config.device)
        self._load_model()
        

    def _load_model(self) -> None:
        """Load the SAM model and IMISNet."""
        try:
            sam = get_sam_model(self.config.model.sam_model_type, self.config).to(self.device)
            self.imis_net = IMISNet(
                sam, 
                test_mode=self.config.model.test_mode, 
                category_weights=self.config.category_weights_path
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
        multimask_output: Optional[bool] = None
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Perform segmentation prediction with point interactions.
        
        Args:
            point_coords: Coordinates of interaction points (N, 2)
            point_labels: Labels for points (1 for positive, 0 for negative)
            mask_input: Previous mask logits for refinement
            bounding_box: Bounding box coordinates for guided segmentation
            text_prompt: Optional text description
            multimask_output: Whether to output multiple masks (uses config default if None)
            
        Returns:
            Tuple of (masks, logits, predicted_category)
        """
        try:
            # Use config default if multimask_output not specified
            if multimask_output is None:
                multimask_output = self.config.prediction.multimask_output
            
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