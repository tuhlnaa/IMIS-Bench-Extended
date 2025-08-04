import numpy as np
from typing import Dict, Any, Tuple, Optional

from src.models.segmentation import MedicalImageSegmentation
from src.utils.visualization import VisualizationUtils

def run_segmentation(
        model: MedicalImageSegmentation,
        vis_utils: VisualizationUtils,
        image: np.ndarray,
        example: Dict[str, Any],
    ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Unified function to run segmentation with any input type and visualize results.
    
    Args:
        model: MedicalImageSegmentation instance
        vis_utils: VisualizationUtils instance
        image: Input image array
        example: Dictionary containing segmentation parameters
    
    Returns:
        tuple: (masks, logits, category_pred)
    """
    print(f"\n=== {example['name']} ===")
    
    # Extract parameters with defaults
    point_coords = example.get('points', None)
    point_labels = example.get('labels', None)
    mask_input = example.get('mask_input', None)
    bounding_box = example.get('bounding_box', None)
    text_prompt = example.get('text_prompt', None)
    multimask_output = example.get('multimask_output', False)
    
    # Run prediction
    masks, logits, category_pred = model.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        mask_input=mask_input,
        bounding_box=bounding_box,
        text_prompt=text_prompt,
        multimask_output=multimask_output
    )
    
    print(f"Predicted category: {category_pred}")
    
    # Prepare visualization parameters
    vis_kwargs = {
        'labels': point_labels,
        'points': point_coords,
        'boxes': bounding_box
    }
    # Remove None values
    vis_kwargs = {k: v for k, v in vis_kwargs.items() if v is not None}
    
    # Visualize results
    vis_utils.visualize_result(
        image, masks, f"{example['name']} Result", "segmentation", **vis_kwargs
    )
    
    return masks, logits, category_pred