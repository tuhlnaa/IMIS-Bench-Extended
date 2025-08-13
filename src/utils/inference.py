# src/utils/inference.py
import torch
import numpy as np

from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from rich import print
from typing import Optional, Tuple
from typing import Dict, Any, List, Tuple
from torch.nn.parallel import DistributedDataParallel as DDP

# Import custom modules
from model import IMISNet
from segment_anything.build_sam import get_sam_model
from segment_anything.predictor import IMISPredictor
from src.utils.visualization import VisualizationUtils


def load_model(config: OmegaConf, device: torch.device):
    """Load the SAM model and IMISNet."""
    try:
        sam = get_sam_model(config.model.sam_model_type, config).to(device)
        
        imis_net = IMISNet(
            sam, 
            test_mode=config.model.test_mode, 
            category_weights=config.category_weights_path
        ).to(device)

        if config.device.multi_gpu.enabled:
            imis = DDP(imis, device_ids=[config.device.multi_gpu.rank], output_device=config.device.multi_gpu.rank)

        predictor = IMISPredictor(imis_net, imis_net.encode_image, imis_net.decode_masks)
        print(f"[blue]Model loaded successfully on {device}[/blue]")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    return imis_net, predictor


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
            print("[bold green]✅ Compatible with FlashAttention V2[/bold green]")
        else:
            print("[bold red]❌ Not compatible with FlashAttention V2[/bold red]")
    
    return device


def run_interactive_demo(
    config: OmegaConf,
    image_path: str,
    examples: List[Dict[str, Any]],
    output_dir: str = "output",
) -> Dict[str, Dict[str, Any]]:
    """Run interactive segmentation demonstration with declarative multi-step workflows."""

    image_path = Path(image_path)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    filename_stem = image_path.stem

    # Initialize model
    device = determine_device(config.device.device)
    imis_net, predictor = load_model(config, device)

    # Initialize components
    vis_utils = VisualizationUtils(output_path, filename_stem)

    # Load and display original image
    image = Image.open(image_path).convert('RGB')
    image_array = np.array(image)

    plt.figure()
    plt.imshow(image)
    vis_utils.save_plot("Original Image", "original")

    predictor.set_image(image_array)

    # Process all examples using unified logic
    results = {}
    for example in examples:
        if example.get('workflow_type') == 'multistep':
            # Handle multi-step workflow
            masks, logits, category_pred = run_workflow_chain(config, predictor, vis_utils, image, example['steps'])
            results[example['name']] = {
                'masks': masks,
                'logits': logits,
                'category_pred': category_pred
            }
        else:
            # Handle single-step example
            masks, logits, category_pred = run_segmentation(config, predictor, vis_utils, image, example)
            results[example['name']] = {
                'masks': masks,
                'logits': logits,
                'category_pred': category_pred
            }

    print(f"\nSegmentation demonstration completed. Results saved to {output_path}")
    return results


def run_workflow_chain(
        config: OmegaConf,
        model: IMISPredictor,
        vis_utils: VisualizationUtils,
        image: np.ndarray,
        workflow_steps: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Run a chain of segmentation steps where each step can use results from previous steps.
    
    Args:
        model: MedicalImageSegmentation instance
        vis_utils: VisualizationUtils instance
        image: Input image array
        workflow_steps: List of workflow step dictionaries
    
    Returns:
        tuple: (final_masks, final_logits, final_category_pred)
    """
    previous_logits = None
    
    for i, step in enumerate(workflow_steps):
        # Create a copy to avoid modifying the original
        current_step = step.copy()
        
        # Add previous logits as mask_input if requested and available
        if (current_step.get('use_previous_logits', False) and 
            previous_logits is not None and 
            'mask_input' not in current_step):
            current_step['mask_input'] = previous_logits
        
        # Remove the flag as it's not needed for the actual segmentation
        current_step.pop('use_previous_logits', None)
        
        masks, logits, category_pred = run_segmentation(config, model, vis_utils, image, current_step)
        previous_logits = logits
    
    return masks, logits, category_pred


def run_segmentation(
        config: OmegaConf, 
        model: IMISPredictor,
        vis_utils: VisualizationUtils,
        image: np.ndarray,
        example: Dict[str, Any], 
    ) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Unified function to run segmentation with any input type and visualize results.
    Automatically handles single or multiple text prompts.
    
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
    
    # Handle text prompts automatically based on length
    if text_prompt is not None and len(text_prompt) > 1:
        # Multiple text prompts - process each individually and concatenate masks
        print(f"Processing {len(text_prompt)} text prompts: {text_prompt}")
        masks_list = []
        logits_list = []
        category_preds = []
        
        for i, single_text in enumerate(text_prompt):
            print(f"  Processing prompt {i+1}/{len(text_prompt)}: {single_text}")
            
            multimask_output = config.prediction.multimask_output

            # Run prediction for single text prompt
            single_masks, single_logits, single_category_pred = model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input,
                bounding_box=bounding_box,
                text_prompt=[single_text],  # Wrap single text in list
                multimask_output=multimask_output,
            )
            
            masks_list.append(single_masks)
            logits_list.append(single_logits)
            category_preds.append(single_category_pred)
        
        # Concatenate all masks along the first axis
        masks = np.concatenate(masks_list, axis=0)
        logits = np.concatenate(logits_list, axis=0)
        
        # For category prediction, you might want to combine them or take the first
        # This depends on your specific use case
        category_pred = category_preds[0] if category_preds else None
        
        print(f"Combined masks shape: {masks.shape}")
        print(f"Predicted categories: {category_preds}")
        
    else:
        # Single text prompt or other input types - use original logic
        if text_prompt is not None:
            print(f"Processing single text prompt: {text_prompt}")

        masks, logits, category_pred = model.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=mask_input,
            bounding_box=bounding_box,
            text_prompt=text_prompt,
            multimask_output=multimask_output,
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
    
    # Visualize results - handle multiple masks
    vis_utils.visualize_result(
        image, masks, f"{example['name']} Result", "segmentation", **vis_kwargs
    )
    
    return masks, logits, category_pred