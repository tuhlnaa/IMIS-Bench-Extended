# src/utils/inference.py
import torch
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image
from rich import print
from typing import Optional, Tuple, Dict, Any, List
from torch.nn.parallel import DistributedDataParallel as DDP
from dataclasses import dataclass

# Import custom modules
from model import IMISNet
from src.utils.build_sam import get_sam_model
from src.utils.predictor import IMISPredictor, ImagePreprocessor
from src.utils.visualization import VisualizationUtils

@dataclass
class ImageState:
    """Holds preprocessed image state to avoid reprocessing"""
    image: np.ndarray
    input_tensor: torch.Tensor
    features: torch.Tensor
    original_size: tuple
    image_path: str


class InteractiveSegmentationSession:
    """
    Main refactoring approach: Session-based management
    Separates model initialization from image processing and example execution.
    """
    
    def __init__(self, config: OmegaConf, output_dir: str = "output"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model components once
        self.device = determine_device(config)
        self.model, self.predictor, self.image_encoder = load_model(config, self.device)
        self.image_preprocessor = ImagePreprocessor(
            image_size=(config.model.image_size, config.model.image_size)
        )
        
        # State management
        self.current_image_state: Optional[ImageState] = None
        
        print(f"[green]Session initialized successfully[/green]")
    

    def load_image(self, image_path: str) -> ImageState:
        """Load and preprocess image, caching the result"""
        image_path = Path(image_path)
        
        # Check if we already have this image loaded
        if (self.current_image_state is not None and 
            self.current_image_state.image_path == str(image_path)):
            print(f"[yellow]Image already loaded: {image_path.name}[/yellow]")
            return self.current_image_state
        
        print(f"[blue]Loading new image: {image_path.name}[/blue]")
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_array = np.array(image)
        
        # Preprocess and encode
        input_tensor = self.image_preprocessor.preprocess_image(image_array, "RGB")
        features = self.image_encoder(input_tensor.to(self.device))
        
        # Update predictor state
        self.predictor.features = features
        self.predictor.original_size = image_array.shape[:2]
        self.predictor.is_image_set = True
        
        # Cache the state
        self.current_image_state = ImageState(
            image=image_array,
            input_tensor=input_tensor,
            features=features,
            original_size=image_array.shape[:2],
            image_path=str(image_path)
        )
        
        return self.current_image_state
    

    def run_examples(self, examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Run examples on currently loaded image"""
        if self.current_image_state is None:
            raise ValueError("No image loaded. Call load_image() first.")
        
        # Initialize visualization utils
        image_path = Path(self.current_image_state.image_path)
        vis_utils = VisualizationUtils(self.output_dir, image_path.stem)
        
        # Display original image
        plt.figure()
        plt.imshow(self.current_image_state.image)
        vis_utils.save_plot("Original Image", "original")
        
        # Process examples
        results = {}
        for example in examples:
            if example.get('workflow_type') == 'multistep':
                masks, logits, category_pred = run_workflow_chain(
                    self.config, self.predictor, vis_utils, 
                    self.current_image_state.image, example['steps']
                )
            else:
                masks, logits, category_pred = run_segmentation(
                    self.config, self.predictor, vis_utils, 
                    self.current_image_state.image, example
                )
            
            results[example['name']] = {
                'masks': masks,
                'logits': logits,
                'category_pred': category_pred
            }
        
        print(f"\nExamples completed for {image_path.name}")
        return results
    

    def process_image_with_examples(self, image_path: str, examples: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Convenience method: load image and run examples in one call"""
        self.load_image(image_path)
        return self.run_examples(examples)


# Keep original functions for backward compatibility and internal use
def load_model(config: OmegaConf, device: torch.device):
    """Load the SAM model and IMISNet."""
    try:
        sam, image_encoder, text_encoder = get_sam_model(config.model.sam_model_type, config)
        sam, image_encoder, text_encoder = sam.to(device), image_encoder.to(device), text_encoder.to(device)

        imis_net = IMISNet(
            config, 
            sam, 
            text_encoder,
            test_mode=config.model.test_mode, 
            category_weights=config.category_weights_path
        ).to(device)

        if config.device.multi_gpu.enabled:
            imis_net = DDP(imis_net, device_ids=[config.device.multi_gpu.rank], 
                          output_device=config.device.multi_gpu.rank)

        predictor = IMISPredictor(config, imis_net, imis_net.decode_masks)
        print(f"[blue]Model loaded successfully on {device}[/blue]")
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
    
    return imis_net, predictor, image_encoder


def determine_device(config: OmegaConf = None, verbose: bool = True) -> torch.device:
    """Determine the appropriate PyTorch device based on configuration and hardware availability."""
    if config.device.device:
        device = torch.device(config.device.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[bold blue]Device used: {device}[/bold blue]")

    if config.device.multi_gpu.enabled:
        config.device.world_size = config.device.multi_gpu.nodes * len(config.device.multi_gpu.gpu_ids)

    if verbose and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[bold blue]GPU: {gpu_name}[/bold blue]")
        
        major, minor = torch.cuda.get_device_capability(0)
        print(f"[bold blue]Compute capability: {major}.{minor}[/bold blue]")
        
        if major >= 8:
            print("[bold green]✅ Compatible with FlashAttention V2[/bold green]")
        else:
            print("[bold red]❌ Not compatible with FlashAttention V2[/bold red]")
    
    return device


def run_workflow_chain(
        config: OmegaConf,
        model: IMISPredictor,
        vis_utils: VisualizationUtils,
        image: np.ndarray,
        workflow_steps: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray, str]:
    """Run a chain of segmentation steps where each step can use results from previous steps."""
    previous_logits = None
    
    for i, step in enumerate(workflow_steps):
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
    """Unified function to run segmentation with any input type and visualize results."""
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

            single_masks, single_logits, single_category_pred = model.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                bounding_box=bounding_box,
                text_prompt=[single_text],  # Wrap single text in list
                mask_input=mask_input,
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
    
    # Visualize results
    vis_utils.visualize_result(
        image, masks, f"{example['name']} Result", "segmentation", **vis_kwargs
    )
    
    return masks, logits, category_pred


# Backward compatibility function (deprecated)
def run_interactive_demo(
    config: OmegaConf,
    image_path: str,
    examples: List[Dict[str, Any]],
    output_dir: str = "output",
) -> Dict[str, Dict[str, Any]]:
    """
    DEPRECATED: Use InteractiveSegmentationSession for better performance.
    This function is kept for backward compatibility.
    """
    print("[yellow]Warning: run_interactive_demo is deprecated. Use InteractiveSegmentationSession for better performance.[/yellow]")
    
    session = InteractiveSegmentationSession(config, output_dir)
    return session.process_image_with_examples(image_path, examples)

