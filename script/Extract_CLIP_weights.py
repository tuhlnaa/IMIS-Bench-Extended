import sys
from omegaconf import OmegaConf
import torch
from pathlib import Path

# Add your project root to path if needed
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import parse_args
from src.utils.inference import determine_device, load_model
from src.models.text_encoder import StandaloneTextEncoder, TextProcessorV2
from segment_anything.build_sam import get_sam_model

class WeightExtractor:
    """Utility class for extracting and managing model weights."""
    
    @staticmethod
    def extract_vit_weights(sam_model) -> dict:
        """Extract ONLY the ViT (image_encoder) weights."""
        vit_state = {}
        
        # Extract image encoder (ViT) weights
        for name, param in sam_model.image_encoder.named_parameters():
            vit_state[name] = param.detach().clone()
            
        return vit_state


    @staticmethod
    def extract_clip_weights(sam_model) -> dict:
        """Extract ONLY the CLIP text model weights."""
        clip_state = {}
        
        # Extract text model weights
        for name, param in sam_model.text_model.named_parameters():
            clip_state[f"text_model.{name}"] = param.detach().clone()
        
        # Extract projection layer weights  
        for name, param in sam_model.text_out_dim.named_parameters():
            clip_state[f"projection.{name}"] = param.detach().clone()
            
        return clip_state

    @staticmethod
    def extract_non_clip_weights(sam_model) -> dict:
        """Extract all weights EXCEPT CLIP-related weights."""
        non_clip_state = {}
        
        # Get all model parameters
        full_state = sam_model.state_dict()
        
        # Filter out text_model and text_out_dim parameters
        for name, param in full_state.items():
            if not name.startswith('text_model.') and not name.startswith('text_out_dim.'):
                non_clip_state[name] = param.detach().clone()
                
        return non_clip_state
    
    @staticmethod
    def extract_non_clip_vit_weights(sam_model) -> dict:
        """Extract all weights EXCEPT CLIP and ViT weights."""
        non_clip_vit_state = {}
        
        # Get all model parameters
        full_state = sam_model.state_dict()
        
        # Filter out text_model, text_out_dim, and image_encoder parameters
        for name, param in full_state.items():
            if (not name.startswith('text_model.') and 
                not name.startswith('text_out_dim.') and 
                not name.startswith('image_encoder.')):
                non_clip_vit_state[name] = param.detach().clone()
                
        return non_clip_vit_state


    @staticmethod
    def extract_non_vit_weights(sam_model) -> dict:
        """Extract all weights EXCEPT ViT (image_encoder) weights."""
        non_vit_state = {}
        
        # Get all model parameters
        full_state = sam_model.state_dict()
        
        # Filter out image_encoder parameters
        for name, param in full_state.items():
            if not name.startswith('image_encoder.'):
                non_vit_state[name] = param.detach().clone()
                
        return non_vit_state

    @staticmethod
    def save_weights(weights_dict: dict, filepath: str):
        """Save weights to file."""
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(weights_dict, filepath)
        print(f"Weights saved to: {filepath}")
        print(f"Number of parameters: {len(weights_dict)}")
        
        # Print some statistics
        total_params = sum(p.numel() for p in weights_dict.values() if hasattr(p, 'numel'))
        print(f"Total parameters: {total_params:,}\n")


def extract_all_weights():
    """Extract ViT weights, CLIP weights, and non-CLIP-ViT weights."""
    # Load weights
    config = parse_args()  # "output/checkpoint/IMISNet-B.pth"
    device = determine_device(config)
    sam_model, _, _ = get_sam_model(config.model.sam_model_type, config)

    weights_dict = sam_model.state_dict()
    print(f"Number of parameters: {len(weights_dict)}")
    total_params = sum(p.numel() for p in weights_dict.values() if hasattr(p, 'numel'))
    print(f"Total parameters: {total_params:,}\n")

    # Create output directory
    output_dir = Path("output/checkpoint")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract ViT weights only
    vit_filepath = output_dir / "vit_weights_only.pth"
    vit_weights = WeightExtractor.extract_vit_weights(sam_model)
    WeightExtractor.save_weights(vit_weights, str(vit_filepath))

    # 2. Extract CLIP weights only
    clip_filepath = output_dir / "CLIP_weights_only.pth"
    clip_weights = WeightExtractor.extract_clip_weights(sam_model)
    WeightExtractor.save_weights(clip_weights, str(clip_filepath))

    # Extract non-CLIP weights
    non_clip_filepath = "output/checkpoint/non_CLIP_weights.pth"
    non_clip_weights = WeightExtractor.extract_non_clip_weights(sam_model)
    WeightExtractor.save_weights(non_clip_weights, non_clip_filepath)

    # Extract non-CLIP and non-ViT weights (prompt_encoder, mask_decoder, etc.)
    non_clip_vit_filepath = output_dir / "IMISNet-B_non_CLIP_vit_weights.pth"
    non_clip_vit_weights = WeightExtractor.extract_non_clip_vit_weights(sam_model)
    WeightExtractor.save_weights(non_clip_vit_weights, str(non_clip_vit_filepath))


def load_and_use_weights():
    """Example of how to load and use the extracted ViT weights."""
    # Load and use CLIP weights
    clip_weights = torch.load("output/checkpoint/CLIP_weights_only.pth", weights_only=True)
    text_encoder = StandaloneTextEncoder()
    text_encoder.load_state_dict(clip_weights)


if __name__ == "__main__":
    # Extract all weight types
    extract_all_weights()
    
    # Example of loading weights
    load_and_use_weights()