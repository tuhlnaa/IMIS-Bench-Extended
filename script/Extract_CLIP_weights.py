import sys
import torch
from pathlib import Path

# Add your project root to path if needed
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import parse_args
from src.utils.inference import determine_device, load_model
from src.models.text_encoder import StandaloneTextEncoder, TextProcessorV2

class TextWeightExtractor:
    """Utility class for extracting and managing text model weights."""
    
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
    def save_weights(weights_dict: dict, filepath: str):
        """Save weights to file."""
        torch.save(weights_dict, filepath)
        print(f"Weights saved to: {filepath}")

def example_usage():
    """Example of extracting both CLIP and non-CLIP weights."""
    
    config = parse_args()
    device = determine_device(config)
    sam_model, predictor, image_encoder = load_model(config, device)

    # Extract CLIP weights (as before)
    clip_filepath = "output/checkpoint/CLIP_weights_only.pth"
    clip_weights = TextWeightExtractor.extract_clip_weights(sam_model)
    TextWeightExtractor.save_weights(clip_weights, clip_filepath)

    # Extract non-CLIP weights (everything else)
    non_clip_filepath = "output/checkpoint/non_CLIP_weights.pth"
    non_clip_weights = TextWeightExtractor.extract_non_clip_weights(sam_model)
    TextWeightExtractor.save_weights(non_clip_weights, non_clip_filepath)

    # Usage example
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and use CLIP weights
    clip_weights = torch.load(clip_filepath, map_location=device, weights_only=True)
    text_encoder = StandaloneTextEncoder()
    text_encoder.load_state_dict(clip_weights)
    text_processor = TextProcessorV2(device, text_encoder=text_encoder)
    
    # Load non-CLIP weights for other components
    non_clip_weights = torch.load(non_clip_filepath, map_location=device, weights_only=True)
    
    print("Successfully separated CLIP and non-CLIP weights!")

if __name__ == "__main__":
    example_usage()