import sys
import torch
from pathlib import Path

# Add your project root to path if needed
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import parse_args
from src.utils.inference import determine_device, load_model
from src.models.text_encoder import StandaloneTextEncoder, TextProcessor


class TextWeightExtractor:
    """Utility class for extracting and managing text model weights."""
    
    @staticmethod
    def extract_combined_weights(sam_model) -> dict:
        """Extract both text model and projection weights as a combined state dict."""
        combined_state = {}
        
        # Extract text model weights with prefix
        for name, param in sam_model.text_model.named_parameters():
            combined_state[f"text_model.{name}"] = param.detach().clone()
        
        # Extract projection layer weights with prefix
        for name, param in sam_model.text_out_dim.named_parameters():
            combined_state[f"projection.{name}"] = param.detach().clone()
            
        return combined_state

    @staticmethod
    def save_text_weights(sam_model, filepath: str):
        """Save combined text weights to file."""
        weights = TextWeightExtractor.extract_combined_weights(sam_model)
        torch.save(weights, filepath)


def example_usage():
    """Example of how to use the refactored components."""
    filepath = "output/checkpoint/CLIP_weights_only.pth"

    config = parse_args()
    device = determine_device(config)
    sam_model, predictor, image_encoder = load_model(config, device)

    # Extract to standalone encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    standalone_encoder = StandaloneTextEncoder.from_sam_model(sam_model, device)
    text_processor = TextProcessor(device, text_encoder=standalone_encoder)
    
    # Save/load weights separately
    TextWeightExtractor.save_text_weights(sam_model, filepath)
    text_weights = torch.load(filepath, map_location=device, weights_only=True)

    text_encoder = StandaloneTextEncoder()
    text_encoder.load_state_dict(text_weights)

    text_processor = TextProcessor(device, text_encoder=text_encoder)
    text_processor.text_encoder.load_state_dict(text_weights)
    

if __name__ == "__main__":
    example_usage()
