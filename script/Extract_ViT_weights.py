import torch
import sys
from pathlib import Path
from typing import Dict, Any

# Add your project root to path if needed
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from segment_anything.build_sam import build_sam
from segment_anything.modeling.image_encoder import ViT


def extract_vit_weights_from_sam_checkpoint(
    checkpoint_path: str,
    output_path: str,
    device: str = "cpu"
) -> None:
    """
    Extract ViT weights from a SAM model checkpoint and save them separately.
    
    Args:
        checkpoint_path: Path to the full SAM checkpoint
        output_path: Path where to save the extracted ViT weights
        device: Device to load the checkpoint on
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Load the full checkpoint
    full_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Extract only the image_encoder (ViT) weights
    vit_weights = {}
    
    for key, value in full_checkpoint.items():
        if key.startswith('image_encoder.'):
            # Remove the 'image_encoder.' prefix to get the actual ViT parameter name
            vit_key = key.replace('image_encoder.', '')
            vit_weights[vit_key] = value
    
    print(f"Extracted {len(vit_weights)} ViT parameters")
    print("Sample ViT parameter names:")
    for i, key in enumerate(list(vit_weights.keys())[:5]):
        print(f"  {key}: {vit_weights[key].shape}")
    
    # Save the ViT weights
    torch.save(vit_weights, output_path)
    print(f"ViT weights saved to: {output_path}")


def load_vit_with_extracted_weights(
    vit_weights_path: str,
    config: Dict[str, Any],
    prompt_embed_dim: int = 768,
    device: str = "cpu"
) -> ViT:
    """
    Load a ViT model and initialize it with extracted weights.
    
    Args:
        vit_weights_path: Path to the extracted ViT weights
        config: ViT configuration dictionary
        prompt_embed_dim: Output channels for the ViT
        device: Device to load the model on
    
    Returns:
        ViT model with loaded weights
    """
    print(f"Loading ViT weights from: {vit_weights_path}")
    
    # Create ViT model
    vit_model = ViT(
        encoder_embed_dim=config["encoder_embed_dim"],
        pretrain_model=config["pretrain_model"],
        out_chans=prompt_embed_dim,
        depth=config["encoder_depth"],
        freeze_encoder=False,  # Set to False if you want to fine-tune
        pretrained=False,
    ).to(device)
    
    # Load the extracted weights
    vit_weights = torch.load(vit_weights_path, map_location=device, weights_only=True)
    vit_model.load_state_dict(vit_weights)
    
    print("ViT weights loaded successfully!")
    
    return vit_model


def verify_extraction(
    original_checkpoint_path: str,
    extracted_weights_path: str,
    config: Dict[str, Any],
    prompt_embed_dim: int = 768,
    device: str = "cpu"
) -> bool:
    """
    Verify that the extracted weights match the original checkpoint.
    
    Args:
        original_checkpoint_path: Path to original SAM checkpoint
        extracted_weights_path: Path to extracted ViT weights
        config: ViT configuration
        prompt_embed_dim: Output channels
        device: Device for computation
    
    Returns:
        True if weights match, False otherwise
    """
    print("Verifying extraction...")
    
    # Create original model and load full checkpoint
    original_sam = build_sam(
        config=config,
        encoder_embed_dim=config["encoder_embed_dim"],
        encoder_depth=config["encoder_depth"],
        encoder_num_heads=config["encoder_num_heads"],
        encoder_global_attn_indexes=config["encoder_global_attn_indexes"],
        checkpoint=original_checkpoint_path,
        pretrain_model=config["pretrain_model"],
        prompt_embed_dim=prompt_embed_dim
    )
    
    # Load model with extracted weights
    extracted_vit = load_vit_with_extracted_weights(
        extracted_weights_path, config, prompt_embed_dim, device
    )
    
    # Compare a few key parameters
    original_vit = original_sam.image_encoder
    
    # Test with random input
    test_input = torch.randn(1, 3, 1024, 1024).to(device)
    
    with torch.no_grad():
        original_output = original_vit(test_input)
        extracted_output = extracted_vit(test_input)
        
        # Check if outputs are close
        if torch.allclose(original_output, extracted_output, atol=1e-6):
            print("✅ Verification successful! Outputs match.")
            return True
        else:
            print("❌ Verification failed! Outputs don't match.")
            return False


# Example usage
if __name__ == "__main__":
    # Configuration for ViT-B (modify as needed)
    SAM_CONFIGS = {
        "vit_b": {
            "encoder_embed_dim": 768,
            "encoder_depth": 12,
            "encoder_num_heads": 12,
            "encoder_global_attn_indexes": [2, 5, 8, 11],
            "pretrain_model": "samvit_base_patch16"
        }
    }
    config = SAM_CONFIGS["vit_b"]

    # Paths (modify these to your actual paths)
    sam_checkpoint_path = "output/checkpoint/IMISNet-B.pth"
    vit_output_path = "output/checkpoint/vit_weights_only.pth"
    
    # Extract ViT weights
    extract_vit_weights_from_sam_checkpoint(
        checkpoint_path=sam_checkpoint_path,
        output_path=vit_output_path,
        device="cpu"
    )
    verify_extraction(sam_checkpoint_path, vit_output_path, config)

    # Load ViT with extracted weights
    vit_model = load_vit_with_extracted_weights(
        vit_weights_path=vit_output_path,
        config=config,
        prompt_embed_dim=768,
        device="cpu"
    )
    
    # Test the loaded model
    test_input = torch.randn(1, 3, 1024, 1024)
    with torch.no_grad():
        output = vit_model(test_input)
        print(f"ViT output shape: {output.shape}")