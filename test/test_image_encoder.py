import sys
import time
import torch
from pathlib import Path

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from segment_anything.modeling.image_encoder import ViT

SAM_CONFIGS = {
    "vit_h": {
        "encoder_embed_dim": 1280,
        "encoder_depth": 32,
        "encoder_num_heads": 16,
        "encoder_global_attn_indexes": [7, 15, 23, 31],
        "pretrain_model": "samvit_huge_patch16"
    },
    "vit_l": {
        "encoder_embed_dim": 1024,
        "encoder_depth": 24,
        "encoder_num_heads": 16,
        "encoder_global_attn_indexes": [5, 11, 17, 23],
        "pretrain_model": "samvit_large_patch16"
    },
    "vit_b": {
        "encoder_embed_dim": 768,
        "encoder_depth": 12,
        "encoder_num_heads": 12,
        "encoder_global_attn_indexes": [2, 5, 8, 11],
        "pretrain_model": "samvit_base_patch16"
    }
}


def test_torch_model():
    """Test the MaskDecoder with random inputs."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    batch_size = 1
    prompt_embed_dim = 768
    image_size = 1024
    device = "cuda"

    config = SAM_CONFIGS["vit_b"]

    # Create transformer
    image_encoder = ViT(
            encoder_embed_dim=config["encoder_embed_dim"],
            pretrain_model=config["pretrain_model"],
            out_chans=prompt_embed_dim,
            depth=config["encoder_depth"],
            freeze_encoder=True,
            pretrained=False,
        ).to(device)
    
    # Create random inputs
    image = torch.randn(batch_size, 3, image_size, image_size).to(device)

    image_embedding =image_encoder(image)
    print(image_embedding.shape)
    
    print(f"\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_torch_model()

"""
torch.Size([1, 768, 64, 64])

✅ All tests completed successfully!
"""



