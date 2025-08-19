import sys
import time
import torch
from pathlib import Path

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from segment_anything.modeling.mask_decoder import MaskDecoder
from segment_anything.modeling.transformer import TwoWayTransformer


def test_torch_model():
    """Test the MaskDecoder with random inputs."""
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Configuration
    batch_size = 1
    prompt_embed_dim = 768
    image_size = 64
    device = "cuda"

    # Create transformer
    transformer = TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            )
    
    # Initialize MaskDecoder
    mask_decoder = MaskDecoder(
        transformer_dim=prompt_embed_dim,
        transformer=transformer,
        num_multimask_outputs=3,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    ).to(device)
    
    print(f"MaskDecoder initialized with {prompt_embed_dim}D transformer")
    print(f"Number of parameters: {sum(p.numel() for p in mask_decoder.parameters()):,}")

    # Create random inputs
    image_embeddings = torch.randn(batch_size, prompt_embed_dim, image_size, image_size).to(device)
    image_pe = torch.randn(1, prompt_embed_dim, image_size, image_size).to(device)
    sparse_prompt_embeddings = torch.randn(batch_size, 3, prompt_embed_dim).to(device)
    dense_prompt_embeddings = torch.randn(batch_size, prompt_embed_dim, image_size, image_size).to(device)
    text_prompt_embeddings = torch.randn(batch_size, prompt_embed_dim).to(device)
    
    print(f"\nInput shapes:")
    print(f"  image embeddings: {image_embeddings.shape}")
    print(f"  image positional encoding: {image_pe.shape}")
    print(f"  sparse prompt embeddings: {sparse_prompt_embeddings.shape}")
    print(f"  dense prompt embeddings: {dense_prompt_embeddings.shape}")
    print(f"  text prompt embeddings: {text_prompt_embeddings.shape}")
    
    # Test with multimask output
    print(f"\n--- Testing with multimask_output=True ---")
    mask_decoder.eval()
    with torch.no_grad():
        start = time.monotonic()
        outputs = mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            text_prompt_embeddings=text_prompt_embeddings,
            multimask_output=True,
        )
        end = time.monotonic()
        print(f"Time: {end - start:.3f}s")
    
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # Test with single mask output
    print(f"\n--- Testing with multimask_output=False ---")
    with torch.no_grad():
        start = time.monotonic()
        outputs = mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            text_prompt_embeddings=text_prompt_embeddings,
            multimask_output=False,
        )
        end = time.monotonic()
        print(f"Time: {end - start:.3f}s")

    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test without text embeddings
    print(f"\n--- Testing without text embeddings ---")
    with torch.no_grad():
        start = time.monotonic()
        outputs = mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            text_prompt_embeddings=None,
            multimask_output=True,
        )
        end = time.monotonic()
        print(f"Time: {end - start:.3f}s")

    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    print(f"\n✅ All tests completed successfully!")


if __name__ == "__main__":
    test_torch_model()

"""
CPU Time: 0.171s
RTX2080 Ti Time: 0.016s

Input shapes:
  image embeddings: torch.Size([1, 768, 64, 64])
  image positional encoding: torch.Size([1, 768, 64, 64])
  sparse prompt embeddings: torch.Size([1, 3, 768])
  dense prompt embeddings: torch.Size([1, 768, 64, 64])
  text prompt embeddings: torch.Size([1, 768])

--- Testing with multimask_output=True ---
  low_res_masks: torch.Size([1, 3, 256, 256])
  iou_pred: torch.Size([1, 3])
  semantic_pred: torch.Size([1, 3, 512])

--- Testing with multimask_output=False ---
  low_res_masks: torch.Size([1, 1, 256, 256])
  iou_pred: torch.Size([1, 1])
  semantic_pred: torch.Size([1, 1, 512])

--- Testing without text embeddings ---
  low_res_masks: torch.Size([1, 3, 256, 256])
  iou_pred: torch.Size([1, 3])
  semantic_pred: torch.Size([1, 3, 512])

✅ All tests completed successfully!
"""