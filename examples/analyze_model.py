import sys
import torch
# import deepspeed

from pathlib import Path

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.image_encoder import ViT
from src.models.mask_decoder import MaskDecoder
from src.models.transformer import TwoWayTransformer
from src.models.prompt_encoder import PromptEncoder
from src.models.text_encoder import StandaloneTextEncoder


def image_encoder_test():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample input
    batch_size = 4
    x = torch.randn(batch_size, 3, 256, 256).to(device)

    # Initialize model
    model = ViT().to(device)

    # Test forward pass
    with torch.no_grad():
        output = model(x)

    print(device)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")
    print(f"Total parameters: {model.get_num_params(trainable_only=False):,}")

    # flops, macs, params = deepspeed.profiling.flops_profiler.profiler.get_model_profile(
    #     model=model, args=[x])
    # print(flops, macs, params)


def mask_decoder_test():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample input
    batch_size = 1
    x = torch.randn(batch_size, 768, 64, 64).to(device)
    y = torch.randn(batch_size, 768, 64, 64).to(device)
    z = torch.randn(batch_size, 2, 768).to(device)
    a = torch.randn(batch_size, 768, 64, 64).to(device)
    b = torch.randn(batch_size, 768).to(device)

    # Initialize model
    DEFAULT_PROMPT_EMBED_DIM = 768
    model = MaskDecoder(
    transformer_dim=DEFAULT_PROMPT_EMBED_DIM,
    transformer=TwoWayTransformer(
        depth=2,
        embedding_dim=DEFAULT_PROMPT_EMBED_DIM,
        mlp_dim=2048,
        num_heads=8,
    ),
    num_multimask_outputs=3,
    iou_head_depth=3,
    iou_head_hidden_dim=256,
    ).to(device)


    # Test forward pass
    with torch.no_grad():
        output = model(x, y, z, a, b, multimask_output=False)

    print(device)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output[0].shape}, {output[1].shape}, {output[2].shape}")
    print(f"Trainable parameters: {model.get_num_params(trainable_only=True):,}")
    print(f"Total parameters: {model.get_num_params(trainable_only=False):,}")

    # flops, macs, params = deepspeed.profiling.flops_profiler.profiler.get_model_profile(
    #     model=model, args=[x, y, z, a, b, False])
    # print(flops, macs, params)
    

def prompt_encoder_test():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample input
    batch_size = 1
    x = torch.randn(batch_size, 1, 2).to(device)
    x2 = torch.ones(batch_size, 1).to(device)
    y = torch.randn(batch_size, 1, 4).to(device)
    z = torch.randn(batch_size, 1, 256, 256).to(device)
    a = torch.randn(batch_size, 768).to(device)

    # Initialize model
    DEFAULT_PROMPT_EMBED_DIM = 768
    image_embedding_size = 1024 // 16
    model = PromptEncoder(
        embed_dim=DEFAULT_PROMPT_EMBED_DIM,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(1024, 1024),
        mask_in_chans=16,
    ).to(device)

    # Test forward pass
    with torch.no_grad():
        output = model((x, x2), y, z, a)

    print(device)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output[0].shape}, {output[1].shape}")

    # flops, macs, params = deepspeed.profiling.flops_profiler.profiler.get_model_profile(
    #     model=model, args=[(x, x2), y, z, a])
    # print(flops, macs, params)


def text_encoder_test():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create sample input
    batch_size = 1
    x = torch.randint(269, 49408, (batch_size, 11)).to(device)
    x2 = torch.ones(batch_size, 11).to(device)

    # Initialize model
    model = StandaloneTextEncoder().to(device)

   # Test forward pass
    with torch.no_grad():
        output = model(x, x2)

    print(device)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # flops, macs, params = deepspeed.profiling.flops_profiler.profiler.get_model_profile(
    #     model=model, args=[x, x2])
    # print(flops, macs, params)


if __name__ == '__main__':
    image_encoder_test()
    mask_decoder_test()
    prompt_encoder_test()
    text_encoder_test()

"""
To profile a trained model in inference:
image_encoder: 276.23 G, 137.41 GMACs, 90.46 M
mask_decoder: 32.18 G, 16.07 GMACs, 23.37 M
prompt_encoder: 106.7 M, 51.65 MMACs, 18 K
text_encoder: 834.94 M, 417.12 MMACs, 63.56 M
"""