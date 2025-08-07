import torch
from functools import partial
from typing import Dict, Any, Optional
from .modeling import ViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
from torch.nn import functional as F


# Model configuration constants
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

# Default model constants
DEFAULT_PROMPT_EMBED_DIM = 768
DEFAULT_VIT_PATCH_SIZE = 16
DEFAULT_PIXEL_MEAN = [123.675, 116.28, 103.53]
DEFAULT_PIXEL_STD = [58.395, 57.12, 57.375]


def build_sam(
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes: list,
    image_size: int,
    checkpoint: Optional[str],
    pretrain_model: str,
    prompt_embed_dim: int = DEFAULT_PROMPT_EMBED_DIM,
    vit_patch_size: int = DEFAULT_VIT_PATCH_SIZE
) -> Sam:
    """
    Build a SAM model with specified configuration.
    
    Args:
        encoder_embed_dim: Vision transformer embedding dimension
        encoder_depth: Number of transformer layers
        encoder_num_heads: Number of attention heads
        encoder_global_attn_indexes: Layers that use global attention
        image_size: Input image size
        checkpoint: Path to checkpoint file (optional)
        pretrain_model: Name of pretrained model
        prompt_embed_dim: Prompt encoder embedding dimension
        vit_patch_size: Vision transformer patch size
    
    Returns:
        Configured SAM model
    """
    image_embedding_size = image_size // vit_patch_size
    
    sam = Sam(
        image_encoder=ViT(
            encoder_embed_dim=encoder_embed_dim,
            pretrain_model=pretrain_model,
            out_chans=prompt_embed_dim,
            depth=encoder_depth,
            freeze_encoder=True,
            pretrained=False,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        text_model=CLIPTextModel(CLIPTextConfig()),
        pixel_mean=DEFAULT_PIXEL_MEAN,
        pixel_std=DEFAULT_PIXEL_STD,
    )
    
    if checkpoint is not None:
        load_checkpoint(sam, checkpoint)
    
    return sam


def load_checkpoint(model: Sam, checkpoint_path: str) -> None:
    """Load checkpoint with proper error handling."""
    try:
        state_dict = torch.load(checkpoint_path, map_location="cuda")
        model.load_state_dict(state_dict)
        print(f"Successfully loaded checkpoint: {checkpoint_path}")
    except FileNotFoundError:
        print(f"Warning: Checkpoint file not found: {checkpoint_path}")
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {str(e)}")


def create_sam_builder(variant: str):
    """Create a SAM builder function for a specific variant."""
    config = SAM_CONFIGS[variant]
    
    def builder(args) -> Sam:
        return build_sam(
            encoder_embed_dim=config["encoder_embed_dim"],
            encoder_depth=config["encoder_depth"],
            encoder_num_heads=config["encoder_num_heads"],
            encoder_global_attn_indexes=config["encoder_global_attn_indexes"],
            image_size=args.image_size,
            checkpoint=args.sam_checkpoint,
            pretrain_model=config["pretrain_model"]
        )
    
    return builder


# Generate builder functions dynamically to eliminate code duplication
build_sam_vit_h = create_sam_builder("vit_h")
build_sam_vit_l = create_sam_builder("vit_l")
build_sam_vit_b = create_sam_builder("vit_b")

# Model registry with clear naming
sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def get_sam_model(variant: str = "default", args = None) -> Sam:
    """
    Convenience function to get a SAM model by variant name.
    
    Args:
        variant: Model variant ("default", "vit_h", "vit_l", "vit_b")
        args: Arguments object containing image_size and sam_checkpoint
    
    Returns:
        Configured SAM model
        
    Raises:
        KeyError: If variant is not supported
    """
    if variant not in sam_model_registry:
        available = list(sam_model_registry.keys())
        raise KeyError(f"Unknown variant '{variant}'. Available: {available}")
    
    return sam_model_registry[variant](args)
