# Adapted from: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/build_sam.py
import torch

from torch import nn
from typing import Any, Dict, Optional
from transformers import CLIPTextConfig, CLIPTextModel
from omegaconf import OmegaConf

# Import custom modules
from .modeling import ViT, MaskDecoder, Sam, PromptEncoder, TwoWayTransformer
from src.models.text_encoder import StandaloneTextEncoder
# from src.models.sam_model import Sam

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
    config: OmegaConf,
    encoder_embed_dim: int,
    encoder_depth: int,
    encoder_num_heads: int,
    encoder_global_attn_indexes: list,
    pretrain_model: str,
    prompt_embed_dim: int = DEFAULT_PROMPT_EMBED_DIM,
    vit_patch_size: int = DEFAULT_VIT_PATCH_SIZE
) -> nn.Module:
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
    image_embedding_size = config.model.image_size // vit_patch_size

    image_encoder = ViT(
        encoder_embed_dim=encoder_embed_dim,
        pretrain_model=pretrain_model,
        out_chans=prompt_embed_dim,
        depth=encoder_depth,
        freeze_encoder=True,
        pretrained=False,
    )

    text_encoder = StandaloneTextEncoder(text_model=CLIPTextModel(CLIPTextConfig()))

    sam = Sam(
        image_encoder=image_encoder,
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(config.model.image_size, config.model.image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            transformer_dim=prompt_embed_dim,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            num_multimask_outputs=3,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        # text_model = CLIPTextModel(CLIPTextConfig()),
        pixel_mean=DEFAULT_PIXEL_MEAN,
        pixel_std=DEFAULT_PIXEL_STD,
    )
    
    if config.checkpoint_path is not None:
        state_dict = torch.load(config.checkpoint_path, weights_only=True)
        sam.load_state_dict(state_dict)

        vit_weights = torch.load(config.checkpoint_vit_path, weights_only=True)
        image_encoder.load_state_dict(vit_weights)

        clip_weights = torch.load(config.checkpoint_clip_path, weights_only=True)
        text_encoder.load_state_dict(clip_weights)

        print(f"Successfully loaded checkpoint: {config.checkpoint_path}")
    return sam, image_encoder, text_encoder


def create_sam_builder(variant: str):
    """Create a SAM builder function for a specific variant."""
    config = SAM_CONFIGS[variant]
    
    def builder(args) -> Sam:
        return build_sam(
            config=args,
            encoder_embed_dim=config["encoder_embed_dim"],
            encoder_depth=config["encoder_depth"],
            encoder_num_heads=config["encoder_num_heads"],
            encoder_global_attn_indexes=config["encoder_global_attn_indexes"],
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


def get_sam_model(variant: str = "default", config: OmegaConf = None) -> Sam:
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
    
    return sam_model_registry[variant](config)

