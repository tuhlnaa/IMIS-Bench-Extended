
# Adapted from: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/image_encoder.py
import timm
import torch
import torch.nn as nn

from timm.layers import resample_abs_pos_embed_nhwc

# Import custom modules
from segment_anything.modeling.common import LayerNorm2d


class ViT(nn.Module):
    """Vision Transformer with SAM encoder backbone and feature neck.
    
    Args:
        encoder_embed_dim: Embedding dimension of the encoder
        pretrain_model: Name of the pretrained model from timm
        out_chans: Number of output channels from the neck
        depth: Number of transformer blocks to use
        pretrained: Whether to load pretrained weights
        freeze_encoder: Whether to freeze encoder parameters
    """
    
    def __init__(
            self,
            encoder_embed_dim: int = 768,
            pretrain_model: str = 'samvit_base_patch16',
            out_chans: int = 256,
            depth: int = 12,
            pretrained: bool = True,
            freeze_encoder: bool = True,
        ) -> None:
        super().__init__()

        # Store configuration
        self.encoder_embed_dim = encoder_embed_dim
        self.depth = depth
        self.pretrain_model = pretrain_model

        # Initialize encoder
        self.sam_encoder = timm.create_model(
            self.pretrain_model, 
            pretrained=pretrained, 
            num_classes=0
        )

        # Freeze encoder if requested
        if freeze_encoder:
            self._freeze_encoder()

        # Build neck module
        self.neck = self._build_neck(encoder_embed_dim, out_chans)


    def _freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for param in self.sam_encoder.parameters():
            param.requires_grad = False


    def _build_neck(self, in_chans: int, out_chans: int) -> nn.Sequential:
        """Build the neck module for feature adaptation."""
        return nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=False),
            LayerNorm2d(out_chans),
            #nn.GroupNorm(1, out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(out_chans),
            #nn.GroupNorm(1, out_chans)
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.sam_encoder.patch_embed(x)

        # Add positional embeddings if available
        if self.sam_encoder.pos_embed is not None:
            x = x + resample_abs_pos_embed_nhwc(
                self.sam_encoder.pos_embed, 
                x.shape[1:3]
            )
        
        # Apply dropout layers
        x = self.sam_encoder.pos_drop(x)
        x = self.sam_encoder.patch_drop(x)
        
        # Pre-normalization
        x = self.sam_encoder.norm_pre(x)

        # Process through transformer blocks
        for i in range(self.depth):
            x = self.sam_encoder.blocks[i](x)
        
        # Convert from NHWC to NCHW format and apply neck
        x = x.permute(0, 3, 1, 2)
        x = self.neck(x)
        return x


    def get_num_params(self, trainable_only: bool = True) -> int:
        """Get the number of parameters in the model."""
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


def main():
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


if __name__ == '__main__':
    main()