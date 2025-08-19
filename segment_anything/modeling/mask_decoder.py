# Adapted from: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/mask_decoder.py
import torch
from torch import nn
from torch.nn import functional as F
from typing import Tuple, Type, Optional, Dict

# Import custom modules
from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    """Mask decoder using transformer architecture for mask prediction.
    
    Predicts masks given image and prompt embeddings using a transformer-based
    architecture with support for multiple mask outputs and quality prediction.
    """
    
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: Type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        semantic_out_dim: int = 512,
    ) -> None:
        """Initialize MaskDecoder.
        
        Args:
            transformer_dim: Channel dimension of the transformer
            transformer: Transformer module for mask prediction
            num_multimask_outputs: Number of masks for disambiguation
            activation: Activation function type for upscaling
            iou_head_depth: Depth of MLP for mask quality prediction
            iou_head_hidden_dim: Hidden dimension of quality prediction MLP
            semantic_out_dim: Output dimension for semantic predictions
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        
        # Initialize embeddings
        self._init_embeddings()
        
        # Initialize output networks
        self._init_output_networks(activation)
        
        # Initialize prediction heads
        self._init_prediction_heads(iou_head_hidden_dim, iou_head_depth, semantic_out_dim)
    

    def _init_embeddings(self) -> None:
        """Initialize embedding layers."""
        self.iou_token = nn.Embedding(1, self.transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, self.transformer_dim)
        self.semantic_tokens = nn.Embedding(self.num_mask_tokens, self.transformer_dim)
    

    def _init_output_networks(self, activation: Type[nn.Module]) -> None:
        """Initialize output upscaling and hypernetworks."""
        dim = self.transformer_dim
        
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(dim, dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(dim // 4),
            activation(),
            nn.ConvTranspose2d(dim // 4, dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        
        self.output_hypernetworks_mlps = nn.ModuleList([
            MLP(dim, dim, dim // 8, 3) 
            for _ in range(self.num_mask_tokens)
        ])
        
        self.text_output = nn.Linear(dim, dim // 8)
    

    def _init_prediction_heads(
        self, 
        hidden_dim: int, 
        depth: int, 
        semantic_dim: int
    ) -> None:
        """Initialize prediction heads for IoU and semantic outputs."""
        self.iou_prediction_head = MLP(
            self.transformer_dim, hidden_dim, self.num_mask_tokens, depth
        )
        self.semantic_prediction_head = MLP(
            self.transformer_dim, hidden_dim, semantic_dim, depth
        )
    

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        text_prompt_embeddings: Optional[torch.Tensor],
        multimask_output: bool,
    ) -> Dict[str, torch.Tensor]:
        """Predict masks given image and prompt embeddings.
        
        Args:
            image_embeddings: Embeddings from image encoder [B, 256, 64, 64]
            image_pe: Positional encoding [1, 256, 64, 64]
            sparse_prompt_embeddings: Point and box embeddings [B, 3, 256]
            dense_prompt_embeddings: Mask input embeddings [B, 256, 64, 64]
            text_prompt_embeddings: Optional text embeddings [B, 256]
            multimask_output: Return multiple masks if True, single mask if False
            
        Returns:
            Dictionary containing:
                - low_res_masks: Predicted masks
                - iou_pred: Mask quality predictions
                - semantic_pred: Semantic predictions
        """
        masks, iou_pred, semantic_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            text_prompt_embeddings=text_prompt_embeddings,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )
        
        # Select appropriate mask slice based on output mode
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        
        return {
            'low_res_masks': masks[:, mask_slice, :, :],
            'iou_pred': iou_pred[:, mask_slice],
            'semantic_pred': semantic_pred[:, mask_slice, :]
        }
    

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        text_prompt_embeddings: Optional[torch.Tensor],
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Core mask prediction logic.
        
        Returns:
            Tuple of (masks, iou_predictions, semantic_predictions)
        """
        # Prepare tokens
        tokens = self._prepare_tokens(sparse_prompt_embeddings)
        
        # Prepare source features
        src = image_embeddings + dense_prompt_embeddings
        src_shape = src.shape
        pos_src = image_pe.repeat(tokens.shape[0], 1, 1, 1)
        
        # Run transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # Extract different token outputs
        iou_token_out, mask_tokens_out, semantic_tokens_out = self._extract_tokens(
            hs, sparse_prompt_embeddings.shape[1]
        )
        
        # Generate masks
        masks = self._generate_masks(src, src_shape, mask_tokens_out, text_prompt_embeddings)
        
        # Generate predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        semantic_pred = self.semantic_prediction_head(semantic_tokens_out)
        
        return masks, iou_pred, semantic_pred
    

    def _prepare_tokens(self, sparse_prompt_embeddings: torch.Tensor) -> torch.Tensor:
        """Prepare and concatenate all tokens for transformer input."""
        output_tokens = torch.cat([
            self.iou_token.weight,
            self.mask_tokens.weight,
            self.semantic_tokens.weight
        ], dim=0)
        
        batch_size = sparse_prompt_embeddings.size(0)
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        
        return torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)
    

    def _extract_tokens(
        self, 
        hs: torch.Tensor, 
        sparse_dim: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract different token types from transformer output."""
        num_tokens = hs.shape[1]
        
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]
        semantic_tokens_out = hs[:, (1 + self.num_mask_tokens):(num_tokens - sparse_dim), :]
        
        return iou_token_out, mask_tokens_out, semantic_tokens_out
    

    def _generate_masks(
        self,
        src: torch.Tensor,
        original_shape: torch.Size,
        mask_tokens_out: torch.Tensor,
        text_prompt_embeddings: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Generate masks from transformer outputs."""
        b, c, h, w = original_shape
        
        # Reshape and upscale source features
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        
        # Apply hypernetworks
        hyper_in = torch.stack([
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            for i in range(self.num_mask_tokens)
        ], dim=1)
        
        # Generate base masks
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        
        # Add text conditioning if available
        if text_prompt_embeddings is not None:
            masks = self._add_text_conditioning(
                masks, upscaled_embedding, text_prompt_embeddings
            )
        
        return masks
    

    def _add_text_conditioning(
        self,
        masks: torch.Tensor,
        upscaled_embedding: torch.Tensor,
        text_prompt_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Add text-based conditioning to masks."""
        b, c, h, w = upscaled_embedding.shape
        
        text_features = self.text_output(text_prompt_embeddings.unsqueeze(1))
        text_mask = (text_features @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)
        text_mask = text_mask.expand(-1, masks.shape[1], -1, -1)
        
        return masks + text_mask


class MLP(nn.Module):
    """
    Simple multi-layer perceptron (also called FFN).
    
    Adapted from:
    https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py

    Simple feedforward network with ReLU activations between layers.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        """Initialize MLP.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output feature dimension
            num_layers: Number of layers
            sigmoid_output: Apply sigmoid to output if True
        """
        super().__init__()
        self.num_layers = num_layers
        self.sigmoid_output = sigmoid_output
        
        # Build layer dimensions
        layer_dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]
        
        # Create layers
        self.layers = nn.ModuleList([
            nn.Linear(layer_dims[i], layer_dims[i + 1])
            for i in range(num_layers)
        ])
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MLP."""
        for i, layer in enumerate(self.layers):
            x = layer(x)
            
            # Apply ReLU to all but last layer
            if i < self.num_layers - 1:
                x = F.relu(x)
        
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        
        return x

