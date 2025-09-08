# Adapted from: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/modeling/prompt_encoder.py
import torch
import numpy as np

from torch import nn
from typing import Optional, Tuple, Type

from .common import LayerNorm2d


class PromptEncoder(nn.Module):
    """Encodes various prompts for input to SAM's mask decoder."""
    
    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Initialize the prompt encoder.

        Args:
            embed_dim: The prompts' embedding dimension
            image_embedding_size: The spatial size of the image embedding, as (H, W)
            input_image_size: The padded size of the image as input to the image encoder, as (H, W)
            mask_in_chans: The number of hidden channels used for encoding input masks
            activation: The activation to use when encoding input masks
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        # Initialize point embeddings
        self.num_point_embeddings = 4  # pos/neg point + 2 box corners
        self.point_embeddings = nn.ModuleList([
            nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)
        ])
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        # Initialize mask encoding layers
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = self._build_mask_downscaler(mask_in_chans, embed_dim, activation)
        self.no_mask_embed = nn.Embedding(1, embed_dim)


    def _build_mask_downscaler(
        self, 
        mask_in_chans: int, 
        embed_dim: int, 
        activation: Type[nn.Module]
    ) -> nn.Sequential:
        """Build the mask downscaling network."""
        return nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )


    def get_dense_pe(self) -> torch.Tensor:
        """
        Get positional encoding for dense point prompts.

        Returns:
            Positional encoding with shape 1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)


    def _embed_points(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
        pad: bool = True,
    ) -> torch.Tensor:
        """
        Embed point prompts.
        
        Args:
            points: Point coordinates (B, N, 2)
            labels: Point labels (B, N)
            pad: Whether to add padding point
            
        Returns:
            Point embeddings (B, N+1, embed_dim) if padded, else (B, N, embed_dim)
        """
        # Shift to center of pixel
        points = points + 0.5
        
        # Add padding if needed
        # if pad:
        #     padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
        #     padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
        #     points = torch.cat([points, padding_point], dim=1)
        #     labels = torch.cat([labels, padding_label], dim=1)

        padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
        padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
        points = torch.cat([points, padding_point], dim=1)
        labels = torch.cat([labels, padding_label], dim=1)

        # Get positional encoding
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        
        # Apply label-specific embeddings
        # Note: Using in-place operations for efficiency
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        
        return point_embedding


    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Embed box prompts.
        
        Args:
            boxes: Box coordinates (B, 4)
            
        Returns:
            Box embeddings (B, 2, embed_dim)
        """
        # Shift to center of pixel
        boxes = boxes + 0.5
        coords = boxes.reshape(-1, 2, 2)
        
        # Get corner embeddings
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        
        return corner_embedding


    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embed mask inputs."""
        return self.mask_downscaling(masks)


    def _get_batch_size(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text: Optional[torch.Tensor],
    ) -> int:
        """Get batch size from input prompts."""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        elif text is not None:
            return text.shape[0]
        else:
            return 1
        

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
        text: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed different types of prompts.

        Args:
            points: Point coordinates and labels to embed  # ([1, 1, 2], [1, 1]), float32, int32
            boxes: Boxes to embed  # [1, 1, 4]
            masks: Masks to embed  # [1, 1, 256, 256]
            text: Text embeddings to include  # [1, 768]

        Returns:
            Tuple of:
                - Sparse embeddings (BxNx(embed_dim))
                - Dense embeddings (Bx(embed_dim)x(embed_H)x(embed_W))
        """
        bs = self._get_batch_size(points, boxes, masks, text)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self.device)

        # Embed points
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)

        # Embed boxes
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)
        
        # Add text embeddings
        if text is not None:
            sparse_embeddings = torch.cat([sparse_embeddings, text.unsqueeze(1)], dim=1)

        # Generate dense embeddings
        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


    @property
    def device(self) -> torch.device:
        """Get the device of the model."""
        return self.point_embeddings[0].weight.device


class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies."""

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        """
        Initialize random position embedding.
        
        Args:
            num_pos_feats: Number of positional features
            scale: Scale factor for the random matrix
        """
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )
        

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Positionally encode points normalized to [0,1].
        
        Args:
            coords: Coordinates with shape d_1 x ... x d_n x 2
            
        Returns:
            Encoded coordinates with shape d_1 x ... x d_n x C
        """
        # Normalize coords to [-1, 1]
        coords = 2 * coords - 1
        
        # Apply Gaussian matrix transformation
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords

        # Return sin and cos encodings
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)


    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """
        Generate positional encoding for a grid of the specified size.
        
        Args:
            size: Grid size as (H, W)
            
        Returns:
            Positional encoding with shape C x H x W
        """
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        
        # Create normalized grid coordinates
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = (grid.cumsum(dim=0) - 0.5) / h
        x_embed = (grid.cumsum(dim=1) - 0.5) / w
        
        # Stack and encode
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W


    def forward_with_coords(
        self, 
        coords_input: torch.Tensor, 
        image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """
        Positionally encode points that are not normalized to [0,1].
        
        Args:
            coords_input: Input coordinates (B, N, 2)
            image_size: Image size as (H, W)
            
        Returns:
            Encoded coordinates (B, N, C)
        """
        # Normalize coordinates to [0, 1]
        coords = coords_input.clone()
        coords[:, :, 0] /= image_size[1]
        coords[:, :, 1] /= image_size[0]
        
        return self._pe_encoding(coords)