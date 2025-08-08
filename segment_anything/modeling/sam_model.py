import torch
from torch import nn
from torch.nn import functional as F
from typing import Any, Dict, List, Tuple, Optional


class Sam(nn.Module):
    """SAM predicts object masks from an image and input prompts."""
    
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
        self,
        image_encoder: nn.Module,
        prompt_encoder: nn.Module,
        mask_decoder: nn.Module,
        text_model: Optional[nn.Module] = None,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        Initialize SAM model.

        Args:
            image_encoder: The backbone used to encode the image into image embeddings
            prompt_encoder: Encodes various types of input prompts
            mask_decoder: Predicts masks from the image embeddings and encoded prompts
            text_model: Optional text model for text-based prompts
            pixel_mean: Mean values for normalizing pixels in the input image
            pixel_std: Std values for normalizing pixels in the input image
        """
        super().__init__()

        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.text_model = text_model
        
        # Only create text output projection if text model is provided
        if self.text_model is not None:
            self.text_out_dim = nn.Linear(512, 768)

        self.register_buffer(
            "pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), persistent=False
        )


    @property
    def device(self) -> torch.device:
        return self.pixel_mean.device


    def forward(
        self, 
        batched_input: Dict[str, Any], 
        multimask_output: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the SAM model.
        
        Args:
            batched_input: Dictionary containing input data
            multimask_output: Whether to output multiple masks
            
        Returns:
            Dictionary containing masks, IoU predictions, and low-res logits
        """
        input_images = batched_input["image"]
        image_embeddings = self.image_encoder(input_images)

        # Extract point coordinates and labels if available
        points = self._extract_points(batched_input)

        # Encode prompts
        # sparse_embeddings: [2, 3, 256],  dense_embeddings: [2, 256, 64, 64]
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes"),
            masks=batched_input.get("mask_inputs"),
        )

        # Decode masks
        low_res_masks, iou_predictions = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(), # [1, 256, 64, 64]
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # Post-process masks to original resolution
        masks = self.postprocess_masks(
            low_res_masks,
            input_size=input_images.shape[-2:],
            original_size=batched_input["original_size"],
        )

        return {
            "masks": masks,
            "iou_predictions": iou_predictions,
            "low_res_logits": low_res_masks,
        }


    def _extract_points(self, batched_input: Dict[str, Any]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Extract point coordinates and labels from batched input."""
        point_coords = batched_input.get("point_coords")
        if point_coords is not None:
            point_labels = batched_input.get("point_labels")
            if point_labels is None:
                raise ValueError("point_labels must be provided when point_coords is given")
            return (point_coords, point_labels)
        return None


    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, int],
        original_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Post-process masks to match original image size.
        
        Args:
            masks: Low-resolution masks from decoder
            input_size: Size of the input image to the model
            original_size: Original size of the image before preprocessing
            
        Returns:
            Masks resized to original image dimensions
        """
        # First interpolate to the model's standard size
        img_size = self.image_encoder.img_size
        masks = F.interpolate(
            masks,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

        # Crop to input size
        masks = masks[..., : input_size[0], : input_size[1]]
        
        # Resize to original size
        masks = F.interpolate(
            masks, size=original_size, mode="bilinear", align_corners=False
        )
        return masks


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std
        
        # Pad to square
        h, w = x.shape[-2:]
        img_size = self.image_encoder.img_size
        padh = img_size - h
        padw = img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x