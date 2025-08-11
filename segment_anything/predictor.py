# Adapted from: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai import transforms
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from dataloaders.data_utils import Resize, PermuteTransform, Normalization


class IMISPredictor:
    """Image segmentation predictor using SAM-based model architecture.
    
    This predictor handles image encoding, prompt processing, and mask generation
    for segmentation tasks with support for point, box, text, and mask prompts.
    
    Attributes:
        model: The underlying SAM model
        device: Computing device (CPU/CUDA)
        features: Cached image features
        is_image_set: Whether an image has been set
        original_size: Original image dimensions
        image_size: Model input dimensions
    """
    
    # Class constants
    SUPPORTED_IMAGE_FORMATS = frozenset(["RGB", "BGR"])
    DEFAULT_SIGMOID_THRESHOLD = 0.5
    
    def __init__(self, sam_model: nn.Module):
        """Initialize the predictor with a SAM model.
        
        Args:
            sam_model: Pre-trained SAM model instance
        """
        self.model = sam_model
        self.device = sam_model.device
        
        # Initialize state
        self._reset_state()
        
        # Cache category mapping if available
        self._setup_category_mapping()
    

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.image_size = self.model.image_size
        self.input_h = None
        self.input_w = None
    

    def _setup_category_mapping(self) -> None:
        """Setup category mapping if model supports it."""
        if hasattr(self.model, 'category_weights') and self.model.category_weights is not None:
            self.idx_to_class = self.model.text_processor.index_to_category
        else:
            self.idx_to_class = None
    

    def set_image(self, image: np.ndarray, image_format: str = "RGB") -> None:
        """Set and preprocess the input image for prediction.
        
        Args:
            image: Input image as numpy array (H, W, C)
            image_format: Image color format ('RGB' or 'BGR')
        """
        if image_format not in self.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"image_format must be in {self.SUPPORTED_IMAGE_FORMATS}, got {image_format}"
            )
        
        # Convert image format if necessary
        input_image = self._convert_image_format(image, image_format)
        
        # Store original dimensions
        self.original_size = input_image.shape[:2]
        self.input_h, self.input_w = self.image_size
        
        # Transform and encode image
        input_tensor = self._preprocess_image(input_image)
        self.features = self.model.encode_image(input_tensor.to(self.device))
        self.is_image_set = True
    

    def _convert_image_format(self, image: np.ndarray, image_format: str) -> np.ndarray:
        """Convert image to model's expected format."""
        if image_format != self.model.image_format:
            return image[..., ::-1]  # BGR <-> RGB conversion
        return image
    

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Apply preprocessing transforms to image."""
        transform = self._get_transforms()

        transformed_image = transform(image).unsqueeze(0)

        # print(image.shape)
        # transformed_image = transform(image).unsqueeze(0)
        # print(transformed_image.shape)
        # print(transformed_image.dtype, transformed_image.max(), transformed_image.min())
        # # processed = transform({'image': image})['image'][None, :, :, :]
        # # print(processed.shape)
        # # print(processed.dtype, processed.max(), processed.min())
        # quit()

        if len(transformed_image.shape) != 4 or transformed_image.shape[1] != 3:
            raise AssertionError(
                f"Preprocessed image must be BCHW with 3 channels, got {transformed_image.shape}"
            )
        
        return transformed_image
    
    # def _get_transforms(self) -> transforms.Compose:
    #     """Get image preprocessing transforms."""
    #     return transforms.Compose([
    #         Resize(keys=["image"], target_size=self.image_size),
    #         PermuteTransform(keys=["image"], dims=(2, 0, 1)),
    #         transforms.ToTensord(keys=["image"]),
    #         Normalization(keys=["image"]),
    #     ])
    
    def _get_transforms(self) -> transforms.Compose:
        """Get image preprocessing transforms."""
        return v2.Compose([
        v2.ToTensor(),
        v2.Resize(self.image_size, interpolation=InterpolationMode.NEAREST),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        text: Optional[List[str]] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate mask predictions for the current image with given prompts.
        
        Args:
            point_coords: Point coordinates (N, 2)
            point_labels: Point labels (N,) - 1 for positive, 0 for negative
            box: Bounding box coordinates (4,) or (N, 4)
            text: Text prompts for segmentation
            mask_input: Previous mask prediction for refinement
            multimask_output: Whether to output multiple masks
            return_logits: Return raw logits instead of binary masks
            
        Returns:
            Tuple of (masks, low_res_masks, class_list)
            
        Raises:
            RuntimeError: If no image has been set
            AssertionError: If point_coords provided without point_labels
        """
        self._validate_image_set()
        
        # Convert prompts to tensors
        prompt_tensors = self._prepare_prompt_tensors(
            point_coords, point_labels, box, text, mask_input
        )
        
        # Generate predictions
        masks, low_res_masks, class_list = self.predict_torch(
            **prompt_tensors,
            multimask_output=multimask_output,
            return_logits=return_logits,
        )

        masks = masks.detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()

        return masks, low_res_masks, class_list
    

    def _validate_image_set(self) -> None:
        """Validate that an image has been set."""
        if not self.is_image_set:
            raise RuntimeError("An image must be set with .set_image(...) before mask prediction.")
    

    def _prepare_prompt_tensors(
        self,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
        text: Optional[List[str]],
        mask_input: Optional[np.ndarray],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Convert numpy prompts to torch tensors."""
        tensors = {}
        
        if point_coords is not None:
            if point_labels is None:
                raise AssertionError("point_labels must be supplied if point_coords is supplied.")
            
            coords = self._transform_coords(point_coords, self.original_size, self.image_size)
            tensors['point_coords'] = torch.as_tensor(coords, dtype=torch.float32, device=self.device)[None, :, :]
            tensors['point_labels'] = torch.as_tensor(point_labels, dtype=torch.int32, device=self.device)[None, :]

        else:
            tensors['point_coords'] = None
            tensors['point_labels'] = None
        
        if box is not None:
            box = self._transform_boxes(box, self.original_size, self.image_size)
            tensors['boxes'] = torch.as_tensor(box, dtype=torch.float32, device=self.device)[None, :]
        else:
            tensors['boxes'] = None
        
        if mask_input is not None:
            tensors['mask_input'] = torch.as_tensor(mask_input, dtype=torch.float32, device=self.device)[None, :, :, :]
        else:
            tensors['mask_input'] = None
        
        tensors['text'] = text  # Keep as list, will be tokenized in predict_torch
        
        return tensors
    

    @torch.no_grad()
    def predict_torch(
        self,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        boxes: Optional[torch.Tensor] = None,
        text: Optional[List[str]] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Generate mask predictions using torch tensors.
        
        Args:
            point_coords: Point coordinates tensor
            point_labels: Point labels tensor
            boxes: Bounding boxes tensor
            text: Text prompts
            mask_input: Previous mask tensor
            multimask_output: Whether to output multiple masks
            return_logits: Return raw logits instead of binary masks
            
        Returns:
            Tuple of (masks, low_res_masks, class_list)
        """
        self._validate_image_set()
        
        # Build prompt dictionary
        prompt = self._build_prompt_dict(point_coords, point_labels, text, mask_input)
        
        # Process boxes if provided
        if boxes is not None:
            masks, low_res_masks, class_list = self._process_box_prompts(boxes, prompt)
        else:
            masks, low_res_masks, class_list = self._process_single_prompt(prompt)
        
        # Post-process masks
        if not return_logits:
            masks = torch.sigmoid(masks)
            masks = (masks > self.DEFAULT_SIGMOID_THRESHOLD).float()

        return masks, low_res_masks, class_list
    

    def _build_prompt_dict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        text: Optional[List[str]],
        mask_input: Optional[torch.Tensor],
    ) -> Dict[str, Any]:
        """Build prompt dictionary for model input."""
        prompt = {}
        
        if text is not None:
            prompt['text_inputs'] = self.model.text_tokenizer(text).to(self.device)
        
        if mask_input is not None:
            prompt['mask_inputs'] = mask_input
        
        if point_coords is not None:
            prompt['point_coords'] = point_coords
            prompt['point_labels'] = point_labels
        
        return prompt
    

    def _process_box_prompts(
        self, boxes: torch.Tensor, prompt: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Process multiple box prompts."""
        masks_list = []
        class_list = []
        
        for i in range(boxes.shape[1]):
            prompt['bboxes'] = boxes[:, i:i+1, ...]
            outputs = self.model.decode_masks(self.features, prompt)
            
            # Process masks
            masks = self._postprocess_masks(outputs['masks'], self.original_size)
            masks_list.append(masks)
            
            # Process class predictions
            class_pred = self._get_class_prediction(outputs.get('semantic_pred'))
            class_list.append(class_pred)
        
        masks = torch.cat(masks_list, dim=0)
        return masks, outputs['low_res_masks'], class_list
    

    def _process_single_prompt(
        self, prompt: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Process a single prompt."""
        outputs = self.model.decode_masks(self.features, prompt)
        
        masks = self._postprocess_masks(outputs['masks'], self.original_size)
        class_pred = self._get_class_prediction(outputs.get('semantic_pred'))
        
        return masks, outputs['low_res_masks'], [class_pred]
    

    def _get_class_prediction(self, semantic_pred: Optional[torch.Tensor]) -> str:
        """Get class prediction from semantic predictions."""
        if semantic_pred is None or self.idx_to_class is None:
            return 'Category recognition not available'
        
        if not hasattr(self.model, 'text_processor'):
            return 'Text processor not available'
        
        # Normalize and compute logits
        logits = F.normalize(semantic_pred, dim=-1) @ self.model.text_processor.src_weights
        probs = F.softmax(logits, dim=-1)
        
        # Get predicted class index
        category_idx = torch.argmax(probs, dim=-1).squeeze()
        if self.device != 'cpu':
            category_idx = category_idx.cpu()
        
        return self.idx_to_class[int(category_idx)]
    

    def _postprocess_masks(
        self, masks: torch.Tensor, target_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Resize masks to target size."""
        return F.interpolate(masks, target_size, mode="bilinear", align_corners=False)
    

    def _apply_sigmoid_threshold(self, masks: torch.Tensor) -> torch.Tensor:
        """Apply sigmoid and threshold to get binary masks."""
        masks = torch.sigmoid(masks)
        return (masks > self.DEFAULT_SIGMOID_THRESHOLD).float()
    

    def _transform_coords(
        self, coords: np.ndarray, original_size: Tuple[int, int], new_size: Tuple[int, int]
    ) -> np.ndarray:
        """Transform coordinates from original to new size."""
        old_h, old_w = original_size
        new_h, new_w = new_size
        
        coords = coords.astype(np.float32).copy()
        coords[..., 0] *= new_w / old_w
        coords[..., 1] *= new_h / old_h
        
        return coords
    

    def _transform_boxes(
        self, boxes: np.ndarray, original_size: Tuple[int, int], new_size: Tuple[int, int]
    ) -> np.ndarray:
        """Transform bounding boxes from original to new size."""
        boxes_reshaped = boxes.reshape(-1, 2, 2)
        boxes_transformed = self._transform_coords(boxes_reshaped, original_size, new_size)
        return boxes_transformed.reshape(-1, 4)
    

    def get_image_embedding(self) -> torch.Tensor:
        """Get the image embeddings for the currently set image.
        
        Returns:
            Image embeddings tensor with shape (1, C, H, W)
            
        Raises:
            RuntimeError: If no image has been set
        """
        self._validate_image_set()
        
        if self.features is None:
            raise RuntimeError("Features must exist if an image has been set.")
        
        return self.features
    

    def reset_image(self) -> None:
        """Reset the currently set image and clear cached features."""
        self._reset_state()
