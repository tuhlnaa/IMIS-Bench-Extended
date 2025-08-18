# Adapted from: https://github.com/facebookresearch/segment-anything/blob/main/segment_anything/predictor.py

import numpy as np
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai import transforms
from torchvision.transforms import v2
from torchvision.transforms import InterpolationMode
from typing import Optional, Tuple, List, Dict, Any


class ImagePreprocessor:
    """Handles image format conversion and preprocessing"""
    
    SUPPORTED_IMAGE_FORMATS = frozenset(["RGB", "BGR"])
    
    def __init__(self, image_size: Tuple[int, int], model_image_format: str = "RGB"):
        self.image_size = image_size
        self.model_image_format = model_image_format
    

    def convert_image_format(self, image: np.ndarray, image_format: str) -> np.ndarray:
        """Convert image to model's expected format."""
        if image_format not in self.SUPPORTED_IMAGE_FORMATS:
            raise ValueError(
                f"image_format must be in {self.SUPPORTED_IMAGE_FORMATS}, got {image_format}"
            )
        
        if image_format != self.model_image_format:
            return image[..., ::-1]  # BGR <-> RGB conversion
        return image
    

    def get_transforms(self) -> v2.Compose:
        """Get image preprocessing transforms."""
        return v2.Compose([
            v2.ToTensor(),
            v2.Resize(self.image_size, interpolation=InterpolationMode.NEAREST),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    
    def preprocess_image(self, image: np.ndarray, image_format: str = "RGB") -> torch.Tensor:
        """Preprocess image for model input.
        
        Args:
            image: Input image as numpy array (H, W, C)
            image_format: Image color format ('RGB' or 'BGR')
            
        Returns:
            preprocessed_tensor
        """
        
        # Convert image format if necessary
        input_image = self.convert_image_format(image, image_format)
        
        # Transform image
        transform = self.get_transforms()
        input_tensor = transform(input_image).unsqueeze(0)
        
        return input_tensor


class PromptProcessor:
    """Converts numpy prompts to tensors and handles transformations"""
    
    def __init__(self, device: torch.device):
        self.device = device
    

    def transform_coords(
        self, 
        coords: np.ndarray, 
        original_size: Tuple[int, int], 
        new_size: Tuple[int, int]
    ) -> np.ndarray:
        """Transform coordinates from original to new size."""
        old_h, old_w = original_size
        new_h, new_w = new_size
        
        coords = coords.astype(np.float32).copy()
        coords[..., 0] *= new_w / old_w
        coords[..., 1] *= new_h / old_h
        
        return coords
    

    def transform_boxes(
        self, 
        boxes: np.ndarray, 
        original_size: Tuple[int, int], 
        new_size: Tuple[int, int]
    ) -> np.ndarray:
        """Transform bounding boxes from original to new size."""
        boxes_reshaped = boxes.reshape(-1, 2, 2)
        boxes_transformed = self.transform_coords(boxes_reshaped, original_size, new_size)
        return boxes_transformed.reshape(-1, 4)
    

    def prepare_prompt_tensors(
        self,
        point_coords: Optional[np.ndarray],
        point_labels: Optional[np.ndarray],
        box: Optional[np.ndarray],
        text: Optional[List[str]],
        mask_input: Optional[np.ndarray],
        original_size: Tuple[int, int],
        image_size: Tuple[int, int],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Convert numpy prompts to torch tensors."""
        tensors = {}
        
        if point_coords is not None:
            if point_labels is None:
                raise AssertionError("point_labels must be supplied if point_coords is supplied.")
            
            coords = self.transform_coords(point_coords, original_size, image_size)

            tensors['point_coords'] = torch.as_tensor(coords, dtype=torch.float32, device=self.device).unsqueeze(0)
            tensors['point_labels'] = torch.as_tensor(point_labels, dtype=torch.int32, device=self.device).unsqueeze(0)
        else:
            tensors['point_coords'] = None
            tensors['point_labels'] = None
        
        if box is not None:
            box = self.transform_boxes(box, original_size, image_size)
            tensors['boxes'] = torch.as_tensor(box, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            tensors['boxes'] = None
        
        if mask_input is not None:
            tensors['mask_input'] = torch.as_tensor(mask_input, dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            tensors['mask_input'] = None
        
        tensors['text'] = text  # Keep as list, will be tokenized later
        
        return tensors
    

    def build_prompt_dict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        text: Optional[List[str]],
        mask_input: Optional[torch.Tensor],
        text_tokenizer: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Build prompt dictionary for model input."""
        prompt = {}

        if text is not None and text_tokenizer is not None:
            prompt['text_inputs'] = text_tokenizer(text).to(self.device)
        
        if mask_input is not None:
            prompt['mask_inputs'] = mask_input
        
        if point_coords is not None:
            prompt['point_coords'] = point_coords
            prompt['point_labels'] = point_labels
        
        return prompt
    

class ClassificationHandler:
    """Manages semantic predictions and class mapping"""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.idx_to_class = self._setup_category_mapping()
    
    
    def _setup_category_mapping(self) -> Optional[Dict[int, str]]:
        """Setup category mapping if model supports it."""
        if hasattr(self.model, 'category_weights') and self.model.category_weights is not None:
            if hasattr(self.model, 'text_processor'):
                return self.model.text_processor.index_to_category
        return None
    

    def get_class_prediction(self, semantic_pred: Optional[torch.Tensor]) -> str:
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
        if self.model.device != 'cpu':
            category_idx = category_idx.cpu()
        
        return self.idx_to_class[int(category_idx)]


class IMISPredictor:
    """Orchestrates the above components for end-to-end prediction"""
    
    def __init__(self, config: OmegaConf, imis_model: nn.Module, encode_image, decode_masks):
        """Initialize the predictor with a SAM model.
        
        Args:
            sam_model: Pre-trained SAM model instance
        """
        self.config = config
        self.model = imis_model
        self.encode_image = encode_image
        self.decode_masks = decode_masks
        self.device = imis_model.device
        
        # Initialize component classes
        self.image_preprocessor = ImagePreprocessor(
            image_size=imis_model.image_size,
            model_image_format=getattr(imis_model, 'image_format', 'RGB')
        )
        self.prompt_processor = PromptProcessor(self.device)
        self.classification_handler = ClassificationHandler(imis_model)
        
        # Initialize state
        self._reset_state()
    

    def _reset_state(self) -> None:
        """Reset internal state variables."""
        self.is_image_set = False
        self.features = None
        self.original_size = None
        self.image_size = (self.config.model.image_size, self.config.model.image_size)
    

    def get_image_features(self, image: np.ndarray, image_format: str = "RGB") -> None:
        """Set and preprocess the input image for prediction.
        
        Args:
            image: Input image as numpy array (H, W, C)
            image_format: Image color format ('RGB' or 'BGR')
        """
        self.original_size = image.shape[:2]

        # Preprocess image
        input_tensor = self.image_preprocessor.preprocess_image(image, image_format)
        
        # Encode image
        self.features = self.encode_image(input_tensor.to(self.device))
        self.is_image_set = True
    

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        bounding_box: Optional[np.ndarray] = None,
        text_prompt: Optional[List[str]] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate mask predictions for the current image with given prompts.
        
        Args:
            point_coords: Point coordinates (N, 2)
            point_labels: Point labels (N,) - 1 for positive, 0 for negative
            bounding_box: Bounding box coordinates (4,) or (N, 4)
            text_prompt: Text prompts for segmentation
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
        prompt_tensors = self.prompt_processor.prepare_prompt_tensors(
            point_coords, point_labels, bounding_box, text_prompt, mask_input,
            self.original_size, self.image_size
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
            raise RuntimeError("An image must be set with .get_image_features(...) before mask prediction.")
    

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
        """Generate mask predictions using torch tensors."""
        # Build prompt dictionary
        prompt = self.prompt_processor.build_prompt_dict(
            point_coords, point_labels, text, mask_input,
            getattr(self.model, 'text_tokenizer', None)
        )
        
        # Process boxes if provided
        if boxes is not None:
            masks, low_res_masks, class_list = self._process_box_prompts(boxes, prompt)
        else:
            masks, low_res_masks, class_list = self._process_single_prompt(prompt)
        
        # Post-process masks
        if not return_logits:
            masks = torch.sigmoid(masks)
            masks = (masks > 0.5).float()

        return masks, low_res_masks, class_list
    

    def _process_box_prompts(
        self, boxes: torch.Tensor, prompt: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Process multiple box prompts."""
        masks_list = []
        class_list = []
        
        for i in range(boxes.shape[1]):
            prompt['bboxes'] = boxes[:, i:i+1, ...]
            outputs = self.decode_masks(self.features, prompt)
            
            # Process masks
            masks = F.interpolate(outputs['masks'], self.original_size, mode="bilinear", align_corners=False)
            masks_list.append(masks)
            
            # Process class predictions
            class_pred = self.classification_handler.get_class_prediction(outputs.get('semantic_pred'))
            class_list.append(class_pred)
        
        masks = torch.cat(masks_list, dim=0)
        return masks, outputs['low_res_masks'], class_list
    

    def _process_single_prompt(
        self, prompt: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Process a single prompt."""
        outputs = self.decode_masks(self.features, prompt)
        
        masks = F.interpolate(outputs['masks'], self.original_size, mode="bilinear", align_corners=False)
        class_pred = self.classification_handler.get_class_prediction(outputs.get('semantic_pred'))
        
        return masks, outputs['low_res_masks'], [class_pred]
    

    def reset_image(self) -> None:
        """Reset the currently set image and clear cached features."""
        self._reset_state()