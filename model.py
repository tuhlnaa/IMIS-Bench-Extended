import re
import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from dataclasses import dataclass
from transformers import AutoTokenizer
from typing import Dict, Optional, Tuple, List, Any
from omegaconf import OmegaConf

# Import custom modules
from dataloaders.data_utils import get_points_from_mask, get_bboxes_from_mask
from src.models.text_encoder import TextProcessorV2


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    point_num_choices: List[int] = None
    template: str = 'A segmentation area of a {}.'
    
    def __post_init__(self):
        if self.point_num_choices is None:
            self.point_num_choices = [1, 3, 4, 7]


class PromptProcessor:
    """Handles all prompt-related processing logic."""
    
    def __init__(self, device: torch.device, test_mode: bool = False):
        self.device = device
        self.test_mode = test_mode
        self.config = PromptConfig()
    
    def process_mask_prompt(self, low_res_masks: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process mask prompts."""
        low_res_masks_logits = low_res_masks.detach().clone()
        return {'mask_inputs': low_res_masks_logits.to(self.device)}
    
    def process_bboxes_prompt(self, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Process bounding box prompts."""
        bs = labels.shape[0]
        bs_bboxes = [get_bboxes_from_mask(labels[idx]) for idx in range(bs)]
        return {'bboxes': torch.stack(bs_bboxes, dim=0).to(self.device)}
    
    def process_points_prompt(
        self, 
        labels: torch.Tensor, 
        pred_masks: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Process point prompts with interaction-based sampling."""
        bs = labels.shape[0]
        point_num = 1 if self.test_mode else random.choice(self.config.point_num_choices)
        
        # Prepare masks
        if pred_masks is not None:
            pred_masks = (torch.sigmoid(pred_masks) > 0.5).bool().squeeze(1)
        
        labels = labels.bool().squeeze(1)
        
        # Calculate error area for interaction-based sampling
        error_area = pred_masks ^ labels if pred_masks is not None else None
        
        # Initialize tensors
        bs_point_coords = torch.empty((bs, point_num, 2), dtype=torch.long, device=labels.device)
        bs_point_labels = torch.empty((bs, point_num), dtype=torch.long, device=labels.device)
        
        for idx in range(bs):
            if pred_masks is None:
                point_coords, point_labels = get_points_from_mask(labels[idx], get_point=1)
            else:
                point_coords, point_labels = self._get_interaction_points(
                    error_area[idx], pred_masks[idx], labels[idx], point_num
                )
            
            bs_point_coords[idx] = torch.as_tensor(point_coords, device=labels.device)
            bs_point_labels[idx] = torch.as_tensor(point_labels, device=labels.device)
        
        return {
            'point_coords': bs_point_coords,
            'point_labels': bs_point_labels
        }
    
    @staticmethod
    def _get_interaction_points(
        error: torch.Tensor, 
        pred: torch.Tensor, 
        gt: torch.Tensor, 
        num_points: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get interaction points based on prediction errors."""
        pred_np, gt_np = pred.cpu().numpy(), gt.cpu().numpy()
        error_np = error.cpu().numpy()
        
        # Find error indices
        error_indices = np.argwhere(error_np == 1)
        
        if error_indices.shape[0] > 0:
            # Sample from error locations
            selected_indices = error_indices[
                np.random.choice(error_indices.shape[0], num_points, replace=True)
            ]
        else:
            # Random sampling if no errors
            selected_indices = np.random.randint(0, 256, size=(num_points, 2))
        
        selected_indices = selected_indices.reshape(-1, 2)
        
        # Generate points and labels
        points, labels = [], []
        for x, y in selected_indices:
            if pred_np[x, y] == 0 and gt_np[x, y] == 1:
                label = 1  # False negative
            elif pred_np[x, y] == 1 and gt_np[x, y] == 0:
                label = 0  # False positive
            else:
                label = -1  # Correct prediction
            
            points.append((y, x))  # Note: (y, x) format for consistency
            labels.append(label)
        
        return np.array(points), np.array(labels)


class IMISNet(nn.Module):
    """Interactive Medical Image Segmentation Network.
    
    This network combines SAM (Segment Anything Model) with text prompts
    for medical image segmentation tasks.
    """
    
    def __init__(
        self,
        config: OmegaConf,
        sam: nn.Module,
        text_model: nn.Module,
        test_mode: bool = False,
        multimask_output: bool = True,
        category_weights: Optional[str] = None,
        select_mask_num: Optional[int] = None
    ):
        super().__init__()
        
        # Core SAM components
        self.device = sam.device
        self.image_encoder = sam.image_encoder
        self.mask_decoder = sam.mask_decoder
        self.prompt_encoder = sam.prompt_encoder
        self.text_model = text_model
        
        # Configuration
        self.category_weights = category_weights
        self.test_mode = test_mode
        self.multimask_output = multimask_output
        self.select_mask_num = select_mask_num
        self.image_size = (config.model.image_size, config.model.image_size)
        
        # Initialize processors
        self.prompt_processor = PromptProcessor(self.device, test_mode)
        self.text_processor = TextProcessorV2(self.device, text_encoder=text_model, category_weights_path=self.category_weights)


    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode input image to embeddings."""
        batch_size = image.shape[0]
        image_embedding = self.image_encoder(image)
        
        assert len(image_embedding.shape) == 4, \
            f'Expected shape (B, C, H, W), got {image_embedding.shape}'
        
        if self.test_mode:
            return image_embedding
        
        # Repeat embeddings for multiple mask selection during training
        image_embedding = image_embedding.detach().clone()
        repeated_embeddings = []
        
        for bs_idx in range(batch_size):
            embed_repeated = image_embedding[bs_idx].repeat(self.select_mask_num, 1, 1, 1)
            repeated_embeddings.append(embed_repeated)
        
        return torch.cat(repeated_embeddings, dim=0)
    

    def decode_masks(
        self, 
        image_embedding: torch.Tensor, 
        prompt: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Decode masks from image embeddings and prompts."""

        # Prepare prompts
        points = None
        if prompt.get("point_coords") is not None:
            points = (prompt["point_coords"], prompt["point_labels"])
        
        # Encode prompts
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=points,
            boxes=prompt.get("bboxes"),
            masks=prompt.get("mask_inputs"),
            text=prompt.get("text_inputs")
        )

        # Decode masks
        outputs = self.mask_decoder(
            image_embeddings=image_embedding,
            image_pe=self.prompt_encoder.get_dense_pe(),
            text_prompt_embeddings=prompt.get("text_inputs"),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output
        )
        
        # Select best mask if multi-mask output
        if self.multimask_output:
            low_res_masks, iou_pred, semantic_pred = self._select_best_mask(outputs)
        else:
            low_res_masks = outputs['low_res_masks']
            iou_pred = outputs['iou_pred']
            semantic_pred = outputs['semantic_pred']
        
        # Upsample masks to original size
        masks = F.interpolate(low_res_masks, size=self.image_size, mode='bilinear', align_corners=False)
        
        return {
            'masks': masks.float(),
            'low_res_masks': low_res_masks,
            'iou_pred': iou_pred,
            'semantic_pred': semantic_pred
        }
    

    def _select_best_mask(
        self, 
        outputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select the best mask based on IoU predictions."""
        low_res_masks = outputs['low_res_masks']
        iou_pred = outputs['iou_pred']
        semantic_pred = outputs['semantic_pred']
        
        # Find best mask index
        max_values, max_indices = torch.max(iou_pred, dim=1, keepdim=True)
        
        # Gather corresponding masks and predictions
        mask_shape = (low_res_masks.shape[2], low_res_masks.shape[3])
        low_mask_indices = max_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, *mask_shape)
        semantic_indices = max_indices.unsqueeze(-1).expand(-1, -1, 512)
        
        selected_masks = torch.gather(low_res_masks, 1, low_mask_indices)
        selected_semantic = torch.gather(semantic_pred, 1, semantic_indices)
        
        return selected_masks, max_values, selected_semantic
    

    def forward(
            self, 
            image: torch.Tensor, 
            prompt: Dict[str, Any], 
            image_embedding: torch.Tensor = None,
            mode: str = 'full'
        ) -> Dict[str, torch.Tensor]:
        """Forward pass of the network."""
        if mode == 'full':
            image_embedding = self.encode_image(image)
            return self.decode_masks(image_embedding, prompt)
        else:
            return self.decode_masks(image_embedding, prompt)
    

    def generate_prompts(
        self,
        prompt_type: str,
        labels: Optional[torch.Tensor] = None,
        classes: Optional[List[str]] = None,
        pred_masks: Optional[torch.Tensor] = None,
        low_res_masks: Optional[torch.Tensor] = None,
        specify_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate prompts for training or inference.
        
        Args:
            prompt_type: 'supervised' or 'unsupervised'
            labels: Ground truth masks
            classes: Class names for text prompts
            pred_masks: Predicted masks for iterative refinement
            low_res_masks: Low resolution masks for mask prompts
            specify_prompt: Type of prompt ('points', 'text', 'bboxes')
        
        Returns:
            Dictionary containing generated prompts
        """
        prompts = {}
        
        # Add mask prompt if available
        if low_res_masks is not None:
            prompts.update(self.prompt_processor.process_mask_prompt(low_res_masks))
        
        # Generate specific prompt type
        if specify_prompt == 'points':
            prompts.update(self.prompt_processor.process_points_prompt(labels, pred_masks))
        elif specify_prompt == 'text' and classes is not None:
            text_embedding = self.text_processor.tokenize_text(classes)
            prompts['text_inputs'] = text_embedding
        elif specify_prompt == 'bboxes':
            prompts.update(self.prompt_processor.process_bboxes_prompt(labels))
        
        if not prompts:
            raise ValueError(f'No valid prompts generated: {prompts}')
        
        return prompts
    

    # Compatibility methods for backward compatibility
    def supervised_prompts(
        self, 
        classes: List[str], 
        labels: torch.Tensor, 
        pred_masks: Optional[torch.Tensor], 
        low_res_masks: Optional[torch.Tensor], 
        specify_prompt: str
    ) -> Dict[str, Any]:
        """Generate supervised prompts (backward compatibility)."""
        return self.generate_prompts(
            'supervised', labels, classes, pred_masks, low_res_masks, specify_prompt
        )
    

    def unsupervised_prompts(
        self,
        pseudo_labels: torch.Tensor,
        pred_masks: Optional[torch.Tensor],
        low_res_masks: Optional[torch.Tensor],
        specify_prompt: str
    ) -> Dict[str, Any]:
        """Generate unsupervised prompts (backward compatibility)."""
        return self.generate_prompts(
            'unsupervised', pseudo_labels, None, pred_masks, low_res_masks, specify_prompt
        )
    
    # Delegate methods to processors for backward compatibility
    def category_loss(
        self, 
        semantic_preds: torch.Tensor, 
        classes: List[str], 
        ce_loss: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute category loss (backward compatibility)."""
        return self.text_processor.compute_category_loss(semantic_preds, classes, ce_loss)
    

    def text_tokenizer(self, text: List[str], template: str = None) -> torch.Tensor:
        """Tokenize text (backward compatibility)."""
        if template:
            self.text_processor.template = template
        return self.text_processor.tokenize_text(text)
