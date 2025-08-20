
import re
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from transformers import CLIPTextModel, CLIPTextConfig, AutoTokenizer
from typing import List, Optional, Tuple


@dataclass
class PromptConfig:
    """Configuration for prompt generation."""
    point_num_choices: List[int] = None
    template: str = 'A segmentation area of a {}.'
    
    def __post_init__(self):
        if self.point_num_choices is None:
            self.point_num_choices = [1, 3, 4, 7]


class StandaloneTextEncoder(nn.Module):
    """Standalone text encoder that combines CLIP text model with projection layer."""
    
    def __init__(
        self, 
        text_model: Optional[CLIPTextModel] = None,
        input_dim: int = 512,
        output_dim: int = 768
    ):
        super().__init__()
        
        # Use provided model or create default
        self.text_model = text_model if text_model is not None else CLIPTextModel(CLIPTextConfig())
        self.projection = nn.Linear(input_dim, output_dim)

        # Freeze text model parameters
        self._freeze_text_model()
    
    def _freeze_text_model(self) -> None:
        """Freeze text model parameters."""
        for param in self.text_model.parameters():
            param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through text model and projection."""
        text_outputs = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        text_embedding = text_outputs.pooler_output
        projected_embedding = self.projection(text_embedding)
        return projected_embedding

    @classmethod
    def from_sam_model(cls, sam_model, device: torch.device = None):
        """Extract and combine weights from existing SAM model."""
        # Extract the text model and projection weights
        text_model = sam_model.text_model
        text_out_dim = sam_model.text_out_dim
        
        # Create new standalone encoder
        encoder = cls(text_model=None, 
                     input_dim=text_out_dim.in_features, 
                     output_dim=text_out_dim.out_features)
        
        # Copy weights
        encoder.text_model.load_state_dict(text_model.state_dict())
        encoder.projection.load_state_dict(text_out_dim.state_dict())
        
        if device:
            encoder = encoder.to(device)
            
        return encoder
    

class TextProcessor:
    """Refactored text processor with standalone encoder."""
    
    def __init__(
        self, 
        device: torch.device,
        text_encoder: Optional[StandaloneTextEncoder] = None,
        tokenizer_name: str = 'openai/clip-vit-base-patch32',
        category_weights_path: Optional[str] = None
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.text_encoder = text_encoder or StandaloneTextEncoder().to(device)
        self.template = PromptConfig().template
        
        # Category mapping attributes
        self.src_weights = None
        self.categories_map = None
        self.category_to_index = None
        self.index_to_category = None
        
        if category_weights_path:
            self.load_category_weights(category_weights_path)
    

    def load_category_weights(self, weights_path: str) -> None:
        """Load category weights and mappings from file."""
        with open(weights_path, "rb") as f:
            (self.src_weights, 
             self.categories_map, 
             self.category_to_index, 
             self.index_to_category) = pickle.load(f)
            self.src_weights = torch.tensor(self.src_weights).to(self.device)
    

    def tokenize_text(self, text: List[str]) -> torch.Tensor:
        """Simplified tokenization using standalone encoder."""
        # Normalize text
        norm_text = [self._normalize_text(t) for t in text]
        text_list = [self.template.format(t) for t in norm_text]
        
        # Tokenize
        tokens = self.tokenizer(text_list, padding=True, return_tensors="pt")
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        
        # Get text embeddings using standalone encoder
        with torch.no_grad():
            text_embedding = self.text_encoder(**tokens)
        
        return text_embedding
    

    def _normalize_text(self, text: str) -> str:
        """Normalize text for processing."""
        if self.categories_map and text in self.categories_map:
            text = self.categories_map[text][0]
        
        text = text.lower().replace('_', ' ').replace("-", " ")
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    

    def get_category_labels(self, classes: List[str]) -> torch.Tensor:
        """Convert class names to category indices."""
        if not self.categories_map:
            raise ValueError("Category weights not loaded")
        
        norm_targets = []
        for cls in classes:
            cls_mapped = self.categories_map[cls][1]
            category = cls_mapped.lower().replace('_', ' ').replace("-", " ")
            category = category.replace('left', '').replace('right', '').strip()
            category = re.sub(r'\s+', ' ', category)
            norm_targets.append(category)
        
        indices = [self.category_to_index[cat] for cat in norm_targets]
        return torch.tensor(indices).unsqueeze(-1).to(self.device)
    

    def compute_category_loss(
        self, 
        semantic_preds: torch.Tensor, 
        classes: List[str], 
        ce_loss: nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute category prediction loss."""
        if not self.src_weights:
            raise ValueError("Category weights not loaded")
        
        labels = self.get_category_labels(classes)
        logits = F.normalize(semantic_preds, dim=-1) @ self.src_weights
        probs = F.softmax(logits, dim=-1)
        loss = ce_loss(probs.squeeze(1), labels.squeeze(1))
        
        return loss, probs
