
import torch
import numpy as np

from torch.nn import functional as F
from typing import  Tuple, Union
from monai import transforms



class LongestSidePadding(transforms.Transform):
    def __init__(self, keys, input_size):
        self.keys = keys
        self.input_size = input_size
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            h, w = d[key].shape[-2:]
            padh = self.input_size - h
            padw = self.input_size - w
            d[key] = F.pad(d[key], (0, padw, 0, padh))
        return d
    

def get_points_from_mask(
    mask: Union[torch.Tensor, np.ndarray], 
    get_point: int = 1, 
    top_ratio: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract random points from a binary mask, preferring points near the centroid.
    
    Args:
        mask: Binary mask (H, W) or (1, H, W)
        get_point: Number of points to extract (currently only supports 1)
        top_ratio: Ratio of foreground points to consider (closest to centroid)
        
    Returns:
        coords: Point coordinates as (1, 2) tensor [x, y]
        labels: Point labels as (1,) tensor (1 for foreground, 0 for background)
    """
    # Squeeze to 2D if needed
    if len(mask.shape) > 2:
        mask = mask.squeeze()
    
    # Convert to numpy if tensor
    if isinstance(mask, torch.Tensor):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask.copy()
    
    # Get foreground and background coordinates
    fg_coords = np.argwhere(mask_np == 1)[:, ::-1]  # Convert to (x, y)
    bg_coords = np.argwhere(mask_np == 0)[:, ::-1]  # Convert to (x, y)
    
    # Try to get foreground point
    if len(fg_coords) > 0:
        if len(fg_coords) == 1:
            # Only one foreground pixel
            coord = fg_coords[0]
            label = 1
        else:
            # Select point near centroid
            centroid = np.mean(fg_coords, axis=0)
            distances = np.linalg.norm(fg_coords - centroid, axis=1)
            sorted_indices = np.argsort(distances)
            
            # Select from top percentage closest to centroid
            top_k = max(1, int(len(fg_coords) * top_ratio))
            top_indices = sorted_indices[:top_k]
            
            selected_idx = np.random.choice(top_indices)
            coord = fg_coords[selected_idx]
            label = 1
    else:
        # No foreground pixels, select background
        if len(bg_coords) == 0:
            raise ValueError("Mask contains no valid pixels")
        coord = bg_coords[np.random.randint(len(bg_coords))]
        label = 0
    
    # Convert to tensors
    coords = torch.as_tensor([coord.tolist()], dtype=torch.float)
    labels = torch.as_tensor([label], dtype=torch.int)
    
    return coords, labels


def get_bboxes_from_mask(
    masks: torch.Tensor, 
    offset: int = 0
) -> torch.Tensor:
    """
    Extract bounding boxes from binary masks.
    
    Args:
        masks: Binary masks of shape (B, H, W) or (B, 1, H, W)
        offset: Random offset to apply to bbox coordinates
        
    Returns:
        Bounding boxes of shape (B, 1, 4) in format [x0, y0, x1, y1]
    """
    # Handle channel dimension
    if masks.dim() == 4 and masks.size(1) == 1:
        masks = masks.squeeze(1)
    
    if masks.dim() != 3:
        raise ValueError(f"Expected masks of shape (B, H, W), got {masks.shape}")
    
    B, H, W = masks.shape
    bounding_boxes = []
    
    for i in range(B):
        mask = masks[i]
        y_coords, x_coords = torch.nonzero(mask, as_tuple=True)
        
        if len(y_coords) == 0:
            # No foreground pixels
            bounding_boxes.append([0, 0, 0, 0])
        else:
            y0, y1 = y_coords.min().item(), y_coords.max().item()
            x0, x1 = x_coords.min().item(), x_coords.max().item()
            
            # Apply random offset if specified
            if offset > 0:
                y0 = max(0, y0 + torch.randint(-offset, offset + 1, (1,)).item())
                y1 = min(H - 1, y1 + torch.randint(-offset, offset + 1, (1,)).item())
                x0 = max(0, x0 + torch.randint(-offset, offset + 1, (1,)).item())
                x1 = min(W - 1, x1 + torch.randint(-offset, offset + 1, (1,)).item())
            
            bounding_boxes.append([x0, y0, x1, y1])
    
    return torch.tensor(bounding_boxes, dtype=torch.float).unsqueeze(1)


def main():
    """Demo function to test bbox extraction functionality."""
    import cv2
    import matplotlib.pyplot as plt
    
    image_path = "./data/samples/x-ray.jpg"
    
    # Load and preprocess mask
    masks = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if masks is None:
        print(f"Could not load image from {image_path}")
        return
    
    # Convert to tensor and add batch dimension
    masks_tensor = torch.from_numpy(masks).unsqueeze(0)
    
    # Extract bounding boxes
    bboxes = get_bboxes_from_mask(masks_tensor, offset=0)
    bboxes_np = bboxes.squeeze(0).numpy()
    
    # Visualize
    plt.figure(figsize=(10, 8))
    plt.imshow(masks, cmap='gray')
    plt.title('Mask with Bounding Boxes')
    
    for box in bboxes_np:
        x0, y0, x1, y1 = box
        width, height = x1 - x0, y1 - y0
        rect = plt.Rectangle((x0, y0), width, height, 
                            edgecolor='green', facecolor='none', linewidth=3)
        plt.gca().add_patch(rect)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()