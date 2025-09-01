import sys
import torch

from pathlib import Path
from typing import Optional

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from model import MaskDecoderONNX
from configs.config import parse_args
from src.utils.inference import determine_device, load_model


def export_mask_decoder_onnx(
    imisnet_model,
    device: torch.device,
    onnx_path: str,
    example_inputs: Optional[dict] = None,
    opset_version: int = 17
):
    """
    Export the mask decoder to ONNX format.
    
    Args:
        imisnet_model: Trained IMISNet model
        onnx_path: Path to save ONNX model
        example_inputs: Dictionary of example inputs (optional)
        opset_version: ONNX opset version
    """
    # Create wrapper
    wrapper = MaskDecoderONNX(imisnet_model)
    wrapper.eval()
    
    # Create example inputs if not provided
    if example_inputs is None:
        batch_size = 1
        example_inputs = {
            'image_embedding': torch.randn(1, 768, 64, 64, device=device),  # Adjust based on your model
            'point_coords': torch.tensor([[[100.0, 100.0]]], device=device),  # [B, N, 2]
            'point_labels': torch.tensor([[1]], device=device),  # [B, N]
            'has_points': torch.tensor([1], device=device),  # [B]
            'boxes': torch.tensor([[50.0, 50.0, 150.0, 150.0]], device=device),  # [B, 4]
            'has_boxes': torch.tensor([0], device=device),  # [B] - no boxes in this example
            'mask_inputs': torch.zeros(1, 1, 256, 256, device=device),  # [B, 1, H, W]
            'has_mask': torch.tensor([0], device=device),  # [B]
            'text_inputs': torch.zeros(1, 768, device=device),  # [B, embed_dim] - adjust embed_dim
            'has_text': torch.tensor([0], device=device),  # [B]
        }
    
    # Define input names
    input_names = [
        'image_embedding',
        'point_coords', 'point_labels', 'has_points',
        'boxes', 'has_boxes',
        'mask_inputs', 'has_mask',
        'text_inputs', 'has_text'
    ]
    
    # Define output names
    output_names = ['masks', 'low_res_masks', 'iou_pred', 'semantic_pred']
    
    # Define dynamic axes for variable batch size
    dynamic_axes = {
        'image_embedding': {0: 'batch_size'},
        'point_coords': {0: 'batch_size'},
        'point_labels': {0: 'batch_size'},
        'has_points': {0: 'batch_size'},
        'boxes': {0: 'batch_size'},
        'has_boxes': {0: 'batch_size'},
        'mask_inputs': {0: 'batch_size'},
        'has_mask': {0: 'batch_size'},
        'text_inputs': {0: 'batch_size'},
        'has_text': {0: 'batch_size'},
        'masks': {0: 'batch_size'},
        'low_res_masks': {0: 'batch_size'},
        'iou_pred': {0: 'batch_size'},
        'semantic_pred': {0: 'batch_size'},
    }
    
    # Define dynamic shapes for TorchDynamo (replaces dynamic_axes)
    from torch.export import Dim
    
    # Create symbolic dimensions
    batch_dim = Dim("batch_size", min=1, max=2)  # Allow batch sizes from 1 to 2
    num_points_dim = Dim("num_points", min=0, max=2)  # Allow 0 to 2 points
    
    dynamic_shapes = {
        'image_embedding': (batch_dim, None, None, None),  # [B, C, H, W]
        'point_coords': (batch_dim, num_points_dim, None),  # [B, N, 2]
        'point_labels': (batch_dim, num_points_dim),  # [B, N]
        'has_points': (batch_dim,),  # [B]
        'boxes': (batch_dim, None),  # [B, 4]
        'has_boxes': (batch_dim,),  # [B]
        'mask_inputs': (batch_dim, None, None, None),  # [B, 1, H, W]
        'has_mask': (batch_dim,),  # [B]
        'text_inputs': (batch_dim, None),  # [B, embed_dim]
        'has_text': (batch_dim,),  # [B]
    }

    # # Export to ONNX
    # torch.onnx.export(
    #     wrapper,
    #     tuple(example_inputs.values()),
    #     onnx_path,
    #     input_names=input_names,
    #     output_names=output_names,
    #     dynamic_axes=dynamic_axes,
    #     opset_version=opset_version,
    #     do_constant_folding=True,
    #     report=True
    # )

    # Export to ONNX (TorchDynamo-based)
    onnx_program = torch.onnx.export(
        wrapper,
        tuple(example_inputs.values()),
        dynamo=True
    )
    onnx_program.optimize()
    onnx_program.save(onnx_path)

    print(f"Model exported to {onnx_path}")
    return wrapper


# Example usage
if __name__ == "__main__":
    config = parse_args()
    device = determine_device(config)
    imisnet_model, _, _ = load_model(config, device)

    # Export example
    wrapper = export_mask_decoder_onnx(
        imisnet_model,
        device,
        "output/checkpoint/mask_decoder_wrapper_IMISNet.onnx",
    )