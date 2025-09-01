import onnxruntime as ort
import numpy as np
from typing import Dict, Tuple, Optional
import torch

class ONNXMaskDecoder:
    """
    ONNX Runtime wrapper for the exported mask decoder model.
    """
    
    def __init__(self, onnx_path: str, providers: Optional[list] = None):
        """
        Initialize the ONNX model.
        
        Args:
            onnx_path: Path to the ONNX model file
            providers: List of execution providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        self.onnx_path = onnx_path
        
        # Set default providers if none specified
        if providers is None:
            providers = ['CPUExecutionProvider']
            # Try to use CUDA if available
            if ort.get_device() == 'GPU':
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Create inference session
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        
        # Get model info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"Loaded ONNX model from: {onnx_path}")
        print(f"Available providers: {self.session.get_providers()}")
        print(f"Input names: {self.input_names}")
        print(f"Output names: {self.output_names}")
    
    def prepare_inputs(self, 
                      image_embedding: np.ndarray,
                      point_coords: Optional[np.ndarray] = None,
                      point_labels: Optional[np.ndarray] = None,
                      boxes: Optional[np.ndarray] = None,
                      mask_inputs: Optional[np.ndarray] = None,
                      text_inputs: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for the ONNX model.
        
        Args:
            image_embedding: Image embedding tensor [B, 768, 64, 64]
            point_coords: Point coordinates [B, N, 2] (optional)
            point_labels: Point labels [B, N] (optional)
            boxes: Bounding boxes [B, 4] (optional)
            mask_inputs: Mask inputs [B, 1, 256, 256] (optional)
            text_inputs: Text embeddings [B, 768] (optional)
            
        Returns:
            Dictionary of prepared inputs
        """
        batch_size = image_embedding.shape[0]
        
        # Prepare inputs with default values
        inputs = {
            'image_embedding': image_embedding.astype(np.float32),
        }
        
        # Handle point inputs
        if point_coords is not None and point_labels is not None:
            inputs['point_coords'] = point_coords.astype(np.float32)
            inputs['point_labels'] = point_labels.astype(np.int64)
            inputs['has_points'] = np.ones((batch_size,), dtype=np.int64)
        else:
            # Default: single point at center
            inputs['point_coords'] = np.array([[[100.0, 100.0]]] * batch_size, dtype=np.float32)
            inputs['point_labels'] = np.array([[1]] * batch_size, dtype=np.int64)
            inputs['has_points'] = np.zeros((batch_size,), dtype=np.int64)
        
        # Handle box inputs
        if boxes is not None:
            inputs['boxes'] = boxes.astype(np.float32)
            inputs['has_boxes'] = np.ones((batch_size,), dtype=np.int64)
        else:
            inputs['boxes'] = np.array([[50.0, 50.0, 150.0, 150.0]] * batch_size, dtype=np.float32)
            inputs['has_boxes'] = np.zeros((batch_size,), dtype=np.int64)
        
        # Handle mask inputs
        if mask_inputs is not None:
            inputs['mask_inputs'] = mask_inputs.astype(np.float32)
            inputs['has_mask'] = np.ones((batch_size,), dtype=np.int64)
        else:
            inputs['mask_inputs'] = np.zeros((batch_size, 1, 256, 256), dtype=np.float32)
            inputs['has_mask'] = np.zeros((batch_size,), dtype=np.int64)
        
        # Handle text inputs
        if text_inputs is not None:
            inputs['text_inputs'] = text_inputs.astype(np.float32)
            inputs['has_text'] = np.ones((batch_size,), dtype=np.int64)
        else:
            inputs['text_inputs'] = np.zeros((batch_size, 768), dtype=np.float32)
            inputs['has_text'] = np.zeros((batch_size,), dtype=np.int64)
        
        return inputs
    
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Run inference on the ONNX model.
        
        Args:
            inputs: Dictionary of input arrays
            
        Returns:
            Dictionary of output arrays
        """
        # Run inference
        outputs = self.session.run(self.output_names, inputs)
        
        # Return as dictionary
        return {name: output for name, output in zip(self.output_names, outputs)}
    
    def __call__(self, **kwargs) -> Dict[str, np.ndarray]:
        """
        Convenience method for running inference.
        """
        inputs = self.prepare_inputs(**kwargs)
        return self.predict(inputs)


def load_and_test_model(onnx_path: str):
    """
    Load the ONNX model and run a simple test.
    
    Args:
        onnx_path: Path to the ONNX model file
    """
    # Initialize the model
    model = ONNXMaskDecoder(onnx_path)
    
    # Create example inputs
    batch_size = 1
    image_embedding = np.random.randn(batch_size, 768, 64, 64).astype(np.float32)
    
    # Option 1: Using point coordinates
    point_coords = np.array([[[120.0, 80.0]]], dtype=np.float32)  # Single point
    point_labels = np.array([[1]], dtype=np.int64)  # Positive point
    
    print("Running inference with point inputs...")
    outputs = model(
        image_embedding=image_embedding,
        point_coords=point_coords,
        point_labels=point_labels
    )
    
    # Print output shapes
    for name, output in outputs.items():
        print(f"{name}: {output.shape}")
    
    # Option 2: Using bounding box
    print("\nRunning inference with box inputs...")
    boxes = np.array([[50.0, 50.0, 150.0, 150.0]], dtype=np.float32)  # [x1, y1, x2, y2]
    
    outputs = model(
        image_embedding=image_embedding,
        boxes=boxes
    )
    
    # Print output shapes
    for name, output in outputs.items():
        print(f"{name}: {output.shape}")
    
    return model, outputs


def convert_torch_to_numpy(torch_tensor: torch.Tensor) -> np.ndarray:
    """
    Helper function to convert PyTorch tensors to NumPy arrays.
    
    Args:
        torch_tensor: PyTorch tensor
        
    Returns:
        NumPy array
    """
    if torch_tensor.requires_grad:
        torch_tensor = torch_tensor.detach()
    
    if torch_tensor.is_cuda:
        torch_tensor = torch_tensor.cpu()
    
    return torch_tensor.numpy()


# Example usage
if __name__ == "__main__":
    # Path to your exported ONNX model
    onnx_model_path = "output/checkpoint/mask_decoder_wrapper_IMISNet.onnx"
    
    try:
        # Load and test the model
        model, outputs = load_and_test_model(onnx_model_path)
        
        print("\n✅ Model loaded and tested successfully!")
        print(f"Model outputs: {list(outputs.keys())}")
        
        # Example of how to use the model in your application
        print("\n--- Example usage in your application ---")
        
        # If you have PyTorch tensors, convert them first
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
            
        # Example with PyTorch tensors (convert to numpy)
        torch_embedding = torch.randn(1, 768, 64, 64, device=device)
        torch_points = torch.tensor([[[100.0, 100.0]]], device=device)
        torch_labels = torch.tensor([[1]], device=device)
        
        # Convert to numpy
        np_embedding = convert_torch_to_numpy(torch_embedding)
        np_points = convert_torch_to_numpy(torch_points)
        np_labels = convert_torch_to_numpy(torch_labels)
        
        # Run inference
        results = model(
            image_embedding=np_embedding,
            point_coords=np_points,
            point_labels=np_labels
        )
        
        print("Inference completed with converted tensors!")
        
    except FileNotFoundError:
        print(f"❌ ONNX model not found at: {onnx_model_path}")
        print("Please make sure you have exported the model first.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Please check that ONNX Runtime is installed: pip install onnxruntime")