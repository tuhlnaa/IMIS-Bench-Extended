import sys
import torch
import onnx
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers import AutoTokenizer

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import parse_args
from src.utils.inference import determine_device
from segment_anything.build_sam import get_sam_model


def export_to_onnx(model, output_path, device):
    """Export model to ONNX using TorchDynamo."""
    model.eval()
    
    # Create dummy inputs
    batch_size, seq_len = 1, 77
    dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=device)
    
    # Export with TorchDynamo
    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=['input_ids', 'attention_mask'],
            output_names=['text_embeddings'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'text_embeddings': {0: 'batch_size'}
            },
            opset_version=17,
            # dynamo=True  # TorchDynamo-based export
        )
    print(f"Model exported to {output_path}")


def test_onnx_model(onnx_path, original_model, device):
    """Load ONNX model and compare with original."""
    # Load ONNX model
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device.type == 'cuda' else ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Prepare test data
    tokenizer = AutoTokenizer.from_pretrained('openai/clip-vit-base-patch32')
    test_texts = ["A segmentation area of a cat.", "A segmentation area of a dog."]
    tokens = tokenizer(test_texts, padding=True, return_tensors="pt")
    
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    # PyTorch inference
    original_model.eval()
    with torch.no_grad():
        torch_output = original_model(input_ids, attention_mask)
    
    # ONNX inference
    onnx_inputs = {
        'input_ids': input_ids.cpu().numpy().astype(np.int64),
        'attention_mask': attention_mask.cpu().numpy().astype(np.int64)
    }
    onnx_outputs = ort_session.run(None, onnx_inputs)
    onnx_output = torch.from_numpy(onnx_outputs[0])
    
    # Compare results
    max_diff = torch.max(torch.abs(torch_output.cpu() - onnx_output)).item()
    print(f"PyTorch output shape: {torch_output.shape}")
    print(f"ONNX output shape: {onnx_output.shape}")
    print(f"Max difference: {max_diff:.8f}")
    print("✓ Models match!" if max_diff < 1e-5 else "⚠ Models differ")


if __name__ == "__main__":
    # Load your model
    config = parse_args()
    device = determine_device(config)
    sam, image_encoder, text_encoder = get_sam_model(config.model.sam_model_type, config)
    text_encoder = text_encoder.to(device)

    # Export to ONNX
    onnx_path = "text_encoder.onnx"
    export_to_onnx(text_encoder, onnx_path, device)
    
    # Verify ONNX model
    onnx.checker.check_model(onnx.load(onnx_path))
    print("✓ ONNX model verified")
    
    # Test the model
    test_onnx_model(onnx_path, text_encoder, device)