import argparse
import torch
import sys
import warnings

from omegaconf import OmegaConf
from pathlib import Path

try:
    import onnxruntime  # type: ignore
    onnxruntime_exists = True
except ImportError:
    onnxruntime_exists = False

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import create_config, parse_args
from src.utils.build_sam import get_sam_model, sam_model_registry
from src.utils.logging_utils import LoggingManager
from src.onnx.onnx_model import SamOnnxModel


def onnx_args(args=None) -> OmegaConf:
    parser = argparse.ArgumentParser(
        description="Export the SAM prompt encoder and mask decoder to an ONNX model."
    )
    # Only keep frequently modified arguments in argparse
    parser.add_argument("--config", type=str, help="YAML config file path")
    parser.add_argument("--checkpoint_path", type=str, help="The path to the SAM model checkpoint.")
    parser.add_argument("--output", type=str, default= "output/checkpoint/sam.onnx", help="The filename to save the ONNX model to.")

    # parser.add_argument("--model-type", type=str, required=True,
    #     help="In ['default', 'vit_h', 'vit_l', 'vit_b']. Which type of SAM model to export.",
    # )

    parser.add_argument("--return-single-mask", action="store_true",
        help=(
            "If true, the exported ONNX model will only return the best mask, "
            "instead of returning multiple masks. For high resolution images "
            "this can improve runtime when upscaling masks is expensive."
        ),
    )

    parser.add_argument("--opset", type=int, default=17, help="The ONNX opset version to use. Must be >=11",
    )

    parser.add_argument("--quantize-out", type=str, default=None,
        help=(
            "If set, will quantize the model and save it with this name. "
            "Quantization is performed with quantize_dynamic from onnxruntime.quantization.quantize."
        ),
    )

    parser.add_argument("--gelu-approximate", action="store_true",
        help=(
            "Replace GELU operations with approximations using tanh. Useful "
            "for some runtimes that have slow or unimplemented erf ops, used in GELU."
        ),
    )

    parser.add_argument("--use-stability-score", action="store_true",
        help=(
            "Replaces the model's predicted mask quality score with the stability "
            "score calculated on the low resolution masks using an offset of 1.0. "
        ),
    )

    parser.add_argument("--return-extra-metrics", action="store_true",
        help=(
            "The model will return five results: (masks, scores, stability_scores, "
            "areas, low_res_logits) instead of the usual three. This can be "
            "significantly slower for high resolution outputs."
        ),
    )
    parsed_args = parser.parse_args(args)
   
    # Create the configuration object
    config = create_config(parsed_args)

    # Print configuration using the logging utility
    LoggingManager.print_config(config, "Configuration")

    return config


def run_export(
    config: OmegaConf, 
    model_type: str,
    checkpoint: str,
    output: str,
    opset: int,
    return_single_mask: bool,
    gelu_approximate: bool = False,
    use_stability_score: bool = False,
    return_extra_metrics=False,
):
    print("Loading model...")
    #sam = sam_model_registry[model_type](checkpoint=checkpoint)
    sam, image_encoder, text_encoder = get_sam_model(model_type, config)
    
    onnx_model = SamOnnxModel(
        model=sam,
        img_size=config.model.image_size,
        return_single_mask=return_single_mask,
        use_stability_score=use_stability_score,
        return_extra_metrics=return_extra_metrics,
    )

    if gelu_approximate:
        for n, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float32),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float32),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float32),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float32),
        "has_mask_input": torch.tensor([1], dtype=torch.float32),
        "orig_im_size": torch.tensor([1024, 1024], dtype=torch.float32),
    }
    
    _ = onnx_model(**dummy_inputs)

    output_names = ["masks", "iou_predictions", "low_res_masks"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(output, "wb") as f:
            print(f"Exporting onnx model to {output}...")
            torch.onnx.export(
                onnx_model,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )
    quit()
    if onnxruntime_exists:
        ort_inputs = {k: to_numpy(v) for k, v in dummy_inputs.items()}
        # set cpu provider default
        providers = ["CPUExecutionProvider"]
        ort_session = onnxruntime.InferenceSession(output, providers=providers)
        _ = ort_session.run(None, ort_inputs)
        print("Model has successfully been run with ONNXRuntime.")


def to_numpy(tensor):
    return tensor.cpu().numpy()


if __name__ == "__main__":
    #config = parse_args()
    args = onnx_args()
    
    run_export(
        config=args,
        model_type=args.model.sam_model_type, #args.model_type,
        checkpoint=args.checkpoint_path, #args.checkpoint,
        output=args.output,
        opset=args.opset,
        return_single_mask=args.return_single_mask,
        gelu_approximate=args.gelu_approximate,
        use_stability_score=args.use_stability_score,
        return_extra_metrics=args.return_extra_metrics,
    )

    if args.quantize_out is not None:
        assert onnxruntime_exists, "onnxruntime is required to quantize the model."
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output,
            model_output=args.quantize_out,
            optimize_model=True,
            per_channel=False,
            reduce_range=False,
            weight_type=QuantType.QUInt8,
        )
        print("Done!")
