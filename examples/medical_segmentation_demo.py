# examples/medical_segmentation_demo.py
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.segmentation import MedicalImageSegmentation
from src.utils.visualization import VisualizationUtils
from src.utils.inference import run_segmentation, run_workflow_chain


def run_interactive_point_demo(image_path: str = 'demo_image/train_177_51.png', output_dir: str = "output") -> None:
    """Run interactive segmentation demonstration with declarative multi-step workflows."""
    image_path = Path(image_path)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    filename_stem = image_path.stem

    # Initialize demo components
    demo = MedicalImageSegmentation()
    vis_utils = VisualizationUtils(output_path, filename_stem)

    # Load and display original image
    image = demo.load_image(image_path)
    plt.figure()
    plt.imshow(image)
    vis_utils.save_plot("Original Image", "original")
    print(f"Image shape: {image.shape}")

    # All examples including multi-step workflows defined declaratively
    examples = [
        # Single-step examples
        {
            'name': 'Single click segmentation',
            'points': np.array([[188, 205]]),
            'labels': np.array([1]),
        },
        {
            'name': 'Different region segmentation',
            'points': np.array([[346, 211]]),
            'labels': np.array([1]),
        },
        {
            'name': 'Bounding box segmentation',
            'bounding_box': np.array([215, 118, 304, 232]),
        },
        {
            'name': 'Text prompt - kidney right',
            'text_prompt': ['kidney_right'],
        },
        {
            'name': 'Text prompt - kidney left',
            'text_prompt': ['kidney_left'],
        },
        {
            'name': 'Text prompt - liver',
            'text_prompt': ['liver'],
        },
        {
            'name': 'Multiple text prompts - both kidneys',
            'text_prompt': ['kidney_right', 'kidney_left']
        },

        # Multi-step workflows defined declaratively
        {
            'name': 'Point-based refinement workflow',
            'workflow_type': 'multistep',
            'steps': [
                {
                    'name': 'Point Initial Prediction (liver)',
                    'points': np.array([[378, 258]]),
                    'labels': np.array([1])
                },
                {
                    'name': 'Point Refined Prediction (liver)',
                    'points': np.array([[311, 287]]),
                    'labels': np.array([1]),
                    'use_previous_logits': True
                }
            ]
        },
        {
            'name': 'Text-to-point refinement workflow',
            'workflow_type': 'multistep',
            'steps': [
                {
                    'name': 'Prompt Initial Prediction (liver)',
                    'text_prompt': ['liver']
                },
                {
                    'name': 'Point Refined Prediction (liver)',
                    'points': np.array([[311, 287]]),
                    'labels': np.array([1]),
                    'use_previous_logits': True
                }
            ]
        }
    ]

    # Process all examples using unified logic
    results = {}
    for example in examples:
        if example.get('workflow_type') == 'multistep':
            # Handle multi-step workflow
            masks, logits, category_pred = run_workflow_chain(demo, vis_utils, image, example['steps'])
            results[example['name']] = {
                'masks': masks,
                'logits': logits,
                'category_pred': category_pred
            }
        else:
            # Handle single-step example
            masks, logits, category_pred = run_segmentation(demo, vis_utils, image, example)
            results[example['name']] = {
                'masks': masks,
                'logits': logits,
                'category_pred': category_pred
            }

    print(f"\nSegmentation demonstration completed. Results saved to {output_path}")
    return results


if __name__ == "__main__":
    results = run_interactive_point_demo()
    
    # Print summary of results
    print(f"\n=== Summary ===")
    print(f"Total examples processed: {len(results)}")
    for name, result in results.items():
        print(f"- {name}: Category {result['category_pred']}")