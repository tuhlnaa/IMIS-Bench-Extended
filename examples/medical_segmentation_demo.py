# examples/medical_segmentation_demo.py
import sys
import numpy as np
from pathlib import Path

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import parse_args
from src.utils.inference import InteractiveSegmentationSession


def main():
    # Initialize configuration
    config = parse_args()
    
    # Initialize session once
    session = InteractiveSegmentationSession(config, output_dir="output")

    examples1 = [
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

    examples2 = [
        {
            'name': 'Multiple bounding boxes segmentation',
            'bounding_box': np.array([[87, 228, 140, 297], [408, 226, 459, 276], [170, 212, 225, 260], [215, 192, 312, 240]])
        }
    ]
    
    examples3 = [
        {
            'name': 'Single click segmentation',
            'points': np.array([[304, 148]]),
            'labels': np.array([1]),
        },
        {
            'name': 'Multiple bounding boxes segmentation',
            'bounding_box': np.array([[264, 85, 450, 364], [53, 88, 273, 370]]),
        }     
    ]

    examples4 = [
        {
            'name': 'Single click segmentation',
            'points': np.array([[116, 158]]),
            'labels': np.array([1]),
        }
    ]

    results1 = session.process_image_with_examples("data/samples/train_177_51.png", examples1)
    results2 = session.process_image_with_examples("data/samples/ABD_001_67.png", examples2)
    results3 = session.process_image_with_examples("data/samples/lung_005_160.png", examples3)
    results4 = session.process_image_with_examples("data/samples/ISIC_0012092.jpg", examples4)

    # Print summary of results
    print(f"\n=== Summary ===")
    print(f"Total examples processed: {len(results1)}")
    for name, result in results1.items():
        print(f"- {name}: Category {result['category_pred']}")


if __name__ == "__main__":
    main()
