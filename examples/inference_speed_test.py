# test_inference_speed.py
import sys
import time
import numpy as np
from pathlib import Path
from PIL import Image
import random
from typing import Dict, List, Tuple

# Import custom modules
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.append(str(PROJECT_ROOT))

from configs.config import parse_args
from src.utils.inference import InteractiveSegmentationSession


def generate_dummy_image(width: int = 512, height: int = 512) -> np.ndarray:
    """Generate a dummy RGB image with random noise."""
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


def save_dummy_image(image_array: np.ndarray, filepath: str) -> None:
    """Save dummy image array as PIL Image."""
    image = Image.fromarray(image_array.astype(np.uint8))
    image.save(filepath)


def generate_random_click(width: int = 512, height: int = 512) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random single click coordinates."""
    x = random.randint(50, width - 50)
    y = random.randint(50, height - 50)
    points = np.array([[x, y]])
    labels = np.array([1])
    return points, labels


def generate_random_bbox(width: int = 512, height: int = 512) -> np.ndarray:
    """Generate random bounding box coordinates."""
    x1 = random.randint(20, width // 2)
    y1 = random.randint(20, height // 2)
    x2 = random.randint(x1 + 50, width - 20)
    y2 = random.randint(y1 + 50, height - 20)
    return np.array([x1, y1, x2, y2])


def generate_consecutive_clicks(width: int = 512, height: int = 512, num_clicks: int = 3) -> List[Dict]:
    """Generate consecutive click examples with use_previous_logits=True."""
    consecutive_examples = []
    
    # First click
    points, labels = generate_random_click(width, height)
    consecutive_examples.append({
        'name': 'Consecutive Click 1',
        'points': points,
        'labels': labels,
    })
    
    # Subsequent clicks with previous logits
    for i in range(2, num_clicks + 1):
        points, labels = generate_random_click(width, height)
        consecutive_examples.append({
            'name': f'Consecutive Click {i}',
            'points': points,
            'labels': labels,
            'use_previous_logits': True
        })
    
    return consecutive_examples


def generate_text_prompt_examples() -> List[Dict]:
    """Generate text prompt examples with the specified prompts."""
    text_prompts = ['liver', 'kidney', 'atrium']
    text_prompt_examples = []
    
    for i, prompt in enumerate(text_prompts):
        text_prompt_examples.append({
            'name': f'Text Prompt {i+1} ({prompt})',
            'text_prompt': [prompt],
        })
    
    return text_prompt_examples


def create_test_examples(width: int = 512, height: int = 512) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Create test examples for all four methods."""
    
    # Method 1: Three random single clicks
    single_click_examples = []
    for i in range(3):
        points, labels = generate_random_click(width, height)
        single_click_examples.append({
            'name': f'Single Click {i+1}',
            'points': points,
            'labels': labels,
        })
    
    # Method 2: Three random bounding boxes
    bbox_examples = []
    for i in range(3):
        bbox = generate_random_bbox(width, height)
        bbox_examples.append({
            'name': f'Bounding Box {i+1}',
            'bounding_box': bbox,
        })
    
    # Method 3: Three consecutive clicks (using previous logits)
    consecutive_examples = generate_consecutive_clicks(width, height, 3)
    
    # Method 4: Text prompts
    text_prompt_examples = generate_text_prompt_examples()
    
    return single_click_examples, bbox_examples, consecutive_examples, text_prompt_examples


def measure_processing_time(
        session: InteractiveSegmentationSession, image_path: str, examples: List[Dict]) -> float:
    """Measure processing time for a set of examples on an image."""
    start_time = time.time()
    
    # Load image and run examples
    session.load_image(image_path)
    session.run_examples(examples)
    
    end_time = time.time()
    return end_time - start_time


def run_speed_test():
    """Main function to run the inference speed test."""
    print("=== Inference Speed Test ===")
    print("Generating dummy images and measuring processing speed...")
    
    # Initialize configuration and session
    config = parse_args()
    session = InteractiveSegmentationSession(config, output_dir="speed_test_output", save=False)
    
    # Create directory for dummy images
    dummy_dir = Path("dummy_images")
    dummy_dir.mkdir(exist_ok=True)
    
    # Generate 11 dummy images (first one will be ignored)
    image_paths = []
    print("Generating dummy images...")
    for i in range(11):  # 0-10, we'll ignore the first one
        dummy_image = generate_dummy_image()
        image_path = dummy_dir / f"dummy_image_{i:02d}.png"
        save_dummy_image(dummy_image, str(image_path))
        image_paths.append(str(image_path))
    
    # Initialize model with first image (to be ignored)
    print("Initializing model with first image (will be ignored)...")
    init_examples = [{
        'name': 'Initialization',
        'points': np.array([[256, 256]]),
        'labels': np.array([1]),
    }]
    session.process_image_with_examples(image_paths[0], init_examples)
    print("Model initialization complete.")
    
    # Storage for timing results
    image_encoder_times =[]
    single_click_times = []
    bbox_times = []
    consecutive_times = []
    text_prompt_times = []


    # Process remaining 10 images
    print(f"\nProcessing {len(image_paths[1:])} test images...")
    
    for i, image_path in enumerate(image_paths[1:], 1):
        print(f"\nProcessing image {i}/10: {Path(image_path).name}")
        
        # Generate test examples for this image
        single_clicks, bboxes, consecutive, text_prompts = create_test_examples()

        # Vision Transformer Encoder
        _, image_encoder_time = session.load_image(image_path)
        image_encoder_times.append(image_encoder_time)

        # Measure Method 1: Single clicks
        print("  - Testing single clicks...")
        time_single = measure_processing_time(session, image_path, single_clicks)
        single_click_times.append(time_single)

        # Measure Method 2: Bounding boxes
        print("  - Testing bounding boxes...")
        time_bbox = measure_processing_time(session, image_path, bboxes)
        bbox_times.append(time_bbox)
        
        # Measure Method 3: Consecutive clicks
        print("  - Testing consecutive clicks...")
        time_consecutive = measure_processing_time(session, image_path, consecutive)
        consecutive_times.append(time_consecutive)
        
        # Measure Method 4: Text prompts
        print("  - Testing text prompts...")
        time_text_prompt = measure_processing_time(session, image_path, text_prompts)
        text_prompt_times.append(time_text_prompt)
        
        print(f"  Times - Single: {time_single:.3f}s, BBox: {time_bbox:.3f}s, Consecutive: {time_consecutive:.3f}s, Text: {time_text_prompt:.3f}s")
    
    # Calculate averages
    avg_image_encoder = np.mean(image_encoder_times)
    avg_single_click = np.mean(single_click_times)
    avg_bbox = np.mean(bbox_times)
    avg_consecutive = np.mean(consecutive_times)
    avg_text_prompt = np.mean(text_prompt_times)
    
    # Print results
    print("\n" + "="*60)
    print("INFERENCE SPEED TEST RESULTS")
    print("="*60)
    print(f"Number of test images: {len(single_click_times)}")
    print(f"Examples per image per method: 3")
    print(f"Total examples processed: {len(single_click_times) * 4 * 3}")
    print()
    print("Average Processing Times:")
    print(f"  Image Encoder:       {avg_image_encoder:.4f} seconds")
    print(f"  Single Clicks:      {avg_single_click:.4f} seconds")
    print(f"  Bounding Boxes:     {avg_bbox:.4f} seconds")
    print(f"  Consecutive Clicks: {avg_consecutive:.4f} seconds")
    print(f"  Text Prompts:       {avg_text_prompt:.4f} seconds")
    print()
    print("Average Processing Times per Example:")
    print(f"  Image Encoder:       {avg_image_encoder:.4f} seconds")
    print(f"  Single Clicks:      {avg_single_click/3:.4f} seconds per example")
    print(f"  Bounding Boxes:     {avg_bbox/3:.4f} seconds per example")
    print(f"  Consecutive Clicks: {avg_consecutive/3:.4f} seconds per example")
    print(f"  Text Prompts:       {avg_text_prompt/3:.4f} seconds per example")
    print()
    print("Relative Performance:")
    fastest = min(avg_single_click, avg_bbox, avg_consecutive, avg_text_prompt)
    print(f"  Single Clicks:      {avg_single_click/fastest:.2f}x")
    print(f"  Bounding Boxes:     {avg_bbox/fastest:.2f}x")
    print(f"  Consecutive Clicks: {avg_consecutive/fastest:.2f}x")
    print(f"  Text Prompts:       {avg_text_prompt/fastest:.2f}x")
    print("="*60)
    
    # Cleanup dummy images
    print(f"\nCleaning up dummy images from {dummy_dir}...")
    for image_path in image_paths:
        Path(image_path).unlink()
    dummy_dir.rmdir()
    
    return {
        'single_click_avg': avg_single_click,
        'bbox_avg': avg_bbox,
        'consecutive_avg': avg_consecutive,
        'text_prompt_avg': avg_text_prompt,
        'single_click_times': single_click_times,
        'bbox_times': bbox_times,
        'consecutive_times': consecutive_times,
        'text_prompt_times': text_prompt_times
    }

if __name__ == "__main__":
    try:
        results = run_speed_test()
        print("\nSpeed test completed successfully!")
    except Exception as e:
        print(f"\nError during speed test: {e}")
        raise

"""
RTX 2080 Ti Pytorch 2.6 model

Vision Transformer Encoder: 196 ms
Single Click: 17 ms
Bounding Boxe: 16 ms
Text Prompt: 28 ms
"""