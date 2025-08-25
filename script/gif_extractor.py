"""
GIF Frame Extractor

This script extracts frames from GIF files and saves them as PNG files.
Supports both single file and folder input processing.

Usage:
    python script/gif_extractor.py path/to/file.gif
    python script/gif_extractor.py path/to/folder/
"""

import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np


def extract_frames_opencv(gif_path, output_dir):
    """Extract frames from GIF using OpenCV (alternative method """
    try:
        cap = cv2.VideoCapture(gif_path)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create output filename
            frame_filename = Path(f"frame_{frame_count:04d}.png")
            frame_path = output_dir / frame_filename

            # Save frame as PNG
            cv2.imwrite(frame_path, frame)
            frame_count += 1
            
        cap.release()
        print(f"✓ Extracted {frame_count} frames from {os.path.basename(gif_path)}")
        return frame_count
        
    except Exception as e:
        print(f"✗ Error processing {gif_path}: {str(e)}")
        return 0


def process_single_gif(gif_path):
    """Process a single GIF file"""
    gif_path = Path(gif_path)
    
    if not gif_path.exists():
        print(f"✗ File not found: {gif_path}")
        return
        
    if gif_path.suffix.lower() != '.gif':
        print(f"✗ Not a GIF file: {gif_path}")
        return
    
    # Create output directory
    output_dir = gif_path.parent / f"{gif_path.stem}_frames"
    output_dir.mkdir(exist_ok=True)
    
    print(f"Processing: {gif_path}")
    print(f"Output directory: {output_dir}")
    
    frame_count = extract_frames_opencv(str(gif_path), str(output_dir))
    
    if frame_count > 0:
        print(f"✓ Successfully extracted {frame_count} frames to {output_dir}")


def process_folder(folder_path):
    """Process all GIF files in a folder"""
    folder_path = Path(folder_path)
    
    if not folder_path.exists() or not folder_path.is_dir():
        print(f"✗ Folder not found: {folder_path}")
        return
    
    # Find all GIF files
    gif_files = list(folder_path.glob("*.gif")) + list(folder_path.glob("*.GIF"))
    
    if not gif_files:
        print(f"✗ No GIF files found in {folder_path}")
        return
    
    print(f"Found {len(gif_files)} GIF file(s) in {folder_path}")
    
    total_frames = 0
    processed_files = 0
    
    for gif_file in gif_files:
        # Create output directory for each GIF
        output_dir = folder_path / f"{gif_file.stem}_frames"
        output_dir.mkdir(exist_ok=True)
        
        print(f"\nProcessing: {gif_file.name}")
        print(f"Output directory: {output_dir}")
        
        frame_count = extract_frames_opencv(str(gif_file), str(output_dir))
        
        if frame_count > 0:
            total_frames += frame_count
            processed_files += 1
    
    print(f"\n✓ Processing complete!")
    print(f"  Files processed: {processed_files}/{len(gif_files)}")
    print(f"  Total frames extracted: {total_frames}")


def main():
    parser = argparse.ArgumentParser(description="Extract frames from GIF files and save as PNG images")
    parser.add_argument('input_path', help='Path to GIF file or folder containing GIF files')
    args = parser.parse_args()
    
    input_path = Path(args.input_path)
    
    if not input_path.exists():
        print(f"✗ Path not found: {input_path}")
        sys.exit(1)
    
    if input_path.is_file():
        process_single_gif(input_path)
    elif input_path.is_dir():
        process_folder(input_path)
    else:
        print(f"✗ Invalid path: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()