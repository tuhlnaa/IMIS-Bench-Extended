"""
Deep Learning Model Weights Downloader

This script downloads pre-trained model weights from Google Drive URLs.
It uses the gdown library to handle Google Drive downloads with proper
authentication and error handling.
"""
import os
import gdown
import logging

from typing import Dict, List
from rich.logging import RichHandler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s", 
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def setup_download_directory(download_dir: str = "model_weights") -> str:
    """Create and return the download directory path."""
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
        logger.info(f"Created download directory: {download_dir}")
    return download_dir


def extract_file_id(url: str) -> str:
    """Extract file ID from Google Drive URL."""
    if "/file/d/" in url:
        return url.split("/file/d/")[1].split("/")[0]
    elif "id=" in url:
        return url.split("id=")[1].split("&")[0]
    else:
        raise ValueError(f"Cannot extract file ID from URL: {url}")


def download_model_weights(models_info: Dict[str, str], download_dir: str) -> List[str]:
    """
    Download model weights from Google Drive URLs.
    
    Args:
        models_info: Dictionary mapping filenames to Google Drive URLs
        download_dir: Directory to save downloaded files
        
    Returns:
        List of successfully downloaded files
    """
    downloaded_files = []
    failed_downloads = []
    
    for filename, url in models_info.items():
        output_path = os.path.join(download_dir, filename)
        
        # Skip if file already exists
        if os.path.exists(output_path):
            logger.info(f"File already exists, skipping: {filename}")
            downloaded_files.append(filename)
            continue
            
        try:
            logger.info(f"Downloading {filename}...")
            
            # Use fuzzy=True to handle Google Drive share URLs
            gdown.download(url, output_path, fuzzy=True, quiet=False)
            
            # Verify the download
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"Successfully downloaded {filename} ({file_size_mb:.2f} MB)")
                downloaded_files.append(filename)
            else:
                logger.error(f"Download failed or file is empty: {filename}")
                failed_downloads.append(filename)
                
        except Exception as e:
            logger.error(f"Error downloading {filename}: {str(e)}")
            failed_downloads.append(filename)
            
            # Clean up incomplete download
            if os.path.exists(output_path):
                os.remove(output_path)
    
    # Summary
    logger.info(f"\nDownload Summary:")
    logger.info(f"Successfully downloaded: {len(downloaded_files)} files")
    if failed_downloads:
        logger.warning(f"Failed downloads: {len(failed_downloads)} files")
        for failed in failed_downloads:
            logger.warning(f"  - {failed}")
    
    return downloaded_files


def main():
    """Main function to download all model weights."""
    
    # Define the models and their Google Drive URLs
    models_info = {
        "vit_weights_only.pth": "https://drive.google.com/file/d/102GYfFC4-Jor6le4Ya-sfq3L2IL2AXkL/view?usp=drive_link",
        "vit_b_weights_only.pth": "https://drive.google.com/file/d/1BJzo4huMaTqbBcv0ZowLGk_noZloHhGN/view?usp=drive_link",
        "non_CLIP_weights.pth": "https://drive.google.com/file/d/1GICMt7ueY00jaE55gJHPqRIR71VTgj6_/view?usp=drive_link",
        "IMISNet-B.pth": "https://drive.google.com/file/d/1rjnKe7W1chgpE_xdjBW1k0os-VMfGUOn/view?usp=drive_link",
        "IMISNet-B_non_CLIP_weights.pth": "https://drive.google.com/file/d/1Y42ulrnD0lRPtASimBujZhfO_EWo03m7/view?usp=drive_link",
        "IMISNet-B_non_CLIP_vit_weights.pth": "https://drive.google.com/file/d/1yXjbD-DFg2iYjMH9gk2Kn1999NIA10bt/view?usp=drive_link",
        "CLIP_weights_only.pth": "https://drive.google.com/file/d/1J4CKhasAPKSah6VxApkThIoXjLxpX0ZS/view?usp=drive_link"
    }
    
    logger.info("Starting model weights download...")
    logger.info(f"Total models to download: {len(models_info)}")
    
    # Setup download directory
    download_dir = setup_download_directory()
    
    # Download all model weights
    downloaded_files = download_model_weights(models_info, download_dir)
    
    logger.info(f"\nDownload process completed!")
    logger.info(f"Files saved to: {os.path.abspath(download_dir)}")
    
    return downloaded_files


if __name__ == "__main__":
    try:
        downloaded_files = main()
        print(f"\nSuccessfully downloaded {len(downloaded_files)} model weight files.")
    except KeyboardInterrupt:
        logger.info("\nDownload interrupted by user.")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise