"""
Image processing module for handling image I/O and basic operations.
"""

import os
import glob
import imageio
import numpy as np
import cv2


def read_image(image_path):
    """
    Read an image from file path with error handling.
    
    Args:
        image_path: Path to the input image
        
    Returns:
        Image array or None if error occurs
    """
    try:
        image = imageio.imread(image_path)
        return image
    except Exception as e:
        print(f"Error reading {os.path.basename(image_path)}: {e}")
        return None


def get_image_files(input_dir):
    """
    Get all image files from a directory.
    
    Args:
        input_dir: Directory containing input images
        
    Returns:
        List of image file paths
    """
    image_extensions = ["*.jpg", "*.jpeg", "*.png", "*.TIF", "*.tiff", "*.tif"]
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    return image_files