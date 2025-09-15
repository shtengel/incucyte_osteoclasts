"""
Legacy app.py - This file now imports from the modular structure.
For new development, use the individual modules or main.py directly.
"""

# Import all functionality from the new modular structure
from main import process_directory, process_image, main
from core import read_image, get_image_files, run_automatic_instance_segmentation
from analysis import extract_shape_features, calculate_plate_coverage
from visualization import visualize_segmentation, add_numbers_to_image, create_combined_visualization
from utils import sort_images_by_group_and_column, calculate_image_statistics



# All functions have been moved to their respective modules
# This file now serves as a compatibility layer



# Legacy compatibility - all functions are now imported from their respective modules
# Use main.py for new development or import specific functions as needed