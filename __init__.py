"""
Incucyte App - Cell segmentation and analysis using MicroSAM.

This package provides modules for:
- Image processing and I/O operations
- MicroSAM-based cell segmentation
- Feature extraction and analysis
- Visualization and plotting
- Batch processing (CLI/server-ready)
- Utility functions for data processing
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main functions for easy access
from .main import process_directory, process_image
from .batch_processor import process_batch
from .core import read_image, get_image_files, run_automatic_instance_segmentation, process_image_core, process_image_from_path, process_image_from_stream
from .analysis import extract_shape_features, calculate_plate_coverage
from .visualization import visualize_segmentation, add_numbers_to_image, create_combined_visualization
from .utils import sort_images_by_group_and_column, calculate_image_statistics

__all__ = [
    'process_directory',
    'process_image',
    'process_batch',
    'read_image',
    'get_image_files',
    'run_automatic_instance_segmentation',
    'process_image_core',
    'process_image_from_path',
    'process_image_from_stream',
    'extract_shape_features',
    'calculate_plate_coverage',
    'visualize_segmentation',
    'add_numbers_to_image',
    'create_combined_visualization',
    'sort_images_by_group_and_column',
    'calculate_image_statistics'
]
