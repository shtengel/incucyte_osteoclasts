"""
Shared image processing logic for both CLI and UI interfaces.
"""

import os

from .image_processing import read_image, read_image_from_stream
from .segmentation import run_automatic_instance_segmentation
from analysis import extract_shape_features, count_cell_neighbors
from visualization import visualize_segmentation, add_numbers_to_image
from utils import calculate_image_statistics, extract_incucyte_info
from filters.filter import filter_dataframe

def process_image_core(image, filename, model_type="vit_b_lm", min_area=200, numbered=False):
    """
    Core image processing logic shared between CLI and UI.
    
    Args:
        image: Input image array
        filename: Name of the image file (without extension)
        model_type: MicroSAM model type
        min_area: Minimum area threshold for cell filtering
        numbered: Write cell number on each cell
    
    Returns:
        Dictionary with processing results including segmentation, features, and visualizations
    """
    # Run instance segmentation
    try:
        segmentation = run_automatic_instance_segmentation(
            image, 
            model_type=model_type, 
            checkpoint_path="checkpoints/checkpoints/vit_b_lm_incucyte/incucyte_2.pt"
        )
    except Exception as e:
        print(f"Error in segmentation for {filename}", e)
        return None
    
    # Extract shape features and apply area filtering
    features_df, area_filtered_segmentation = extract_shape_features(
        segmentation, image
    )
    
    features_df, area_filtered_segmentation = filter_dataframe(
        features_df, features_df['area'] > min_area, area_filtered_segmentation
    ) 
    
    # Filter cells based on intensity percentile if specified
    filtered_segmentation = area_filtered_segmentation.copy()
    
    # Create visualizations
    segmentation_vis, random_colors = visualize_segmentation(image, segmentation)
    # area_filtered_vis, _ = visualize_segmentation(image, area_filtered_segmentation, random_colors)
    final_filtered_vis, _ = visualize_segmentation(image, filtered_segmentation, random_colors)
    
    if numbered:
        add_numbers_to_image(final_filtered_vis, segmentation, features_df)
    
    adjacency, neighbor_counts, touch_mask = count_cell_neighbors(segmentation.copy())
    # for cell, neigh in adjacency.items():
    #     features_df["cell_id" == cell] = len(neigh)

    # Calculate image statistics
    image_stats = calculate_image_statistics(filtered_segmentation, features_df, neighbor_counts,filename)
    
    # append filtered segmentation to image stats for stitching
    image_stats["filtered_segmentation"] = filtered_segmentation
    
    return {
        'image': image,
        'segmentation': segmentation,
        'area_filtered_segmentation': area_filtered_segmentation,
        'filtered_segmentation': filtered_segmentation,
        'features_df': features_df,
        'image_stats': image_stats,
        'incucyte_info': extract_incucyte_info(os.path.basename(filename)),
        'visualizations': {
            'segmentation_vis': segmentation_vis,
            # 'area_filtered_vis': area_filtered_vis,
            'final_filtered_vis': final_filtered_vis,
            'random_colors': random_colors
        }
    }


def process_image_from_path(image_path, model_type="vit_b_lm", min_area=200, numbered=False):
    """
    Process an image from file path (for CLI usage).
    
    Args:
        image_path: Path to the input image
        model_type: MicroSAM model type
        min_area: Minimum area threshold for cell filtering
        numbered: Write cell number on each cell
    
    Returns:
        Dictionary with processing results or None if error
    """
    # Get image filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Read the image
    image = read_image(image_path)
    if image is None:
        return None
    
    return process_image_core(image, filename, model_type, min_area, numbered)


def process_image_from_stream(image_stream, filename, model_type="vit_b_lm", min_area=200, numbered=False):
    """
    Process an image from stream (for UI usage).
    
    Args:
        image_stream: Stream object containing image data
        filename: Name of the image file (without extension)
        model_type: MicroSAM model type
        min_area: Minimum area threshold for cell filtering
        numbered: Write cell number on each cell
    
    Returns:
        Dictionary with processing results or None if error
    """
    # Read the image from stream
    image = read_image_from_stream(image_stream=image_stream)
    return process_image_core(image, filename, model_type, min_area, numbered)
