"""
Visualization module for plotting and image visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import regionprops


def visualize_segmentation(original_image, segmentation, random_colors=None):
    """
    Create visualization of the segmentation without drawing the circle
    
    Args:
        original_image: Original input image
        segmentation: Instance segmentation mask
        random_colors: Optional pre-computed random colors
        
    Returns:
        RGB visualization of the segmentation
        Random colors used for visualization
    """
    # Generate random colors for visualization
    max_label = np.max(segmentation) if np.max(segmentation) > 0 else 1
    if random_colors is None:
        random_colors = np.random.randint(0, 255, size=(max_label + 1, 3))
        # Make background black
        random_colors[0] = [0, 0, 0]
    
    # Create RGB segmentation visualization
    segmentation_vis = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
    for label in range(1, max_label + 1):
        mask = segmentation == label
        if np.any(mask):  # Only process if mask contains any True values
            segmentation_vis[mask] = random_colors[label]
    
    return segmentation_vis, random_colors


def add_numbers_to_image(visualize_segmentation_image, filtered_segmentation, features_df): 
    """
    Adds a numbering to each segmented cell

    Args:
        visualize_segmentation_image: RGB visualization image
        filtered_segmentation: Filtered segmentation mask
        features_df: DataFrame with cell features
    """
    # Make a copy to annotate
    vis_with_numbers = visualize_segmentation_image
    
    # Get region properties for the filtered segmentation
    props = regionprops(filtered_segmentation)

    # Map label to centroid for quick lookup
    label_to_centroid = {prop.label: prop.centroid for prop in props}

    # Iterate through the features DataFrame in CSV order
    for idx, row in features_df.iterrows():
        cell_label = row['cell_id']
        if cell_label in label_to_centroid:
            y, x = map(int, label_to_centroid[cell_label])
            # Write the CSV row number (starting from 1) at the centroid
            cv2.putText(
                vis_with_numbers,
                str(int(cell_label)),         # CSV row number (1-based)
                (x, y),               # (x, y) coordinates
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,                  # Font scale
                (255, 255, 255),      # White text
                3,                    # Thickness
                cv2.LINE_AA
            )


def create_combined_visualization(image, segmentation_vis, area_filtered_vis, final_filtered_vis, 
                                segmentation_count, min_area, final_count, output_path):
    """
    Create and save a combined visualization showing original, segmentation, and filtered results.
    
    Args:
        image: Original input image
        segmentation_vis: Visualization of all segmented cells
        area_filtered_vis: Visualization of area-filtered cells
        final_filtered_vis: Visualization of final filtered cells
        segmentation_count: Number of cells in original segmentation
        min_area: Minimum area threshold used
        final_count: Number of cells in final result
        output_path: Path to save the combined visualization
    """
    plt.figure(figsize=(20, 5))
    
    # Original image
    plt.subplot(1, 4, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    # Original segmentation visualization
    plt.subplot(1, 4, 4)
    plt.imshow(segmentation_vis)
    plt.title(f"All Cells ({segmentation_count})")
    plt.axis("off")
    
    # Area-filtered segmentation visualization
    plt.subplot(1, 4, 3)
    plt.imshow(area_filtered_vis)
    plt.title(f"Area Filtered (>{min_area} px²)")
    plt.axis("off")
    
    # Final filtered segmentation visualization
    plt.subplot(1, 4, 2)
    plt.imshow(final_filtered_vis)
    plt.title(f"Final Filtered ({final_count} cells)")
    plt.axis("off")
    
    # Save combined visualization
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
