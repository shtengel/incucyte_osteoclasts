"""
Feature extraction module for shape and intensity feature analysis.
"""

import numpy as np
import pandas as pd
from skimage.measure import regionprops


def extract_shape_features(segmentation, original_image, plate_mask=None):
    """
    Extract shape features (area, perimeter, and mean intensity) from segmentation mask
    
    Args:
        segmentation: Instance segmentation mask
        original_image: Original input image for intensity measurements
        min_area: Minimum area threshold for cell filtering
        plate_mask: Optional binary mask of the plate region
        
    Returns:
        DataFrame with area, perimeter, and mean intensity for each cell
        Filtered segmentation mask
    """
    # Apply plate mask if provided
    if plate_mask is not None:
        filtered_segmentation = segmentation.copy()
        filtered_segmentation[~plate_mask] = 0
    else:
        filtered_segmentation = segmentation.copy()
    
    # Get region properties for each labeled region
    props = regionprops(filtered_segmentation, intensity_image=original_image)
    
    # Extract area, perimeter, and mean intensity for each cell
    features = []
    for prop in props:
        # Skip background (label 0)
        if prop.label > 0:
            features.append({
                'cell_id': prop.label,
                'area': prop.area,
                'perimeter': prop.perimeter,
                'eccentricity': prop.eccentricity,
            })
    
    return pd.DataFrame(features), filtered_segmentation


def calculate_plate_coverage(features_df, plate_radius=0):
    """
    Calculate the percentage of the plate area covered by cells
    
    Args:
        features_df: DataFrame with cell features
        plate_radius: Radius of the plate in pixels
        
    Returns:
        Percentage of plate area covered by cells
    """
    if features_df.empty:
        return 0.0
    
    # Calculate total cell area
    total_cell_area = features_df['area'].sum()
    
    # Calculate plate area
    plate_area = np.pi * (plate_radius ** 2)
    
    # Calculate coverage percentage
    coverage_percentage = (total_cell_area / plate_area) * 100
    
    return coverage_percentage
