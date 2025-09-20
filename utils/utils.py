"""
Utility functions for data processing and image sorting.
"""
from pathlib import Path
import os
import re
from analysis import compute_coverage
from core.segmentation import stitch_segmentations

def extract_incucyte_info(name, splitType=5):
    [vid, plateNumber, positionIndex, time] = Path(os.path.basename(name)).stem.split("_")

    position = ""
    if splitType == 5:
        if int(positionIndex) == 1:
            position = "top"
        elif int(positionIndex) == 2:
            position = "left"
        elif int(positionIndex) == 3:
            position = "center"
        elif int(positionIndex) == 4:
            position = "right"
        elif int(positionIndex) == 5:
            position = "bottom"
    return { "key": "%s_%s_%s" % (vid, plateNumber, time), "plateNumber": plateNumber, "position": position, "time": time }


def sort_images_incucyte(images=["VID167_E7_1_02d18h00m", "VID167_E7_3_02d18h00m"], splitType=5):
    grouped = {}

    for img in images:
        info = extract_incucyte_info(img, splitType)
        key = info["key"]

        if key not in grouped:
            grouped[key] = { "files": [] }

        grouped[key]["files"].append(img)
        grouped[key]["results"] = []
        grouped[key][info["position"]] = img

    return grouped



def sort_images_by_group_and_column(images, groups=[("B", "C", "D"), ("E", "F", "G")]):
    """
    Sort images by group and column based on filename pattern.
    
    Args:
        images: List of image dictionaries with 'image_name' key
        groups: List of tuples defining groups of row letters
        
    Returns:
        Sorted list of images
    """
    # Build group priority map: 'B' → (0, 0), 'C' → (0, 1), etc.
    group_priority = {
        letter: (group_idx, letter_idx)
        for group_idx, group in enumerate(groups)
        for letter_idx, letter in enumerate(group)
    }
    
    def parse_image(obj):
        name = obj["image_name"]
        match = re.search(r'_([A-Z])(\d{2})f', name)
        if match:
            row_letter = match.group(1)
            col_number = int(match.group(2))
            return row_letter, col_number
        return None, None  # Fallback for bad format

    def sort_key(obj):
        name = obj["image_name"]
        match = re.search(r'_([A-Z])(\d{2})f', name)
        if match:
            row_letter = match.group(1)
            col_number = int(match.group(2))
            group_info = group_priority.get(row_letter, (float('inf'), float('inf')))
            return (col_number, group_info)
        return (float('inf'), (float('inf'), float('inf')))  # fallback for bad format

    result = []
    for group in groups:
        # Filter images in current group
        group_images = []
        for image in images:
            row_letter, col_number = parse_image(image)
            if row_letter in group:
                group_images.append(image)
                
        # Sort by col_number
        group_images = sorted(group_images, key=sort_key)
        
        # Append sorted images
        result = result + group_images 

    return result


def empty_image_statistics(filename):
    return {
        'image_name': filename,
        'num_cells': 0,
        "cells_touching": 0,
        'mean_eccentricity': 0,
        'mean_area': 0,
        'mean_perimeter': 0,
        'plate_coverage_percent': 0
    }

def calculate_image_statistics(segmentation, features_df, cells_touching, filename):
    """
    Calculate image statistics for final CSV output.
    
    Args:
        features_df: DataFrame with cell features
        filename: Name of the image file
        
    Returns:
        Dictionary with image statistics
    """
    return {
        'image_name': filename,
        'num_cells': len(features_df),
        "cells_touching": cells_touching,
        'mean_eccentricity': features_df['eccentricity'].mean() if not features_df.empty else 0,
        'mean_area': features_df['area'].mean() if not features_df.empty else 0,
        'mean_perimeter': features_df['perimeter'].mean() if not features_df.empty else 0,
        'plate_coverage_percent': compute_coverage(segmentation)
    }

def combine_image_statistics(image_stats):
    grouped = {}
    for stats in image_stats:
        info = extract_incucyte_info(stats["image_name"])
        if not info["key"] in grouped:
            grouped[info["key"]] = []
        stats["position"] = info["position"]
        grouped[info["key"]].append(stats)
    
    results = []
    for key, statsList in grouped.items():

        if len(statsList) < 5:
            conbined_computed_coverage = 0
        else:
            top = next((p for p in statsList if p["position"] == "top"), None)
            left = next((p for p in statsList if p["position"] == "left"), None)
            right = next((p for p in statsList if p["position"] == "right"), None)
            bottom = next((p for p in statsList if p["position"] == "bottom"), None)
            center = next((p for p in statsList if p["position"] == "center"), None)

            stitched_segmentations, valid_mask = stitch_segmentations(center=center["filtered_segmentation"], top=top["filtered_segmentation"], bottom=bottom["filtered_segmentation"], left=left["filtered_segmentation"], right=right["filtered_segmentation"])
            conbined_computed_coverage = compute_coverage(stitched_segmentations, valid_mask)

        results.append({
            "image_name": key,
            'num_cells': sum([item['num_cells'] for item in statsList]),
            "cells_touching": sum([item['cells_touching'] for item in statsList]),
            'mean_eccentricity': sum([item['mean_eccentricity'] for item in statsList]) / len(statsList),
            'mean_area': sum([item['mean_area'] for item in statsList]) / len(statsList),
            'mean_perimeter': sum([item['mean_perimeter'] for item in statsList]) / len(statsList),
            'plate_coverage_percent': conbined_computed_coverage
        })
    return results