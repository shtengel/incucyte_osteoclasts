"""
Main module for CLI interface and orchestration of the cell segmentation pipeline.
"""

import os
import argparse
from time import time
from tqdm import tqdm
import pandas as pd

from core import read_image, get_image_files, process_image_from_path
from visualization import create_combined_visualization
from utils import sort_images_by_group_and_column, sort_images_incucyte, extract_incucyte_info, combine_image_statistics, empty_image_statistics


def process_image(image_path, output_dir, model_type="vit_b_lm", min_area=200, numbered=False):
    """
    Process a single image with MicroSAM and save outputs
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs
        model_type: MicroSAM model type
        min_area: Minimum area threshold for cell filtering
        numbered: Write cell number on each cell
    
    Returns:
        Dictionary with image statistics for final CSV
    """
    # Extract results from shared processing
    filename = os.path.splitext(os.path.basename(image_path))[0]

    # Use shared processing logic
    result = process_image_from_path(image_path, model_type, min_area, numbered)
    if result is None:
        return empty_image_statistics(filename=filename), pd.DataFrame(), None, None, extract_incucyte_info(os.path.basename(filename))
    
    image = result['image']
    segmentation = result['segmentation']
    filtered_segmentation = result['filtered_segmentation']
    features_df = result['features_df']
    image_stats = result['image_stats']
    visualizations = result['visualizations']
    
    # Create output paths
    os.makedirs(output_dir, exist_ok=True)
    final_filtered_output_path = os.path.join(output_dir, f"{filename}_final_filtered.png")
    features_output_path = os.path.join(output_dir, f"{filename}_features.csv")
    
    # Create and save combined visualization
    create_combined_visualization(
        image, 
        visualizations['segmentation_vis'], 
        visualizations['area_filtered_vis'], 
        visualizations['final_filtered_vis'],
        int(segmentation.max()), 
        min_area, 
        len(features_df), 
        final_filtered_output_path
    )
    
    # Save features CSV
    features_df.to_csv(features_output_path, index=False)
    
    return image_stats, features_df, segmentation, filtered_segmentation, result["incucyte_info"]


def process_directory(input_dir, output_dir, model_type="vit_b_lm", min_area=200, numbered=False):
    """
    Process all images in a directory with MicroSAM with progress bar.

    Incucyte batches require 5 images per well/time point, named with the
    underscore format VID_plate_position_time where position is 1-5
    (top, left, center, right, bottom). Missing positions disable combined
    3x3 stitching for that group.

    Args:
        input_dir: Directory containing input images
        output_dir: Directory to save outputs
        model_type: MicroSAM model type
        min_area: Minimum area threshold for cell filtering
        numbered: Show cells with numbers

    Returns:
        DataFrame with final statistics or None if no images processed
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files in input directory
    image_files = get_image_files(input_dir)

    incucyte_group = sort_images_incucyte(images=image_files)
    print(f"Found {len(image_files)} images in {input_dir}")
    print(f"Area filtering: Enabled (min area = {min_area} pixels²)")
    print(
        "Incucyte batches require 5 positions per well/time point (1=top, 2=left, "
        "3=center, 4=right, 5=bottom); missing positions disable 3x3 stitching."
    )

    # Process each image and collect statistics with progress bar
    all_image_stats = []
    
    for image_path in tqdm(image_files, desc="Processing images", unit="image"):
        print(f"\nProcessing {os.path.basename(image_path)}")
        image_stats, features_df, segmentation, _, incucyte_info = process_image(image_path, output_dir, model_type, min_area, numbered)
        if image_stats:
            incucyte_group[incucyte_info["key"]]["results"].append(tuple([incucyte_info["position"], image_stats, features_df, segmentation]))
            all_image_stats.append(image_stats)
    
    final_combined_stats = combine_image_statistics(all_image_stats)

    # Create final statistics CSV
    if final_combined_stats:
        stats_df = pd.DataFrame(final_combined_stats)
        final_csv_path = os.path.join(output_dir, "FINAL_STATS.csv")
        stats_df.to_csv(final_csv_path, index=False)

        print(f"\nSaved final statistics to {final_csv_path}")

        # Print summary
        print("\nSummary:")
        print(f"Total images processed: {len(all_image_stats)}")
        print(f"Average cells per image: {stats_df['num_cells'].mean():.2f}")
        print(f"Average plate coverage: {stats_df['plate_coverage_percent'].mean():.2f}%")

        return stats_df
    elif image_files:
        print("No images were successfully processed.")
        return None
    else:
        return None


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Process directory of images with MicroSAM")
    parser.add_argument("--input", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--output", type=str, required=True, help="Output directory for results")
    parser.add_argument("--model", type=str, default="vit_b_lm", 
                       help="MicroSAM model type (vit_b_lm, vit_h_lm, etc.)")
    parser.add_argument("--min-area", type=int, default=100,
                       help="Minimum cell area in pixels² for filtering small objects")
    parser.add_argument("--numbered", action="store_true", help="Show cells with numbers")
    
    args = parser.parse_args()
    
    start_time = time()
    process_directory(args.input, args.output, args.model, args.min_area, args.numbered)
    end_time = time()
    
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")


if __name__ == "__main__":
    main()
