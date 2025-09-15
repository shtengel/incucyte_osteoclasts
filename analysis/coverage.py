import numpy as np


def compute_coverage(segmentation_mask, valid_mask=None):
    """Return coverage (%) of cells in an image mask.
       If valid_mask is provided, only count those pixels."""
    if valid_mask is None:
        total_pixels = segmentation_mask.size
        cell_pixels = np.count_nonzero(segmentation_mask)
    else:
        total_pixels = np.count_nonzero(valid_mask)
        cell_pixels = np.count_nonzero(segmentation_mask[valid_mask])
        
    return cell_pixels / total_pixels
