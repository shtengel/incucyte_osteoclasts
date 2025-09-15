"""
Segmentation module for MicroSAM operations and instance segmentation.
"""

import numpy as np
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation


def stitch_segmentations(center, top, bottom, left, right):
    H, W = center.shape
    stitched = np.zeros((3*H, 3*W), dtype=np.int32)
    valid_mask = np.zeros((3*H, 3*W), dtype=bool)

    # Place tiles
    stitched[H:2*H, W:2*W] = center
    valid_mask[H:2*H, W:2*W] = True

    stitched[0:H, W:2*W] = top
    valid_mask[0:H, W:2*W] = True

    stitched[2*H:3*H, W:2*W] = bottom
    valid_mask[2*H:3*H, W:2*W] = True

    stitched[H:2*H, 0:W] = left
    valid_mask[H:2*H, 0:W] = True

    stitched[H:2*H, 2*W:3*W] = right
    valid_mask[H:2*H, 2*W:3*W] = True

    return stitched, valid_mask

def run_automatic_instance_segmentation(image, model_type="vit_b_lm", checkpoint_path=None):
    """
    Automatic Instance Segmentation by training an additional instance decoder in SAM.

    Args:
        image: The input image.
        model_type: The choice of the `µsam` model.
        checkpoint_path: Path to checkpoint file

    Returns:
        The instance segmentation.
    """
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,  # choice of the Segment Anything model
        checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
    )

    # Step 2: Get the instance segmentation for the given image.
    prediction = automatic_instance_segmentation(
        predictor=predictor,  # the predictor for the Segment Anything model.
        segmenter=segmenter,  # the segmenter class responsible for generating predictions.
        input_path=image,  # the filepath to image or the input array for automatic segmentation.
        ndim=2,  # the number of input dimensions.
    )

    return prediction
