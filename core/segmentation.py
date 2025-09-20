"""
Segmentation module for MicroSAM operations and instance segmentation.
"""
import time
import numpy as np
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    get_predictor_and_decoder,
    mask_data_to_segmentation
)
from micro_sam.util import precompute_image_embeddings
from .embedding_cache import EmbeddingCache

models_loaded = {}
embeddings_cache = EmbeddingCache(max_items=30)

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

def run_automatic_instance_segmentation(image, model_type="vit_b_lm", checkpoint_path=None, cache_embeddings=True):
    """
    Optimized Automatic Instance Segmentation with µsam.

    Args:
        image: The input image (array or filepath).
        model_type: µsam model type (e.g. "vit_b_lm").
        checkpoint_path: Optional checkpoint file path.
        cache_embeddings: If True, cache precomputed image embeddings for reuse.

    Returns:
        Segmentation prediction (numpy array).
    """
    key = f"{model_type}.{checkpoint_path}"
    start_time = time.perf_counter()

    # Load predictor + decoder (cached globally)
    if key in models_loaded:
        predictor, decoder = models_loaded[key]
    else:
        predictor, decoder = get_predictor_and_decoder(
            model_type=model_type,
            checkpoint_path=checkpoint_path,
        )
        models_loaded[key] = (predictor, decoder)

    print(f"get_predictor_and_decoder time: {(time.perf_counter() - start_time) * 1000:.3f} ms")

    # Compute / reuse embeddings
    start_time = time.perf_counter()
    img_key = (key, id(image))  # unique key per model+image
    if cache_embeddings and img_key in embeddings_cache:
        image_embeddings = embeddings_cache[img_key]
    else:
        image_embeddings = precompute_image_embeddings(
            predictor=predictor,
            input_=image,
            ndim=2,
        )
        if cache_embeddings:
            embeddings_cache[img_key] = image_embeddings
    print(f"precompute_image_embeddings time: {(time.perf_counter() - start_time) * 1000:.3f} ms")

    # Run instance segmentation using cached embeddings
    start_time = time.perf_counter()
    ais = InstanceSegmentationWithDecoder(predictor, decoder)
    ais.initialize(image=image, image_embeddings=image_embeddings)
    prediction = ais.generate()
    prediction = mask_data_to_segmentation(prediction, with_background=True)
    print(f"InstanceSegmentationWithDecoder time: {(time.perf_counter() - start_time) * 1000:.3f} ms")

    return prediction


# def run_automatic_instance_segmentation(image, model_type="vit_b_lm", checkpoint_path=None):
#     """
#     Automatic Instance Segmentation by training an additional instance decoder in SAM.

#     Args:
#         image: The input image.
#         model_type: The choice of the `µsam` model.
#         checkpoint_path: Path to checkpoint file

#     Returns:
#         The instance segmentation.
#     """
#     predictor, segmenter = get_predictor_and_segmenter(
#         model_type=model_type,  # choice of the Segment Anything model
#         checkpoint=checkpoint_path,  # overwrite to pass your own finetuned model.
#     )

#     # Step 2: Get the instance segmentation for the given image.
#     prediction = automatic_instance_segmentation(
#         predictor=predictor,  # the predictor for the Segment Anything model.
#         segmenter=segmenter,  # the segmenter class responsible for generating predictions.
#         input_path=image,  # the filepath to image or the input array for automatic segmentation.
#         ndim=2,  # the number of input dimensions.
#     )

#     return prediction