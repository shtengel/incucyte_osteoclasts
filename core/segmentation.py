"""
Segmentation module for MicroSAM operations and instance segmentation.
"""
import os
import time
import numpy as np
from micro_sam.automatic_segmentation import get_predictor_and_segmenter, automatic_instance_segmentation

from micro_sam.instance_segmentation import (
    InstanceSegmentationWithDecoder,
    get_predictor_and_decoder,
)
from micro_sam.util import precompute_image_embeddings
from .embedding_cache import EmbeddingCache

models_loaded = {}
embeddings_cache = EmbeddingCache(max_items=30)


def _validate_checkpoint(checkpoint_path):
    """Ensure the checkpoint file exists and is not a Git LFS pointer."""
    if checkpoint_path is None:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "If the file is tracked by Git LFS, run `git lfs pull` to download it."
        )
    with open(checkpoint_path, "rb") as f:
        header = f.read(64)
    if header.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"Checkpoint file {checkpoint_path} is a Git LFS pointer, not the actual model weights. "
            "Run `git lfs pull` to download the real checkpoint, or replace it with the model weights."
        )


def stitch_segmentations(center=None, top=None, bottom=None, left=None, right=None):
    # find first non-None tile to determine dimensions
    ref = next((x for x in [center, top, bottom, left, right] if x is not None), None)
    if ref is None:
        # all are None -> return default
        return np.zeros((1, 1), dtype=np.int32), np.zeros((1, 1), dtype=bool)
    
    H, W = ref.shape

    stitched = np.zeros((3*H, 3*W), dtype=np.int32)
    valid_mask = np.zeros((3*H, 3*W), dtype=bool)

    def place(tile, y1, y2, x1, x2):
        if tile is None:
            return
        if tile.shape != (H, W):
            raise ValueError(f"Expected tile shape {(H, W)}, got {tile.shape}")
        stitched[y1:y2, x1:x2] = tile
        valid_mask[y1:y2, x1:x2] = True

    # Place tiles
    place(center, H, 2*H, W, 2*W)   # center
    place(top, 0, H, W, 2*W)        # top
    place(bottom, 2*H, 3*H, W, 2*W) # bottom
    place(left, H, 2*H, 0, W)       # left
    place(right, H, 2*H, 2*W, 3*W)  # right

    return stitched, valid_mask

def run_automatic_instance_segmentation(image, model_type="vit_b_lm", checkpoint_path=None, cache_embeddings=False):
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
    _validate_checkpoint(checkpoint_path)
    start_time = time.perf_counter()

    # Load predictor + decoder (cached globally)
    if key in models_loaded:
        predictor, decoder = models_loaded[key]
        predictor._features = None
        predictor.reset_image()
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
    prediction = ais.generate(output_mode="instance_segmentation").astype(np.uint32)
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