# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run CLI
python main.py --input /path/to/images --output /path/to/results [--model vit_b_lm] [--min-area 500] [--numbered]

# Run web UI
streamlit run ui.py

# Install dependencies
pip install -r requirements.txt
```

## Project Overview

Microscopy cell segmentation pipeline using MicroSAM. Processes Incucyte time-lapse images (5-position wells: center, top, bottom, left, right), segments cells, extracts shape features (area, perimeter, eccentricity), and computes plate coverage.

## Architecture

**Entry points:**
- `main.py` — CLI with argparse. Calls `process_image()` per file, combines stats at the end.
- `ui.py` — Streamlit web app. Two tabs: single image processing and batch upload. Produces ZIP output with CSVs, visualizations, and an XLSX with embedded graphs.

**Shared processing pipeline** (`core/processor.py`):
1. `process_image_core()` — central function used by both CLI and UI
2. Image is read (and downscaled by 2×) via `core/image_processing.py`
3. Segmentation via `core/segmentation.py` using `run_automatic_instance_segmentation()` — loads a MicroSAM predictor+decoder (cached globally in `models_loaded` dict), precomputes image embeddings (optionally cached in `EmbeddingCache`, LRU max 30), runs `InstanceSegmentationWithDecoder`
4. Shape features extracted via `analysis/feature_extraction.py` using `regionprops`
5. Cell filtering via `filters/filter.py` — `filter_dataframe(df, boolean_condition, segmentation_mask)` drops cells and zeros out their mask IDs
6. Neighbor counting via `analysis/neighbours.py` — `count_cell_neighbors()` checks dilated overlap between all cell pairs
7. Coverage computed via `analysis/coverage.py` — `compute_coverage()` as pixel ratio
8. Visualization via `visualization/visualization.py` — random-color overlay + optional numbered labels

**Incucyte 5-position stitching** (`utils/utils.py`):
- `extract_incucyte_info()` parses filenames (`VID167_E7_3_02d18h00m` → position, time, plate)
- `sort_images_incucyte()` groups images by `(vid, plate, time)`
- `stitch_segmentations()` in `core/segmentation.py` assembles a 3×3 grid from the 5 tile positions
- `combine_image_statistics()` in `utils.py` merges results per group and computes combined coverage after stitching

**Checkpoints:** Custom MicroSAM fine-tune at `checkpoints/checkpoints/vit_b_lm_incucyte/incucyte_2.pt`

## Key Paths

- `core/processor.py:14` — `process_image_core()`: shared pipeline entry
- `core/segmentation.py:48` — `run_automatic_instance_segmentation()`: MicroSAM predictor+decoder load and inference
- `core/segmentation.py:19` — `stitch_segmentations()`: 5-tile → 3×3 grid assembly
- `filters/filter.py` — `filter_dataframe()`: boolean-mask cell filtering
- `utils/utils.py:10` — `extract_incucyte_info()`: filename parsing
- `checkpoints/` — microsam model checkpoints
