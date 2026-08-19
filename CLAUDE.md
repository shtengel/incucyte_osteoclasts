# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run CLI
conda activate micro-sam
python main.py --input /path/to/images --output /path/to/results [--model vit_b_lm] [--min-area 100] [--numbered]

# Run web UI
streamlit run ui.py

# Install dependencies
pip install -r requirements.txt
```

## Project Overview

Microscopy cell segmentation pipeline using MicroSAM. Processes Incucyte time-lapse images (5-position wells: center, top, bottom, left, right), segments cells, extracts shape features (area and perimeter), counts touching-cell neighbors, and computes plate coverage — including combined coverage after stitching the 5 tile positions into a 3×3 grid.

## Architecture

**Entry points:**
- `main.py` — CLI with argparse. Calls `process_image()` per file, combines stats at the end.
- `ui.py` — Streamlit web app. Three tabs: single image processing, batch upload, and in-app documentation.

**Shared processing pipeline** (`core/processor.py:14`):
1. `process_image_core()` — central function used by both CLI and UI.
2. Image is read (and downscaled by 2×) via `core/image_processing.py:read_image()` / `read_image_from_stream()`.
3. Segmentation via `core/segmentation.py:run_automatic_instance_segmentation()` — loads a MicroSAM predictor+decoder (cached globally in `models_loaded` dict), precomputes image embeddings (optionally cached in `EmbeddingCache`, LRU max 30), runs `InstanceSegmentationWithDecoder`. The default custom checkpoint is `checkpoints/checkpoints/vit_b_lm_incucyte/incucyte_2.pt`.
4. Shape features extracted via `analysis/feature_extraction.py:extract_shape_features()` using `regionprops`.
5. Cell filtering via `filters/filter.py:filter_dataframe()` — drops cells and zeros out their mask IDs.
6. Neighbor counting via `analysis/neighbours.py:count_cell_neighbors()` — checks dilated overlap between all cell pairs.
7. Coverage computed via `analysis/coverage.py:compute_coverage()` as pixel ratio.
8. Visualization via `visualization/visualization.py:visualize_segmentation()` — random-color overlay + optional numbered labels.

**Incucyte 5-position stitching** (`utils/utils.py`):
- `extract_incucyte_info()` parses filenames (`VID167_E7_3_02d18h00m` → position, time, plate).
- `sort_images_incucyte()` groups images by `(vid, plate, time)`.
- `stitch_segmentations()` in `core/segmentation.py` assembles a 3×3 grid from the 5 tile positions.
- `combine_image_statistics()` in `utils/utils.py` merges results per group and computes combined coverage after stitching.

**Checkpoints:** Custom MicroSAM fine-tune at `checkpoints/checkpoints/vit_b_lm_incucyte/incucyte_2.pt`.

## Key Paths

- `core/processor.py:14` — `process_image_core()`: shared pipeline entry.
- `core/processor.py:86` — `process_image_from_path()`: CLI wrapper.
- `core/processor.py:110` — `process_image_from_stream()`: UI wrapper.
- `main.py:16` — `process_image()`: CLI per-file orchestration.
- `main.py:68` — `process_directory()`: CLI batch entry.
- `ui.py:15` — `process_image_for_ui()`: Streamlit single-image wrapper.
- `ui.py:88` — `process_uploaded_files()`: Streamlit batch wrapper.
- `core/segmentation.py:48` — `run_automatic_instance_segmentation()`: MicroSAM predictor+decoder load and inference.
- `core/segmentation.py:19` — `stitch_segmentations()`: 5-tile → 3×3 grid assembly.
- `analysis/feature_extraction.py:10` — `extract_shape_features()`: area and perimeter extraction.
- `analysis/neighbours.py` — `count_cell_neighbors()`: touching-neighbor detection.
- `analysis/coverage.py` — `compute_coverage()`: plate coverage computation.
- `filters/filter.py` — `filter_dataframe()`: boolean-mask cell filtering.
- `utils/utils.py:10` — `extract_incucyte_info()`: filename parsing.
- `utils/utils.py:28` — `sort_images_incucyte()`: Incucyte image grouping.
- `utils/utils.py:134` — `combine_image_statistics()`: group-level stitching and stats aggregation.
