# Incucyte Osteoclast Segmentation & Analysis

A Streamlit web app and command-line tool for automated segmentation and quantitative analysis of **Incucyte time-lapse microscopy images** (5-position wells: center, top, bottom, left, right). It uses a custom fine-tuned [micro-SAM](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html) model to segment individual cells, extracts shape features, and computes plate coverage — including optional 5-position well stitching.

---

## Features

- **Single image mode** — upload one image and get an annotated overlay + per-cell table.
- **Batch mode** — process many images at once and download a ZIP with overlays, masks, per-image CSVs, and a combined `FINAL_STATS.csv`.
- **Incucyte 5-position stitching** — automatically groups images by `VID_plate_time` key and stitches center/top/bottom/left/right tiles into a 3×3 grid for combined coverage calculation.
- **Adjustable filters** — `Min Area` removes small fragments; optional numbered labels help match cells to the results table.
- **Custom MicroSAM checkpoint** — `vit_b_lm_incucyte` fine-tuned checkpoint loaded by default from `checkpoints/checkpoints/vit_b_lm_incucyte/incucyte_2.pt`.
- **In-app documentation** — open the 📖 Documentation tab in the running app for a walkthrough.

---

## Installation

This project depends on [micro-SAM](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html), which is easiest to install via conda. Follow the official install guide:

👉 **[micro-SAM installation instructions](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html#from-conda)**

After activating the micro-SAM environment, install the remaining dependencies:

```bash
pip install -r requirements.txt
```

The custom MicroSAM checkpoint (`checkpoints/checkpoints/vit_b_lm_incucyte/incucyte_2.pt`) is tracked by **Git LFS**. Download the actual model weights with:

```bash
git lfs pull
```

If the checkpoint remains a small LFS pointer, segmentation will fail with an `invalid load key` error.

Then launch either the web UI or the CLI:

```bash
# Web UI
streamlit run ui.py

# CLI
python main.py --input /path/to/images --output /path/to/results
```

---

## Quick start

### 1. Process a single image

1. Run `streamlit run ui.py`.
2. Open the **🖼 Single Image** tab.
3. Upload a `.png`, `.tif`, `.tiff`, or `.jpg`.
4. Adjust the sidebar parameters (see [Parameters](#parameters)).
5. Click **Process Image** and review the comparison.

### 2. Process a batch

1. Open the **📂 Batch Processing** tab.
2. Select multiple files at once (Ctrl/⌘-click).
3. Click **Process Uploaded Batch** — a progress bar tracks each image.
4. Download the **📦 results ZIP** containing annotated overlays, per-image CSVs, and `FINAL_STATS.csv`.

> **Note:** Segmentation results may vary slightly between different computers due to differences in hardware, floating-point behavior, and dependency versions. Always verify outputs on your own system before drawing conclusions.

---

## Outputs and results

For every processed image the app writes three files into the results folder (and into the batch ZIP):

- `<image>_final_filtered.png` — side-by-side debug visualization showing original image, final filtered cells, and all detected cells.
- `<image>_features.csv` — one row per detected cell with its measured features.
- `<image>.png` (in batch only) — the input image saved alongside the results.

For batches, a fourth file `FINAL_STATS.csv` summarizes every image on one row.

### Per-cell features (`<image>_features.csv`)

Each row corresponds to a single cell that passed all filters.

| Column | Description |
|---|---|
| `cell_id` | Unique numeric label assigned to the cell. |
| `area` | Cell area in pixels. |
| `perimeter` | Cell perimeter in pixels. |

### Batch summary (`FINAL_STATS.csv`)

`FINAL_STATS.csv` contains one summary row per image or per Incucyte well group:

| Column | Description |
|---|---|
| `image_name` | Name of the input image or the Incucyte `VID_plate_time` group key. |
| `num_cells` | Total number of cells that passed all filters. |
| `cells_touching` | Number of cell pairs that are touching (dilated mask overlap). |
| `mean_area` | Average cell area among detected cells (pixels). |
| `mean_perimeter` | Average cell perimeter among detected cells (pixels). |
| `plate_coverage_percent` | Estimated percentage of the plate area covered by cells. |

For Incucyte 5-position filenames (`VID167_E7_3_02d18h00m`), rows are grouped by `VID_plate_time` and coverage is computed on the stitched 3×3 grid. The stitched grid itself is not saved as an image — it only appears in `FINAL_STATS.csv` as the combined `plate_coverage_percent` for the well group.

---

## Recommended workflow

Before running a full batch, **calibrate the parameters on a few representative images** using the Single Image tab:

1. Start with the most permissive value: **`Min Area = 0`** so nothing is dropped.
2. Process a handful of representative images.
3. Manually inspect the annotated overlays — note the smallest *true* cells you want to keep.
4. Set **`Min Area`** just below the smallest true cell you want to keep.
5. Re-run the single-image cases to confirm the filters look correct.
6. Only then switch to **📂 Batch Processing** with the chosen values.

---

## Parameters

All parameters live in the left sidebar of the Streamlit UI and can also be passed via the command line.

| Parameter | Default | Description |
|---|---|---|
| **Min Area** | 100 (CLI) / 500 (UI) | Drop any cell whose area is **lower** than this pixel count. Increase to remove fragments, decrease to keep small cells. |
| **Numbered Labels** | Off (CLI) / On (UI) | Overlay numeric IDs on each detected cell so they can be matched to the results table. |
| **Model Type** | `vit_b_lm` | SAM backbone. The default uses the custom `vit_b_lm_incucyte` fine-tuned checkpoint. |

### CLI options

When running the pipeline from the terminal (`python main.py`), the following flags are available:

| Flag | Default | Description |
|---|---|---|
| `--input` (required) | — | Directory containing input images. |
| `--output` (required) | — | Directory where results and `FINAL_STATS.csv` are saved. |
| `--model` | `vit_b_lm` | MicroSAM model type. Passed to `core/segmentation.py`, which loads the custom checkpoint `checkpoints/checkpoints/vit_b_lm_incucyte/incucyte_2.pt` when available. |
| `--min-area` | 100 | Minimum cell area in pixels². Cells smaller than this are dropped. |
| `--numbered` | False | Add numeric labels to each cell in the output overlay. |

### Tuning notes

- **Min Area**: This is the main control for removing segmentation fragments and noise. Start at `0` while calibrating, identify the smallest true cell you want to keep, then set Min Area just below that value.
- **Numbered Labels**: Adds white numeric IDs at cell centroids. Enable when you need to cross-reference the overlay with the per-cell CSV.
- **Model Type**: The app is designed around a custom `vit_b_lm_incucyte` fine-tuned checkpoint. Only change this if you have a compatible replacement checkpoint.

---

## Incucyte filename format

The app recognizes Incucyte filenames split by underscores into four parts:

```
VID167_E7_3_02d18h00m
│     │  │  │
│     │  │  └── Time (e.g. 02d18h00m)
│     │  └──── Position index (1–5 → top/left/center/right/bottom)
│     └─────── Plate identifier (e.g. E7)
└─────────── VID / experiment identifier
```

Images sharing the same `VID_plate_time` key are grouped and stitched into a 3×3 well view for combined coverage calculation.

### Batch processing requirements

To compute the combined well coverage, **each well/time point must include exactly 5 images**, one per position, using the exact position order encoded in the filename:

| Position index | Position | Tile in 3×3 grid |
|---|---|---|
| 1 | top | top-center |
| 2 | left | middle-left |
| 3 | center | middle-center |
| 4 | right | middle-right |
| 5 | bottom | bottom-center |

For example, a single well/time point requires these five files:

```
VID167_E7_1_02d18h00m   # top
VID167_E7_2_02d18h00m   # left
VID167_E7_3_02d18h00m   # center
VID167_E7_4_02d18h00m   # right
VID167_E7_5_02d18h00m   # bottom
```

If one or more positions are missing for a `VID_plate_time` group, the pipeline still processes the individual images, but the stitched 3×3 combined coverage for that group will be reported as `0%` and the per-image statistics are not merged into a single group row.

---

## Project structure

```
incucyte_osteoclasts/
├── main.py                  # CLI entry point
├── ui.py                    # Streamlit web interface
├── app.py                   # Legacy compatibility layer
├── core/                    # Core processing modules
│   ├── image_processing.py  # Image I/O and 2× downscaling
│   ├── segmentation.py      # MicroSAM inference + 5-position stitching
│   ├── processor.py         # Shared CLI/UI processing logic
│   └── embedding_cache.py   # LRU embedding cache
├── analysis/                # Analysis modules
│   ├── feature_extraction.py # Area and perimeter
│   ├── neighbours.py         # Touching-cell neighbor counts
│   └── coverage.py           # Coverage computation
├── visualization/           # Visualization modules
│   └── visualization.py     # Overlay rendering + numbered labels
├── filters/                 # Cell filtering
│   └── filter.py            # Boolean-mask filter helper
├── utils/                   # Utilities
│   └── utils.py             # Incucyte sorting, stats, stitching orchestration
├── checkpoints/             # Custom micro-SAM fine-tuned checkpoint
└── requirements.txt         # Dependency snapshot
```

---

## License

Released under the [MIT License](LICENSE).
