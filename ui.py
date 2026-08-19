import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import math
import pandas as pd
import tempfile
import os
import shutil
from core import process_image_from_stream, process_image_from_path
from utils import extract_incucyte_info, empty_image_statistics
from batch_processor import process_batch as _run_batch

# --- UI-specific processing function ---
def process_image_for_ui(image_path, image_stream=None, output_dir=None, model_type="vit_b_lm", min_area=200, numbered=False):
    """
    Process a single image for UI display, returning visualization arrays and features.
    
    Args:
        image_path: Path to the input image or filename
        image_stream: Optional stream object for uploaded files
        output_dir: Directory to save outputs (optional for UI)
        model_type: MicroSAM model type
        min_area: Minimum area threshold for cell filtering
        numbered: Write cell number on each cell
    
    Returns:
        Tuple of (visualization_arrays, image_stats, titles, features_df)
    """
    # Get image filename without extension
    filename = os.path.splitext(os.path.basename(image_path))[0]
    
    # Use shared processing logic
    if image_stream is not None:
        result = process_image_from_stream(image_stream, filename, model_type, min_area, numbered)
    else:
        # For file paths, we need to use the path-based function
        result = process_image_from_path(image_path, model_type, min_area, numbered)
    
    if result is None:
        return None, empty_image_statistics(filename=filename), None, pd.DataFrame(), None, extract_incucyte_info(filename)
    
    # Extract results from shared processing
    segmentation = result['segmentation']
    features_df = result['features_df']
    image_stats = result['image_stats']
    visualizations = result['visualizations']
    
    # Prepare visualization arrays for UI
    vis_arrays = [
        result['image'], 
        visualizations['final_filtered_vis'], 
        # visualizations['area_filtered_vis'], 
        visualizations['segmentation_vis']
    ]
    titles = [
        "Original Image",
        f"Final Filtered ({len(features_df)} cells)",
        # f"Area Filtered (>{min_area} px²)",
        f"All Cells ({np.max(segmentation)})"
    ]
    
    return vis_arrays, image_stats, titles, features_df, result['segmentation'], result["incucyte_info"]

# --- Utility: display images ---
def display_image_batch(images, titles=None, columns=3):
    num_images = len(images)
    rows = math.ceil(num_images / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(columns * 4, rows * 4))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < num_images:
            img = images[i]
            if img.ndim == 2:
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            ax.axis("off")
            if titles:
                ax.set_title(titles[i], fontsize=12)
        else:
            ax.remove()
    st.pyplot(fig)

# --- Utility: process batch of uploaded files (thin Streamlit wrapper) ---
def process_uploaded_files(uploaded_files, model_type="vit_b_lm", min_area=200, numbered=False):
    # Save uploaded files to a temporary input directory
    tmp_input_dir = tempfile.mkdtemp()
    for file in uploaded_files:
        save_path = os.path.join(tmp_input_dir, file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())

    try:
        # Delegate all processing to the standalone batch_processor module
        zip_path, stats_df = _run_batch(
            input_source=tmp_input_dir,
            output_dir=os.path.join(tmp_input_dir, "results"),
            model_type=model_type,
            min_area=min_area,
            numbered=numbered,
        )

        if zip_path is None:
            return None, None

        # Copy the ZIP to a persistent temp file (Streamlit needs it after function returns)
        final_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
        shutil.copy(zip_path, final_zip.name)
        return final_zip.name, stats_df
    finally:
        shutil.rmtree(tmp_input_dir, ignore_errors=True)

# --- Streamlit layout ---
st.set_page_config(layout="wide")
st.title("Image Processing App")

allowed_extensions = ["png", "tiff", "tif", "jpeg", "jpg"]
results_df = pd.DataFrame(columns=["Cell ID", "Area", "Confidence"])

# --- Sidebar Parameters ---
st.sidebar.header("Processing Parameters")
min_area = st.sidebar.number_input("Min Area", min_value=0, value=500)
st.sidebar.caption("Filter out small cells by pixel size")
numbered = st.sidebar.checkbox("Numbered Labels", value=True)
model_type = st.sidebar.selectbox("Model Type", ["vit_b_lm"])
model_desc = {
    "vit_b_lm": "ViT-B: Base model with good balance between speed and accuracy"
}
st.sidebar.caption(model_desc[model_type])

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🖼 Single Image", "📂 Batch Processing", "📖 Documentation"])

if "zip_path" not in st.session_state:
    st.session_state.zip_path = None
if "batch_stats_df" not in st.session_state:
    st.session_state.batch_stats_df = None

# --- Tab 3: Documentation ---
with tab3:
    st.subheader("Documentation & Guides")

    ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")

    def show_image(filename, caption=None, width=None):
        """Render an image from the assets/ folder if it exists, else show a notice."""
        path = os.path.join(ASSETS_DIR, filename)
        if os.path.isfile(path):
            st.image(path, caption=caption, width=width)
        else:
            st.info(f"📷 Add screenshot at `assets/{filename}` to display here.")

    section = st.radio(
        "Jump to section:",
        ["Overview", "Recommended Workflow", "Parameters Guide", "Single Image Tutorial", "Batch Processing Tutorial", "Output Files"],
        horizontal=True,
    )

    if section == "Overview":
        st.markdown("""
        ### What this tool does
        This app automatically segments and analyzes **Incucyte time-lapse microscopy images** of cell-culture wells.

        It uses a custom fine-tuned [micro-SAM](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html)
        model (`vit_b_lm_incucyte`) to:

        - Segment individual cells in each image.
        - Extract per-cell **area** and **perimeter**.
        - Count **touching-cell neighbors**.
        - Compute **plate coverage** per image.
        - For Incucyte 5-position filenames, stitch center/top/bottom/left/right tiles into a 3×3 grid and compute combined coverage.

        **Intended users:** researchers running Incucyte live-cell assays who need consistent, quantitative segmentation across many images and time points.
        """)
        show_image("overview_example.png", caption="Example: segmented cells overlaid on an Incucyte image.")

    elif section == "Recommended Workflow":
        st.markdown("""
        ### Recommended workflow: calibrate before batching

        Before running a full batch, **calibrate the parameters on a few representative images** using the 🖼 Single Image tab:

        1. Start with the most permissive value: **`Min Area = 0`** so nothing is dropped.
        2. Process a handful of representative images.
        3. Manually inspect the annotated overlays — note the smallest *true* cells you want to keep.
        4. Set **`Min Area`** just **below** the smallest true cell you want to keep.
        5. Re-run the same single images to confirm the filters look correct.
        6. Only then switch to **📂 Batch Processing** with the chosen values.
        """)
        show_image("workflow_calibration_area.png", caption="Calibrate Min Area: keep true cells while removing small fragments.")

    elif section == "Parameters Guide":
        st.markdown("""
        ### Parameters guide
        All parameters live in the left sidebar.

        #### Min Area
        Filter cells by pixel size. Any cell with an area **lower** than this value is dropped.
        Use a higher value to remove small fragments; use a lower value to keep small cells.

        #### Numbered Labels
        Overlay numeric IDs on each detected cell so you can match them to the results table.

        #### Model Type
        The default is `vit_b_lm`, which loads the custom `vit_b_lm_incucyte` fine-tuned checkpoint.
        Only change this if you have a compatible replacement model/checkpoint.
        """)
        show_image("parameters_sidebar.png", caption="The parameter sidebar.", width=400)

    elif section == "Single Image Tutorial":
        st.markdown("""
        ### Single image tutorial

        1. Open the **🖼 Single Image** tab.
        2. Click *Choose a single image file* and select a `.png`, `.tif`, `.tiff`, or `.jpg`.
        3. Adjust the parameters in the sidebar if needed.
        4. Click **Process Image**.
        5. Review the comparison: your uploaded image and the annotated overlay.
        6. Expand **📊 Show Results Table** to see per-cell measurements and the number of touching cells.
        """)
        show_image("tutorial_single_upload.png", caption="Step 2: Upload a single image.")
        show_image("tutorial_single_output.png", caption="Step 5: Original vs. processed output.")

    elif section == "Batch Processing Tutorial":
        st.markdown("""
        ### Batch processing tutorial

        1. Open the **📂 Batch Processing** tab.
        2. Click *Upload multiple image files from a folder* and select many files at once (Ctrl/⌘-click).
        3. Click **Process Uploaded Batch** — a progress bar will update as each image runs.
        4. When complete, click **📦 Download Results (ZIP)**.
        5. The ZIP contains annotated overlays, per-image feature CSVs, and a `FINAL_STATS.csv` summary table.

        #### What is in `FINAL_STATS.csv`?
        - `image_name`: input image name or Incucyte `VID_plate_time` group key.
        - `num_cells`: cells that passed the Min Area filter.
        - `cells_touching`: touching-cell neighbor pairs.
        - `mean_area`: average cell area in pixels.
        - `mean_perimeter`: average cell perimeter in pixels.
        - `plate_coverage_percent`: estimated plate area covered by cells.

        For Incucyte 5-position filenames (e.g. `VID167_E7_3_02d18h00m`), images are grouped by `VID_plate_time`, stitched into a 3×3 grid, and coverage is computed on the combined well view.

        > **Note:** Segmentation results may vary slightly between different computers due to hardware differences, floating-point behavior, and dependency versions. Always verify outputs on your own system.
        """)
        show_image("tutorial_batch_upload.png", caption="Step 2: Select multiple files.")
        show_image("tutorial_batch_results.png", caption="Step 4: Download the results ZIP.")

    elif section == "Output Files":
        st.markdown("""
        ### Output files

        Every processed image generates the following files in the results folder (and in the batch ZIP):

        - `<image>_final_filtered.png` — side-by-side debug visualization showing the original image, final filtered cells, and all detected cells.
        - `<image>_features.csv` — per-cell measurements.
        - `<image>.png` — the input image saved alongside results (batch mode only).

        Batches also include `FINAL_STATS.csv`, which contains one summary row per image or per stitched Incucyte well group.

        #### Per-cell features (`<image>_features.csv`)

        Each row is one cell that passed the Min Area filter.

        | Column | Description |
        |---|---|
        | `cell_id` | Numeric label of the cell. |
        | `area` | Cell area in pixels. |
        | `perimeter` | Cell perimeter in pixels. |

        #### Batch summary (`FINAL_STATS.csv`)

        One summary row per image or well group.

        | Column | Description |
        |---|---|
        | `image_name` | Input image file name or Incucyte group key. |
        | `num_cells` | Cells that passed the Min Area filter. |
        | `cells_touching` | Touching-cell neighbor pairs. |
        | `mean_area` | Mean cell area in pixels. |
        | `mean_perimeter` | Mean cell perimeter in pixels. |
        | `plate_coverage_percent` | Estimated plate area covered by cells.|
        """)
        show_image("features.png", caption="Example per-cell features CSV opened in a spreadsheet.")
        show_image("final_stats.png", caption="Example FINAL_STATS.csv opened in a spreadsheet.")

# --- Tab 1: Single Image ---
with tab1:
    st.subheader("Single Image Processing")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        uploaded_file = st.file_uploader("Choose a single image file", type=allowed_extensions, key="single")
        process_clicked = st.button("Process Image")

    with col_right:
        if uploaded_file:
            uploaded_image_preview = Image.open(uploaded_file)
            st.image(uploaded_image_preview, width=150, caption="Preview")

    if uploaded_file and process_clicked:
        with st.spinner("Processing image..."):
            uploaded_file.seek(0)  # reset pointer
            image_bytes = uploaded_file.read()
            process_stream = io.BytesIO(image_bytes)

            processed_img_array, result_dict, titles, features_df, segmentation, incucyte_info = process_image_for_ui(
                image_path=uploaded_file.name,
                image_stream=process_stream,
                output_dir=None,
                model_type=model_type,
                min_area=min_area,
                numbered=numbered
            )

        if features_df is None:
            features_df = pd.DataFrame()
            result_dict = {"cells_touching": 0}

        st.subheader("Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image_preview, caption="Uploaded Image", use_container_width=True)
        with col2:
            st.subheader("Processed Image")
            if processed_img_array is not None:
                display_image_batch(images=processed_img_array[1:], titles=titles[1:])

        with st.expander("📊 Show Results Table"):
            st.subheader("Cells Touching (%d)" % result_dict["cells_touching"])
            st.dataframe(features_df.reset_index(drop=True), hide_index=True)
        
# --- Tab 2: Batch Processing ---
with tab2:
    st.subheader("Batch Processing (Multiple Images)")
    uploaded_files = st.file_uploader("Upload multiple image files from a folder", type=allowed_extensions, accept_multiple_files=True, key="multi")

    if uploaded_files and st.button("Process Uploaded Batch", key="process_batch"):
        with st.spinner("Processing batch..."):
            zip_path, batch_stats_df = process_uploaded_files(
                uploaded_files,
                model_type=model_type,
                min_area=min_area,
                numbered=numbered
            )

        st.session_state.zip_path = zip_path
        st.session_state.batch_stats_df = batch_stats_df

    if st.session_state.zip_path:
        st.success("Batch processing complete.")
        st.info(f"📦 ZIP file includes {len(uploaded_files) * 2 + 1} files:\n"
                "• Individual feature CSV files for each image\n"
                "• Combined visualizations for each image\n"
                "• Final summary statistics CSV")
        with open(st.session_state.zip_path, "rb") as f:
            st.download_button("📦 Download Results (ZIP)", f, file_name="results.zip", mime="application/zip")

        with st.expander("📊 Show Summary Table"):
            st.dataframe(st.session_state.batch_stats_df)
    else:
        st.warning("No images were successfully processed.")
