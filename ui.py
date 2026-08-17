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
tab1, tab2 = st.tabs(["🖼 Single Image", "📂 Batch Processing"])

if "zip_path" not in st.session_state:
    st.session_state.zip_path = None
if "batch_stats_df" not in st.session_state:
    st.session_state.batch_stats_df = None

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
            st.image(uploaded_image_preview, caption="Uploaded Image", use_column_width=True)
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
