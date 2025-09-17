import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import math
import pandas as pd
import tempfile
import os
import zipfile
import shutil
from core import process_image_from_stream
from utils import sort_images_by_group_and_column, sort_images_incucyte, extract_incucyte_info, combine_image_statistics

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
        from core import process_image_from_path
        result = process_image_from_path(image_path, model_type, min_area, numbered)
    
    if result is None:
        return None, None, None, None
    
    # Extract results from shared processing
    segmentation = result['segmentation']
    features_df = result['features_df']
    image_stats = result['image_stats']
    visualizations = result['visualizations']
    
    # Prepare visualization arrays for UI
    vis_arrays = [
        # image, 
        visualizations['final_filtered_vis'], 
        # visualizations['area_filtered_vis'], 
        visualizations['segmentation_vis']
    ]
    titles = [
        # "Original Image",
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

# --- Utility: process batch of uploaded files ---
def process_uploaded_files(uploaded_files, model_type="vit_b_lm", min_area=200, numbered=False):
    with tempfile.TemporaryDirectory() as tmp_input_dir:
        output_dir = os.path.join(tmp_input_dir, "results")
        os.makedirs(output_dir, exist_ok=True)

        input_paths = []
        for file in uploaded_files:
            save_path = os.path.join(tmp_input_dir, file.name)
            with open(save_path, "wb") as f:
                f.write(file.read())
            input_paths.append(save_path)

        all_image_stats = []
        total = len(input_paths)

        # Streamlit progress bar and status
        progress_bar = st.progress(0)
        status_text = st.empty()

        incucyte_group = sort_images_incucyte(images=map(lambda x: x.name, uploaded_files))
        print(incucyte_group)
        for idx, image_path in enumerate(input_paths):
            status_text.text(f"Processing {os.path.basename(image_path)} ({idx + 1}/{total})")

            visArr, image_stats, titles, features, segmentation, incucyte_info = process_image_for_ui(
                image_path=image_path,
                output_dir=output_dir,
                model_type=model_type,
                min_area=min_area,
                numbered=numbered
            )
            if image_stats:
                incucyte_group[incucyte_info["key"]]["results"].append(tuple([incucyte_info["position"], image_stats, features, segmentation]))
                all_image_stats.append(image_stats)
                
                # Save individual image results
                filename = os.path.splitext(os.path.basename(image_path))[0]
                
                # Save features CSV
                features_csv_path = os.path.join(output_dir, f"{filename}_features.csv")
                features.to_csv(features_csv_path, index=False)
                
                # Save visualizations
                if visArr and len(visArr) >= 4:
                    # Create a combined visualization for each image
                    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
                    
                    # Original image
                    axes[0].imshow(visArr[0])
                    axes[0].set_title(titles[0])
                    axes[0].axis("off")
                    
                    # Final filtered
                    axes[1].imshow(visArr[1])
                    axes[1].set_title(titles[1])
                    axes[1].axis("off")
                    
                    # Area filtered
                    # All cells
                    axes[2].imshow(visArr[2])
                    axes[2].set_title(titles[2])
                    axes[2].axis("off")
                    
                    # All cells
                    # axes[3].imshow(visArr[3])
                    # axes[3].set_title(titles[3])
                    # axes[3].axis("off")
                    
                    # Save combined visualization
                    vis_output_path = os.path.join(output_dir, f"{filename}_visualization.png")
                    plt.tight_layout()
                    plt.savefig(vis_output_path, dpi=150, bbox_inches='tight')
                    plt.close()

            progress_bar.progress((idx + 1) / total)

        status_text.text("Processing complete.")
        progress_bar.empty()

        if all_image_stats:
            final_combined_stats = combine_image_statistics(all_image_stats)
            stats_df = pd.DataFrame(final_combined_stats)
            final_csv_path = os.path.join(output_dir, "FINAL_STATS.csv")
            stats_df.to_csv(final_csv_path, index=False)

            zip_path = os.path.join(tmp_input_dir, "results.zip")
            file_count = 0
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, output_dir)
                        zipf.write(file_path, arcname)
                        file_count += 1

            final_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip")
            shutil.copy(zip_path, final_zip.name)

            return final_zip.name, stats_df

        return None, None

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

        st.subheader("Comparison")
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded_image_preview, caption="Uploaded Image", use_column_width=True)
        with col2:
            st.subheader("Processed Image")
            display_image_batch(images=processed_img_array, titles=titles)

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

        if zip_path:
            st.success("Batch processing complete.")
            st.info(f"📦 ZIP file includes {len(uploaded_files) * 2 + 1} files:\n"
                   "• Individual feature CSV files for each image\n"
                   "• Combined visualizations for each image\n"
                   "• Final summary statistics CSV")
            with open(zip_path, "rb") as f:
                st.download_button("📦 Download Results (ZIP)", f, file_name="results.zip", mime="application/zip")

            with st.expander("📊 Show Summary Table"):
                st.dataframe(batch_stats_df)
        else:
            st.warning("No images were successfully processed.")
