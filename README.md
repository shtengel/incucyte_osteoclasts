# Incucyte App - Modular Cell Segmentation Pipeline

This application provides a modular pipeline for cell segmentation and analysis using MicroSAM (Segment Anything Model for microscopy).

## Project Structure

The code has been organized into logical sub-folders:

```
incucyte_app/
├── core/                    # Core processing modules
│   ├── __init__.py
│   ├── image_processing.py  # Image I/O operations and basic processing
│   ├── segmentation.py      # MicroSAM-based cell segmentation
│   └── processor.py         # ⭐ Shared processing logic for CLI and UI
├── analysis/                # Analysis modules
│   ├── __init__.py
│   └── feature_extraction.py # Shape and intensity feature extraction
├── visualization/           # Visualization modules
│   ├── __init__.py
│   └── visualization.py     # Plotting and image visualization
├── utils/                   # Utility modules
│   ├── __init__.py
│   └── utils.py            # Utility functions and data processing
├── main.py                  # Main CLI interface and orchestration
├── ui.py                    # Streamlit web interface
├── app.py                   # Legacy compatibility layer
├── __init__.py              # Package initialization and exports
└── README.md                # This file
```

## Usage

### Web Interface (Recommended)

Launch the Streamlit web interface:

```bash
streamlit run ui.py
```

This provides an interactive web interface with:
- Single image processing with real-time visualization
- Batch processing with progress tracking
- Interactive parameter adjustment
- Download results as ZIP files containing:
  - Individual feature CSV files for each image
  - Combined visualizations for each image
  - Final summary statistics CSV

### Command Line Interface

```bash
python main.py --input /path/to/images --output /path/to/results [options]
```

### Available Options

- `--input`: Input directory containing images (required)
- `--output`: Output directory for results (required)
- `--model`: MicroSAM model type (default: vit_b_lm)
- `--min-area`: Minimum cell area in pixels² (default: 500)
- `--numbered`: Show cells with numbers on visualization

### Programmatic Usage

```python
from incucyte_app import process_directory, process_image

# Process a single image
stats = process_image("path/to/image.png", "output/dir")

# Process a directory of images
results_df = process_directory("input/dir", "output/dir", min_area=200)
```

## Module Details

### core/
- **`image_processing.py`**
  - `read_image()` - Read images with error handling
  - `get_image_files()` - Get all image files from directory
- **`segmentation.py`**
  - `run_automatic_instance_segmentation()` - Run MicroSAM segmentation
- **`processor.py`** ⭐ **NEW**
  - `process_image_core()` - Core shared processing logic
  - `process_image_from_path()` - Process image from file path
  - `process_image_from_stream()` - Process image from stream (for UI)

### analysis/
- **`feature_extraction.py`**
  - `extract_shape_features()` - Extract cell shape features
  - `calculate_plate_coverage()` - Calculate plate coverage percentage

### visualization/
- **`visualization.py`**
  - `visualize_segmentation()` - Create segmentation visualizations
  - `add_numbers_to_image()` - Add cell numbers to images
  - `create_combined_visualization()` - Create multi-panel visualizations

### utils/
- **`utils.py`**
  - `sort_images_by_group_and_column()` - Sort images by experimental groups
  - `calculate_image_statistics()` - Calculate summary statistics

## Dependencies

### Core Dependencies
- micro-sam
- opencv-python
- scikit-image
- matplotlib
- pandas
- numpy
- imageio
- tqdm

### Web Interface Dependencies
- streamlit
- Pillow (PIL)

## Installation

```bash
# Install core dependencies
pip install micro-sam opencv-python scikit-image matplotlib pandas numpy imageio tqdm

# Install web interface dependencies
pip install streamlit Pillow

# Or use conda
conda activate micro-sam
conda install streamlit pillow
```

## Migration from Legacy Code

The original `app.py` file has been refactored into modules while maintaining backward compatibility. All original functions are still available through imports:

```python
# Old way (still works)
from app import process_directory

# New way (recommended)
from main import process_directory
# or
from incucyte_app import process_directory
```

## Shared Processing Logic

The core processing logic is now centralized in `core/processor.py` to ensure consistency between CLI and UI:

```python
# Direct access to shared processing functions
from incucyte_app import process_image_core, process_image_from_path, process_image_from_stream

# Process from file path (CLI usage)
result = process_image_from_path("image.png", model_type="vit_b_lm", min_area=200)

# Process from stream (UI usage)  
result = process_image_from_stream(image_stream, "image.png", model_type="vit_b_lm", min_area=200)

# Access all processing results
image = result['image']
segmentation = result['segmentation'] 
features_df = result['features_df']
visualizations = result['visualizations']
```

This ensures that both the CLI (`main.py`) and web interface (`ui.py`) use exactly the same processing logic, eliminating inconsistencies and making the codebase more maintainable.
