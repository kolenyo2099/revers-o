# Revers-o: GroundedSAM + Perception Encoder for Image Similarity Search

A powerful application that uses Grounded SAM for object detection and Perception Encoder for semantic understanding to build a searchable database of image regions.

## Quick Start - Just Two Commands!

```bash
# First time setup - Run this once
./easy_setup.sh

# Start the application - Run this each time
./run.sh
```

That's it! The setup script automatically detects your hardware and configures everything for you.

### Having problems?

Run the troubleshooting script for help:
```bash
./troubleshoot.sh
```

---

## Features

- **Automated Object Detection**: Uses Grounded SAM to detect objects based on text prompts
- **Semantic Understanding**: Leverages Perception Encoder for deep semantic region embeddings
- **Searchable Database**: Builds a Qdrant vector database for fast similarity search
- **Interactive Interface**: Gradio-based web interface for easy interaction
- **Batch Processing**: Process entire folders of images automatically
- **Crash Recovery**: Resume processing from checkpoints if interrupted

## How to Use

1. **Run the setup script**:
   ```bash
   ./easy_setup.sh
   ```
   This will:
   - Detect your hardware (Apple Silicon, NVIDIA GPU, or CPU)
   - Create a Python virtual environment
   - Install the correct dependencies for your system
   - Set up directories and clone required repositories

2. **Add images**:
   - Place your images in the `my_images` folder

3. **Start the application**:
   ```bash
   ./run.sh
   ```

4. **Using the interface**:
   - **Main Menu**: Choose between "Create New Database" or "Search Existing Database"
   - **Create Database**: Select folder, enter prompts, and start processing
   - **Search Database**: Upload an image, detect regions, and search for similar regions

## Supported Hardware

The application automatically configures itself for:
- **Apple Silicon** (M1/M2 Macs): Uses Metal Performance Shaders (MPS) for acceleration
- **NVIDIA GPUs**: Uses CUDA for acceleration
- **CPU-only systems**: Optimized for CPU processing

## Troubleshooting

If you encounter issues:

1. Run the troubleshooting script:
   ```bash
   ./troubleshoot.sh
   ```

2. Common solutions:
   - Reinstall by running `./easy_setup.sh` again
   - Check that you have enough disk space and memory
   - Ensure your Python version is 3.8 or newer

## System Requirements

- **Python 3.8+**
- **Memory**: At least 8GB RAM recommended for processing
- **Storage**: Depends on your image collection size
- **GPU/MPS**: Optional but recommended for faster processing

## Installation

1. **Clone the repository and install basic requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the setup script to handle special dependencies:**
   ```bash
   python setup.py
   ```
   
   This will:
   - Clone the Facebook Perception Models repository
   - Set up Apple Silicon compatibility (if needed)
   - Create necessary project directories
   - Clean up any existing lock files

## Quick Start

### Option 1: Complete Pipeline (Recommended)
```bash
# Build database and launch interface in one command
python main.py --all
```

### Option 2: Step by Step

1. **Add your images to the `./my_images` folder** (created by setup.py)

2. **Build the database:**
   ```bash
   python main.py --build-database
   ```

3. **Launch the interface:**
   ```bash
   python main.py --interface
   ```

## Usage Options

### Command Line Arguments

- `--build-database`: Process images and build the search database
- `--interface`: Launch the Gradio web interface
- `--all`: Build database and launch interface
- `--folder PATH`: Specify custom images folder (default: `./my_images`)
- `--prompts "text1, text2"`: Custom detection prompts (default: "person, building, car, text, object")
- `--collection NAME`: Custom collection name (default: "grounded_image_regions")

### Example Commands

```bash
# Process custom folder with specific prompts
python main.py --build-database --folder ./photos --prompts "person, dog, car, house"

# Launch interface only (assumes database already exists)
python main.py --interface

# Build database with custom collection name
python main.py --build-database --collection my_custom_regions
```

## Web Interface

Once launched, the Gradio interface will be available at `http://127.0.0.1:7860`

### How to Use:

1. **Upload an image** using the file uploader
2. **Enter text prompts** describing objects you want to detect (e.g., "person, car, building")
3. **Click "Detect Regions"** to process the image
4. **Select a region** from the dropdown to preview it
5. **Adjust similarity threshold** to filter search results
6. **Click "Search Similar Regions"** to find matching regions in your database
7. **Select results** to view enlarged versions

## Project Structure

```
grounded-sam-region-search/
├── requirements.txt          # Python dependencies
├── setup.py                 # Setup script for special dependencies
├── main.py                  # Main application file
├── README.md               # This file
├── my_images/              # Default folder for your images (add your images here)
├── perception_models/      # Cloned Facebook repository (created by setup.py)
└── image_retrieval_project/
    ├── qdrant_data/       # Vector database storage
    ├── checkpoints/       # Processing checkpoints
    └── images/           # Processed images cache
```

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- BMP (.bmp)
- TIFF (.tiff)

## Configuration

The application uses several configurable parameters:

- **Optimal Layer**: Layer 40 (semantic understanding layer)
- **Minimum Area Ratio**: 0.005 (minimum region size)
- **Max Regions per Image**: 5
- **Similarity Threshold**: 0.5 (adjustable in interface)
- **Checkpoint Interval**: Every 3 images

## Advanced Usage

### Custom Processing Parameters:

You can modify processing parameters by editing the main.py file:

```python
# In process_folder_for_region_database function:
client = process_folder_for_region_database(
    folder_path=args.folder,
    collection_name=args.collection,
    text_prompts=args.prompts,
    pe_model=pe_model,
    pe_vit_model=pe_vit_model,
    preprocess=preprocess,
    device=device,
    optimal_layer=40,         # Adjust this for different semantic levels
    min_area_ratio=0.005,     # Increase to detect only larger regions
    max_regions=5,            # Increase to detect more regions per image
    visualize_samples=True,   # Set to False for headless operation
    sample_count=2,           # Number of sample visualizations to generate
    checkpoint_interval=3,    # How often to save progress
    resume_from_checkpoint=True  # Set to False to start fresh
)
```

### Environment Variables

You can set these environment variables before running:

```bash
# Enable more verbose logging
export VERBOSE_LOGGING=1

# Force CPU usage even when GPU is available
export FORCE_CPU=1

# Set custom port for Gradio interface
export GRADIO_PORT=8080
```

### Using as a Library

You can import functions from main.py to use in your own applications:

```python
from main import extract_region_embeddings_autodistill, load_pe_model, setup_device

# Setup
device = setup_device()
pe_model, pe_vit_model, preprocess = load_pe_model(device)

# Process a single image
image, masks, embeddings, metadata, labels = extract_region_embeddings_autodistill(
    "path/to/image.jpg",
    "cat, dog, person",
    pe_model_param=pe_model,
    pe_vit_model_param=pe_vit_model,
    preprocess_param=preprocess,
    device_param=device
)

# Use the embeddings and metadata as needed
```

## License and Credits

This application uses:
- [GroundingDINO and SAM](https://github.com/IDEA-Research/GroundingDINO) for object detection
- [Facebook Perception Models](https://github.com/facebookresearch/perception_models) for embeddings
- [Qdrant](https://github.com/qdrant/qdrant) for vector search
- [Gradio](https://github.com/gradio-app/gradio) for the web interface

Please refer to each project for their respective licenses.