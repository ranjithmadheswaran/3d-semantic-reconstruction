# 3D Scene Reconstruction and Semantic Labeling

This project creates a 3D semantic point cloud from a simple smartphone video. It uses a combination of classical Structure from Motion (SfM) with COLMAP and modern deep learning models for semantic segmentation.

## Project Structure

```
├── data/                 # Input data (videos, extracted frames)
│   ├── video.mp4         # Your input video
│   └── frames/           # Extracted frames from the video
├── results/              # Output files (final point cloud)
├── scripts/              # All Python scripts for the pipeline
├── .gitignore
├── README.md
└── requirements.txt
```

## Prerequisites

1.  **Python 3.8+**
2.  **COLMAP:** You must install the COLMAP command-line tool. Follow the official installation guide at [colmap.github.io/install.html](https://colmap.github.io/install.html).
3.  **PyTorch with CUDA (Recommended):** For GPU acceleration on the segmentation step. Visit [pytorch.org](https://pytorch.org/) for installation instructions.

## Setup

1.  **Clone the repository or create the files as provided.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Execution Pipeline

Follow these steps in order.

### Step 0: Add Your Video

Place your video file inside the `data/` directory and name it `video.mp4`.

### Step 1: Extract Video Frames

This script converts your video into a sequence of images.
```bash
python scripts/1_extract_frames.py
```

### Step 2: Run COLMAP for 3D Reconstruction

Run the following commands from your project's root directory. This is the most time-consuming step.

*See the full list of commands in the `scripts/` directory or the original guide.*

### Step 3: Run 2D Semantic Segmentation

This script runs a deep learning model on each frame to identify objects.
```bash
python scripts/2_run_segmentation.py
```

### Step 4: Fuse Semantics into the 3D Point Cloud

This script combines the 3D points from COLMAP and the 2D masks to create the final colored point cloud.
```bash
python scripts/3_fuse_semantics.py
```

### Step 5: Visualize the Result

View your final creation!
```bash
python scripts/4_visualize.py
```