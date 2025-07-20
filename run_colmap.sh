#!/bin/bash
#
# A robust script to run the full COLMAP pipeline, from sparse to dense reconstruction.
# It cleans up old data, runs each step sequentially, and checks for success.
#
# Usage:
# 1. Make the script executable: chmod +x run_colmap.sh
# 2. Run it from the project root: ./run_colmap.sh

set -e # Exit immediately if a command exits with a non-zero status.

PROJECT_PATH=$(pwd)
DATA_PATH="$PROJECT_PATH/data"
IMAGE_PATH="$DATA_PATH/frames"

echo "--- Starting COLMAP Pipeline ---"

# --- 1. Clean up previous runs ---
echo "--> Cleaning up old data..."
rm -rf "$DATA_PATH/sparse" "$DATA_PATH/dense" "$DATA_PATH/database.db"
mkdir -p "$DATA_PATH/sparse" "$DATA_PATH/dense"

# --- 2. Feature Extraction ---
echo "--> Extracting features..."
colmap feature_extractor \
    --database_path "$DATA_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --ImageReader.camera_model OPENCV \
    --ImageReader.single_camera 1

# --- 3. Feature Matching ---
echo "--> Matching features..."
colmap exhaustive_matcher \
    --database_path "$DATA_PATH/database.db"

# --- 4. Sparse Reconstruction (Mapping) ---
echo "--> Performing sparse reconstruction (mapping)..."
colmap mapper \
    --database_path "$DATA_PATH/database.db" \
    --image_path "$IMAGE_PATH" \
    --output_path "$DATA_PATH/sparse"

echo "!!! Sparse reconstruction finished. Checking for success... !!!"

# --- 5. Check for Success and Proceed to Dense Reconstruction ---
if [ ! -d "$DATA_PATH/sparse/0" ]; then
    echo "--------------------------------------------------------------------"
    echo "ERROR: Sparse reconstruction FAILED. The 'data/sparse/0' directory was not created."
    echo "This is almost always due to poor video quality (blurry, fast motion, textureless surfaces)."
    echo "Please use a high-quality video (like the sample provided) and try again."
    echo "--------------------------------------------------------------------"
    exit 1
fi

echo "--> Sparse reconstruction successful! Proceeding to dense reconstruction."

# --- 6. Dense Reconstruction ---
colmap image_undistorter --image_path "$IMAGE_PATH" --input_path "$DATA_PATH/sparse/0" --output_path "$DATA_PATH/dense" --output_type COLMAP
colmap patch_match_stereo --workspace_path "$DATA_PATH/dense" --workspace_format COLMAP --PatchMatchStereo.geom_consistency true
colmap stereo_fusion --workspace_path "$DATA_PATH/dense" --workspace_format COLMAP --input_type geometric --output_path "$DATA_PATH/dense/fused.ply"

echo "--- COLMAP Pipeline Finished Successfully! ---"
echo "Dense point cloud is ready at: $DATA_PATH/dense/fused.ply"

