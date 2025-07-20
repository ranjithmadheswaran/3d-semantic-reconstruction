import numpy as np
import open3d as o3d
import pycolmap
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt

def fuse_semantics(colmap_path, masks_path, dense_ply_path, output_ply_path):
    """
    Fuses 2D semantic masks onto a 3D point cloud from COLMAP.
    This version is modified to work with the SPARSE point cloud from COLMAP,
    making it compatible with systems that do not have a CUDA-enabled GPU.
    """
    colmap_path = Path(colmap_path)
    masks_path = Path(masks_path)
    
    # --- 1. Validate Inputs and Load COLMAP Reconstruction ---
    print("Loading COLMAP model...")
    if not colmap_path.exists() or not any(colmap_path.glob('*')):
        print(f"Error: The COLMAP model directory is empty or does not exist: '{colmap_path}'")
        print("This means the 'colmap mapper' command likely failed in the previous step.")
        print("Please re-run the COLMAP pipeline and check its output for errors.")
        return

    try:
        reconstruction = pycolmap.Reconstruction(colmap_path)
    except Exception as e:
        print(f"Error loading COLMAP reconstruction: {e}")
        print("The model files might be corrupt or incomplete. Please re-run the COLMAP pipeline.")
        return
        
    cameras = reconstruction.cameras
    images = reconstruction.images
    
    # --- 2. Get 3D Points from the SPARSE Reconstruction ---
    print("Loading sparse 3D points from the reconstruction...")
    sparse_points = reconstruction.points3D
    if not sparse_points:
        print("Error: The COLMAP reconstruction contains no 3D points.")
        print("This indicates a failure in the 'colmap mapper' step.")
        return

    # Convert sparse points to a NumPy array of XYZ coordinates
    points_3d = np.array([p.xyz for p in sparse_points.values()])
    num_points = len(points_3d)
    
    # --- 3. Initialize Data Structures ---
    # We store a list of label "votes" for each 3D point
    point_labels = [[] for _ in range(num_points)]

    # --- 4. Project Points and Gather Labels from Each Image ---
    print("Projecting points and gathering labels...")
    for image_id, image in tqdm(images.items(), desc="Processing images"):
        cam = cameras[image.camera_id]
        
        # Load the corresponding semantic mask
        mask_file = masks_path / f"{Path(image.name).stem}_mask.png"
        if not mask_file.exists():
            continue
        mask = np.array(Image.open(mask_file))
        
        # Manually construct the projection matrix P = K @ [R|t]
        # This is the most robust method across different pycolmap versions.
        # 1. Get K, the camera intrinsics matrix
        K = cam.calibration_matrix()
        # 2. Get [R|t], the camera extrinsics matrix (world-to-camera)
        # Note the parentheses after cam_from_world() to call the method.
        Rt = image.cam_from_world().matrix()[:3, :]
        proj_matrix = K @ Rt

        # Project all 3D points into the current camera view
        points_3d_homo = np.hstack((points_3d, np.ones((num_points, 1))))
        points_2d_homo = (proj_matrix @ points_3d_homo.T).T
        
        # De-homogenize and filter out points that are behind the camera (z <= 0)
        visible_mask = points_2d_homo[:, 2] > 0
        points_2d = points_2d_homo[visible_mask, :2] / points_2d_homo[visible_mask, 2, np.newaxis]
        
        # Check which points are within the image boundaries
        h, w = cam.height, cam.width
        in_bounds_mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] < w) & \
                         (points_2d[:, 1] >= 0) & (points_2d[:, 1] < h)
        
        # Get the 3D indices of points that are visible and in bounds
        visible_3d_indices = np.where(visible_mask)[0][in_bounds_mask]
        
        # Get the 2D coordinates for these valid points
        valid_2d_coords = points_2d[in_bounds_mask].astype(int)
        
        # Look up the semantic label for each valid point from the mask
        for i, (x, y) in enumerate(valid_2d_coords):
            point_3d_idx = visible_3d_indices[i]
            label = mask[y, x]
            if label > 0: # Ignore background label (usually 0)
                point_labels[point_3d_idx].append(label)

    # --- 5. Aggregate Labels for Each Point (Majority Vote) ---
    print("Aggregating labels...")
    final_labels = np.zeros(num_points, dtype=int)
    for i in range(num_points):
        if point_labels[i]:
            most_common = Counter(point_labels[i]).most_common(1)[0]
            final_labels[i] = most_common[0]

    # --- 6. Create a Color Map for Visualization ---
    max_label = final_labels.max()
    if max_label == 0:
        print("Warning: No semantic labels were successfully projected onto the point cloud.")
        print("The output model will have a single color. Check mask quality and camera poses.")
        point_colors = np.full_like(points_3d, 0.5)
    else:
        colormap = plt.get_cmap("viridis", max_label + 1)
        point_colors = colormap(final_labels / max_label)[:, :3] # Get RGB

    # --- 7. Create Final Point Cloud and Save ---
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    o3d.io.write_point_cloud(output_ply_path, pcd)
    print(f"Successfully saved semantic point cloud to {output_ply_path}")


if __name__ == '__main__':
    # --- Configuration ---
    COLMAP_MODEL_PATH = 'data/sparse/0'
    MASKS_INPUT_DIR = 'data/masks'
    # DENSE_PLY_INPUT is no longer used in this version of the script.
    DENSE_PLY_INPUT = 'data/dense/fused.ply' # This path is now ignored.
    SEMANTIC_PLY_OUTPUT = 'results/semantic_scene.ply'
    # ---------------------

    fuse_semantics(COLMAP_MODEL_PATH, MASKS_INPUT_DIR, DENSE_PLY_INPUT, SEMANTIC_PLY_OUTPUT)