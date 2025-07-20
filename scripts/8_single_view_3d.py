import torch
import open3d as o3d
import numpy as np
from PIL import Image
from transformers import DPTForDepthEstimation, DPTImageProcessor

def create_3d_from_single_image(image_path, output_ply_path):
    """
    Creates a 3D point cloud from a single 2D image using an AI depth estimation model.

    Args:
        image_path (str): Path to the input color image.
        output_ply_path (str): Path to save the resulting .ply point cloud.
    """
    # --- 1. Load the AI Model and Processor ---
    print("Loading AI depth estimation model (MiDaS)...")
    # This model is trained to predict depth from a single image.
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"Using device: {device}")

    # --- 2. Load and Process the Image ---
    try:
        color_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Input image not found at '{image_path}'")
        return

    width, height = color_image.size
    
    # Prepare the image for the model
    inputs = processor(images=color_image, return_tensors="pt").to(device)

    # --- 3. Predict the Depth Map ---
    print("Predicting depth from the image...")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate the depth map to the original image size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=color_image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    depth_map = prediction.squeeze().cpu().numpy()
    
    # Invert the depth map so that closer objects have smaller Z values
    depth_map = np.max(depth_map) - depth_map

    # --- 4. Unproject to a 3D Point Cloud ---
    print("Creating 3D point cloud from color and depth images...")
    
    # Convert the color image to a NumPy array for color mapping
    color_data = np.array(color_image)
    
    # Create an Open3D RGBDImage from the color and depth data
    # We need to convert our numpy arrays to Open3D Image types
    o3d_color = o3d.geometry.Image(color_data)
    o3d_depth = o3d.geometry.Image(depth_map)
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_color, o3d_depth, depth_scale=1.0, depth_trunc=1000.0, convert_rgb_to_intensity=False
    )

    # Assume some camera intrinsic parameters. These are reasonable defaults.
    focal_length = 1.2 * max(width, height)
    cx = width / 2
    cy = height / 2
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, focal_length, focal_length, cx, cy)

    # Create the point cloud from the RGBD image and camera intrinsics
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, camera_intrinsics)

    # The default orientation is often sideways, so we rotate it for better viewing
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # --- 5. Save and Visualize ---
    print(f"Saving point cloud to '{output_ply_path}'...")
    o3d.io.write_point_cloud(output_ply_path, pcd)
    
    print("Displaying the 3D point cloud. Close the window to exit.")
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    # --- Configuration ---
    # Use one of the frames you extracted earlier
    INPUT_IMAGE_PATH = 'data/frames/00001.png'
    OUTPUT_PLY_PATH = 'results/single_view_3d.ply'
    # ---------------------
    create_3d_from_single_image(INPUT_IMAGE_PATH, OUTPUT_PLY_PATH)