import open3d as o3d

def visualize_point_cloud(ply_path):
    """
    Loads and displays a .ply file.
    """
    print(f"Loading point cloud from {ply_path}...")
    pcd = o3d.io.read_point_cloud(ply_path)
    if not pcd.has_points():
        print("Error: The point cloud is empty.")
        return
    
    print("Displaying point cloud. Press 'Q' to close the window.")
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    SEMANTIC_PLY_FILE = 'results/semantic_scene.ply'
    visualize_point_cloud(SEMANTIC_PLY_FILE)