import open3d as o3d
import numpy as np

def create_mesh_from_point_cloud(point_cloud_path, output_mesh_path, depth=8):
    """
    Loads a point cloud, computes normals, and generates a mesh using
    Poisson surface reconstruction. It preserves the vertex colors.

    Args:
        point_cloud_path (str): Path to the input .ply point cloud file.
        output_mesh_path (str): Path to save the output .ply mesh file.
        depth (int): The depth of the octree used for reconstruction. Higher
                     values mean more detail but more memory usage.
    """
    print(f"Loading point cloud from {point_cloud_path}...")
    pcd = o3d.io.read_point_cloud(point_cloud_path)

    if not pcd.has_points():
        print("Error: The point cloud is empty. Cannot create a mesh.")
        return

    print("Estimating normals...")
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.orient_normals_consistent_tangent_plane(100)

    print(f"Creating mesh with Poisson reconstruction (depth={depth})...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth
    )

    print(f"Saving mesh to {output_mesh_path}...")
    o3d.io.write_triangle_mesh(output_mesh_path, mesh)
    print("Mesh created successfully.")

if __name__ == '__main__':
    # --- Configuration ---
    SEMANTIC_PLY_INPUT = 'results/semantic_scene.ply'
    SEMANTIC_MESH_OUTPUT = 'results/semantic_mesh.ply'
    # ---------------------

    create_mesh_from_point_cloud(SEMANTIC_PLY_INPUT, SEMANTIC_MESH_OUTPUT)

