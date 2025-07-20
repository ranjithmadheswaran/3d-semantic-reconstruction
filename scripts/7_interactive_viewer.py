import open3d as o3d
import numpy as np

def main():
    """
    An interactive viewer application built directly in Python using Open3D.
    This avoids the need for a full game engine like Unity or Godot.
    """
    # --- 1. Load the Semantic Mesh ---
    asset_path = 'results/semantic_asset.glb'
    try:
        mesh = o3d.io.read_triangle_mesh(asset_path)
        print(f"Successfully loaded semantic asset from '{asset_path}'")
    except Exception as e:
        print(f"Error: Could not load asset from '{asset_path}'. {e}")
        print("Please ensure you have run the full pipeline, including '6_convert_to_gltf.py'.")
        return

    # --- 2. Set up the Scene for Raycasting ---
    # The RaycastingScene allows us to efficiently check for collisions.
    scene = o3d.t.geometry.RaycastingScene()
    # Convert mesh to a Tensor-based geometry for the scene
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene.add_triangles(t_mesh)

    # --- 3. Create the "Character" ---
    # We'll use a small red sphere to represent our character.
    character = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
    character.paint_uniform_color([1.0, 0.1, 0.1]) # Paint it red
    # Initial position and rotation (facing forward along the Z-axis)
    character_transform = np.eye(4)
    character_transform[2, 3] = -1.0 # Start at z = -1

    # --- 4. Set up the Visualizer and State ---
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Interactive Semantic Viewer", width=1280, height=720)
    vis.add_geometry(mesh)
    vis.add_geometry(character)

    # --- Enhance the "Light View" ---
    render_option = vis.get_render_option()
    render_option.mesh_show_wireframe = True
    render_option.mesh_show_back_face = True
    render_option.light_on = True
    render_option.background_color = np.asarray([0.1, 0.1, 0.1]) # Dark gray background


    # --- 5. Define the Interaction Logic (The "Brain") ---
    def move_forward(vis):
        nonlocal character_transform
        
        # Define movement properties
        step_size = 0.05
        # Get the current forward direction from the character's transform matrix
        forward_vector = character_transform[:3, 2]
        # Get the current position
        current_pos = character_transform[:3, 3]
        # Calculate the potential next position
        next_pos = current_pos + forward_vector * step_size

        # --- Raycast to check for collisions ---
        # Create a ray from slightly above the current position to the next position
        ray_origin = current_pos + np.array([0, 0.1, 0]) # Start ray slightly above ground
        ray = o3d.core.Tensor([ray_origin], dtype=o3d.core.Dtype.Float32)
        
        # Cast the ray and get the result
        ans = scene.cast_rays(ray, direction=o3d.core.Tensor([forward_vector], dtype=o3d.core.Dtype.Float32))
        
        # Get the distance to the first hit
        hit_distance = ans['t_hit'].numpy()[0]

        # --- Make a decision based on the collision ---
        # If the wall is too close (less than our step size), don't move.
        if hit_distance < step_size:
            print("Collision detected! Cannot move forward.")
            return False # Indicate that no update is needed

        # If the path is clear, update the character's position
        character_transform[:3, 3] = next_pos
        character.transform(character_transform) # Apply the new transform
        return True # Indicate that the visualizer should update

    def rotate_left(vis):
        # This function would handle rotation (e.g., for 'A' key)
        print("Rotate Left (Not implemented)")
        return False

    def rotate_right(vis):
        # This function would handle rotation (e.g., for 'D' key)
        print("Rotate Right (Not implemented)")
        return False

    # --- 6. Register Callbacks and Run ---
    # Map keys to functions. ASCII codes: W=87, A=65, D=68
    vis.register_key_callback(87, move_forward)
    vis.register_key_callback(65, rotate_left)
    vis.register_key_callback(68, rotate_right)

    print("\n--- Controls ---")
    print("Press 'W' to move the character forward.")
    print("Close the window to exit.")
    
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()