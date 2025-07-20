import trimesh
from pathlib import Path

def convert_ply_to_glb(input_ply_path, output_glb_path):
    """
    Loads a .ply mesh file and saves it as a .glb (binary glTF) file.
    This is a standard format for use in game engines and web viewers.
    This version uses the robust 'trimesh' library for conversion.

    Args:
        input_ply_path (str): The path to the input .ply mesh file.
        output_glb_path (str): The path to save the output .glb file.
    """
    input_path = Path(input_ply_path)

    if not input_path.exists():
        print(f"Error: Input file not found at '{input_path}'")
        return

    print(f"Loading mesh from '{input_path}' using trimesh...")
    mesh = trimesh.load(input_path, process=False)

    print(f"Converting and exporting to '{output_glb_path}'...")
    mesh.export(output_glb_path, file_type='glb')
    print("Conversion successful! A valid .glb file has been created.")

if __name__ == '__main__':
    # --- Configuration ---
    SEMANTIC_MESH_INPUT = 'results/semantic_mesh.ply'
    FINAL_ASSET_OUTPUT = 'results/semantic_asset.glb'
    # ---------------------
    convert_ply_to_glb(SEMANTIC_MESH_INPUT, FINAL_ASSET_OUTPUT)