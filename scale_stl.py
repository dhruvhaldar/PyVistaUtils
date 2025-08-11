import pyvista as pv

def scale_stl_pyvista(input_file, output_file, scale_factor):
    """
    Scales an STL file using PyVista and saves the result to a new file.

    Args:
        input_file (str): The path to the input STL file.
        output_file (str): The path to the output STL file.
        scale_factor (float or list): The factor by which to scale the model.
                                      Can be a single float (e.g., 2.0 for uniform scaling)
                                      or a list/tuple of floats for non-uniform scaling (e.g., [1.0, 2.0, 1.0]).
    """
    try:
        # Load the STL file as a PyVista PolyData object
        your_mesh = pv.read(input_file)
        
        print(f"Successfully loaded {input_file}.")
        
        # Scale the mesh using the scale() method
        scaled_mesh = your_mesh.scale(scale_factor, inplace=False)
        
        print(f"Successfully scaled mesh by a factor of {scale_factor}.")
        
        # Save the scaled mesh to a new STL file
        scaled_mesh.save(output_file, binary=False) # 'binary=False' saves as an ASCII STL
        
        print(f"Successfully saved scaled mesh to {output_file}.")

    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # --- Configuration ---
    input_stl = 'input.stl'         # The original STL file
    output_stl = 'output_scaled_pyvista.stl' # The name for the new scaled STL file
    scale_by = 2.0                    # The scaling factor (e.g., 2.0 doubles the size)
    # For non-uniform scaling, you could use a list: scale_by = [2.0, 1.5, 1.0]

    # --- Run the function ---
    scale_stl_pyvista(input_stl, output_stl, scale_by)
