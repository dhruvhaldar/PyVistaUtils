import asyncio
import platform
import os
import vtk
import pyvista as pv
import numpy as np
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_vtm_parallel(file_path):
    """Read a VTM file in parallel and process blocks."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"VTM file not found: {file_path}")
        
        # Read the VTM file
        multiblock = pv.read(file_path)
        num_blocks = multiblock.n_blocks
        
        # Distribute blocks among processes
        blocks_per_process = max(1, num_blocks // size)
        start_block = rank * blocks_per_process
        end_block = min((rank + 1) * blocks_per_process, num_blocks) if rank < size - 1 else num_blocks
        
        # Local results and blocks for this process
        local_results = []
        local_blocks = pv.MultiBlock()
        
        for i in range(start_block, end_block):
            block = multiblock[i]
            if block:
                # Example processing: count points and compute scalar mean
                num_points = block.n_points
                scalar_name = list(block.point_data.keys())[0] if block.point_data else None
                mean_scalar = np.mean(block.point_data[scalar_name]) if scalar_name else 0.0
                local_results.append((i, num_points, mean_scalar))
                local_blocks.append(block)
        
        # Gather results from all processes
        all_results = comm.gather(local_results, root=0)
        
        # Process 0 collects and prints results
        if rank == 0:
            print("Results from all processes:")
            for proc_id, results in enumerate(all_results):
                for block_id, num_points, mean_scalar in results:
                    print(f"Process {proc_id}, Block {block_id}: {num_points} points, Mean Scalar: {mean_scalar:.2f}")
        
        return local_blocks, all_results
    
    except Exception as e:
        if rank == 0:
            print(f"Error reading VTM file: {str(e)}")
        return None, None

def compress_vtm(local_blocks, all_results, output_file):
    """Compress the MultiBlock dataset and save as a VTM file with ZLib level 9."""
    try:
        # Write compressed VTU files for local blocks
        local_vtu_files = []
        start_block = sum(len(results) for results in all_results[:rank]) if all_results else 0
        for i, block in enumerate(local_blocks):
            block_id = start_block + i
            vtu_file = f"{output_file}_block_{block_id}_rank_{rank}.vtu"
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(vtu_file)
            writer.SetInputData(block)
            writer.SetDataModeToBinary()
            writer.SetCompressorTypeToZLib()
            writer.SetCompressionLevel(9)
            writer.Write()
            local_vtu_files.append((block_id, vtu_file))
        
        # Gather VTU file information
        all_vtu_files = comm.gather(local_vtu_files, root=0)
        
        # Rank 0 writes the VTM file
        if rank == 0:
            output_multiblock = pv.MultiBlock()
            
            # Add blocks from all processes
            for proc_id, vtu_files in enumerate(all_vtu_files):
                for block_id, _ in vtu_files:
                    # Get the block from the process that owns it
                    if proc_id == 0:
                        block = local_blocks[block_id - (sum(len(results) for results in all_results[:proc_id]) if proc_id > 0 else 0)]
                    else:
                        proc_blocks = comm.recv(source=proc_id, tag=proc_id)
                        block = proc_blocks[block_id - sum(len(results) for results in all_results[:proc_id])]
                    output_multiblock.append(block)
                    output_multiblock.set_block_name(len(output_multiblock) - 1, f"Block_{block_id}")
            
            # Write compressed VTM file
            output_multiblock.save(output_file, binary=True)
            print(f"Compressed VTM file written to: {output_file}")
        
        else:
            # Send local blocks to rank 0 if any
            if local_vtu_files:
                comm.send(local_blocks, dest=0, tag=rank)
    
    except Exception as e:
        if rank == 0:
            print(f"Error compressing VTM file: {str(e)}")

def create_sample_vtm():
    """Create a sample VTM dataset for demonstration."""
    multiblock = pv.MultiBlock()
    for i in range(4):  # Create 4 sample blocks
        # Create points
        points = np.array([[0, 0, i], [1, 0, i], [0, 1, i], [0, 0, i + 1]], dtype=np.float32)
        
        # Create a single tetrahedral cell in VTK legacy format
        cells = np.array([4, 0, 1, 2, 3], dtype=np.int32)  # [n_points, id1, id2, id3, id4]
        
        # Create vtkUnstructuredGrid
        grid = pv.UnstructuredGrid(cells, [vtk.VTK_TETRA], points)
        
        # Add scalar data
        grid.point_data[f"Scalar_{i}"] = np.array([i * 1.0, i * 1.1, i * 1.2, i * 1.3], dtype=np.float32)
        
        # Verify grid integrity
        if grid.n_cells == 0 or grid.n_points != 4:
            if rank == 0:
                print(f"Error: Invalid grid in block {i}, n_cells={grid.n_cells}, n_points={grid.n_points}")
            return None
        
        multiblock.append(grid)
    
    # Write sample VTM file
    file_path = f"sample_{rank}.vtm" if size > 1 else "sample.vtm"
    try:
        multiblock.save(file_path, binary=True)
        if rank == 0:
            print(f"Created sample VTM file: {file_path}")
    except Exception as e:
        if rank == 0:
            print(f"Error creating sample VTM file: {str(e)}")
        return None
    return file_path

async def main():
    """Main async loop for Pyodide compatibility."""
    # Create a sample VTM file for demonstration (only on rank 0)
    input_file = create_sample_vtm() if rank == 0 else None
    
    # Broadcast input file path to all processes
    input_file = comm.bcast(input_file, root=0)
    
    # Exit if file creation failed
    if input_file is None:
        if rank == 0:
            print("Failed to create sample VTM file. Exiting.")
        return
    
    # Ensure file is written before reading
    comm.Barrier()
    
    # Read and process VTM file in parallel
    local_blocks, all_results = read_vtm_parallel(input_file)
    
    # Compress the VTM file for ParaView
    if local_blocks is not None and all_results is not None:
        output_file = "compressed_output.vtm"
        compress_vtm(local_blocks, all_results, output_file)
    
    # Control frame rate
    await asyncio.sleep(1.0 / 60)  # 60 FPS

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
        