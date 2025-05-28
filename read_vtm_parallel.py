import asyncio
import platform
import os

import vtk
import pyvista as pv
import numpy as np
from mpi4py import MPI
from tqdm import tqdm

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def read_vtm_parallel(file_path):
    """
    Read and analyze a VTM file in parallel.
    Each rank processes its assigned blocks and gathers statistics.
    Rank 0 prints the overall analysis.
    Returns:
        tuple: (local_blocks, all_results) where
               local_blocks is a MultiBlock with blocks assigned to this rank
               all_results contains block statistics gathered from all ranks
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"VTM file not found: {file_path}")
        
        multiblock = pv.read(file_path,progress_bar=True)
        num_blocks = multiblock.n_blocks
        if rank == 0:
            print(f"Reading VTM file with {num_blocks} blocks")
        
        local_blocks = pv.MultiBlock()
        local_results = []
        block_indices = list(range(rank, num_blocks, size))
        for i in tqdm(block_indices, desc=f"Rank {rank} reading blocks", position=rank):
            block = multiblock[i]
            if block:
                num_points = block.n_points
                scalar_name = list(block.point_data.keys())[0] if block.point_data else None
                mean_scalar = np.mean(block.point_data[scalar_name]) if scalar_name else 0.0
                if rank == 0:
                    # print(f"Block {i}: {num_points} points, {block.n_cells} cells")
                    if block.point_data:
                        for name in block.point_data:
                            arr = block.point_data[name]
                            # print(f"  Scalar '{name}': min={np.min(arr):.3f}, max={np.max(arr):.3f}, mean={np.mean(arr):.3f}")
                local_results.append((i, num_points, mean_scalar))
                local_blocks.append(block)
                local_blocks.set_block_name(len(local_blocks) - 1, str(i))
        # No error if a rank has no blocks to process
        all_results = comm.gather(local_results, root=0)
        # if rank == 0:
        #     # print("Results from all processes:")
        #     for proc_id, results in enumerate(all_results):
        #         for block_id, num_points, mean_scalar in results:
        #             print(f"Process {proc_id}, Block {block_id}: {num_points} points, Mean Scalar: {mean_scalar:.2f}")
        return local_blocks, all_results
    except Exception as e:
        print(f"Rank {rank}: Error reading VTM file: {str(e)}")
        return None, None

def compress_vtm(local_blocks, all_results, output_file):
    """
    Compress the MultiBlock dataset and save as a VTM file with ZLib level 9 compression.
    Gathers all blocks from all ranks and writes the final VTM file on rank 0.
    Removes intermediate VTU files after writing.
    """
    try:
        local_vtu_files = []
        local_block_info = []
        import glob
        for i, block in enumerate(tqdm(local_blocks, desc=f"Rank {rank} compressing blocks", position=rank)):
            block_id = local_blocks.get_block_name(i)
            
            # Choose appropriate writer based on data type
            if isinstance(block, pv.ImageData):
                writer = vtk.vtkXMLImageDataWriter()
                ext = "vti"
            elif isinstance(block, pv.RectilinearGrid):
                writer = vtk.vtkXMLRectilinearGridWriter()
                ext = "vtr"
            elif isinstance(block, pv.StructuredGrid):
                writer = vtk.vtkXMLStructuredGridWriter()
                ext = "vts"
            elif isinstance(block, pv.PolyData):
                writer = vtk.vtkXMLPolyDataWriter()
                ext = "vtp"
            else:  # Default to UnstructuredGrid
                writer = vtk.vtkXMLUnstructuredGridWriter()
                ext = "vtu"
            
            output_file_block = f"{output_file}_block_{block_id}_rank_{rank}.{ext}"
            writer.SetFileName(output_file_block)
            writer.SetInputData(block)
            writer.SetDataModeToBinary()
            writer.SetCompressorTypeToZLib()
            writer.SetCompressionLevel(9)
            writer.Write()
            local_vtu_files.append((block_id, output_file_block))
            local_block_info.append((block_id, block))

        # if rank == 0:
        #     print(f"Rank {rank} wrote {len(local_vtu_files)} blocks")

        # Gather all block info (block_id, block) from all ranks to rank 0
        all_block_info = comm.gather(local_block_info, root=0)
        comm.Barrier()

        if rank == 0:
            output_multiblock = pv.MultiBlock()
            # Flatten the list and sort by block_id for deterministic order
            flat_blocks = []
            for proc_blocks in all_block_info:
                flat_blocks.extend(proc_blocks)
            # Sort by block_id as int if possible, else as str
            def block_id_key(x):
                try:
                    return int(x[0])
                except Exception:
                    return x[0]
            flat_blocks.sort(key=block_id_key)
            for idx, (block_id, block) in enumerate(flat_blocks):
                output_multiblock.append(block)
                output_multiblock.set_block_name(idx, f"Block_{block_id}")
            # print(f"Rank 0 collected {len(flat_blocks)} blocks for VTM")
            
            # Use VTK writer directly for better compression control
            writer = vtk.vtkXMLMultiBlockDataWriter()
            writer.SetFileName(output_file)
            writer.SetInputData(output_multiblock)
            writer.SetDataModeToBinary()
            writer.SetCompressorTypeToZLib()
            writer.SetCompressionLevel(9)
            writer.Write()
            
            print(f"Compressed VTM file written to: {output_file}")

            # Remove intermediate files generated by all ranks
            patterns = [f"{output_file}_block_*.vt?"]  # Matches .vtu, .vti, .vtr, .vts, .vtp
            for pattern in patterns:
                for temp_file in glob.glob(pattern):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        print(f"Warning: Could not remove {temp_file}: {e}")

        comm.Barrier()
    except Exception as e:
        if rank == 0:
            print(f"Error compressing VTM file: {str(e)}")

def create_sample_vtm():
    """
    Create a sample VTM dataset for demonstration and testing.
    Returns the file path of the created VTM file, or None on error.
    """
    multiblock = pv.MultiBlock()
    for i in range(4):
        points = np.array([[0, 0, i], [1, 0, i], [0, 1, i], [0, 0, i + 1]], dtype=np.float32)
        cells = np.array([4, 0, 1, 2, 3], dtype=np.int32)
        grid = pv.UnstructuredGrid(cells, [vtk.VTK_TETRA], points)
        grid.point_data[f"Scalar_{i}"] = np.array([i * 1.0, i * 1.1, i * 1.2, i * 1.3], dtype=np.float32)
        if grid.n_cells == 0 or grid.n_points != 4:
            if rank == 0:
                print(f"Error: Invalid grid in block {i}, n_cells={grid.n_cells}, n_points={grid.n_points}")
            return None
        multiblock.append(grid)
    
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
    """
    Main async loop for parallel VTM read/compress workflow.
    Handles error propagation and synchronization across ranks.
    """
    # --- Choose input file for workflow ---
    # To use a sample VTM file, uncomment the next two lines:
    # input_file = create_sample_vtm() if rank == 0 else None
    # input_file = comm.bcast(input_file, root=0)

    # Use an existing VTM file
    input_file = "/home/dhruv/Desktop/downsampled.vtm"  # Change this to your actual VTM file path

    local_blocks, all_results = read_vtm_parallel(input_file)
    if rank == 0:
        local_error = 0 if (local_blocks is not None and all_results is not None) else 1
    else:
        local_error = 0 if (local_blocks is not None) else 1
    all_errors = comm.gather(local_error, root=0)
    global_error = 0
    if rank == 0:
        global_error = int(any(all_errors))
    global_error = comm.bcast(global_error, root=0)
    if global_error:
        print(f"Rank {rank}: Exiting due to error in reading VTM file.")
        comm.Barrier()
        return

    output_file = "compressed_output.vtm"
    compress_vtm(local_blocks, all_results, output_file)
    await asyncio.sleep(1.0 / 60)

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())
