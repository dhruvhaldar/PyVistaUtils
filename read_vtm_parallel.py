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
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"VTM file not found: {file_path}")
        
        multiblock = pv.read(file_path)
        num_blocks = multiblock.n_blocks
        if rank == 0:
            print(f"Reading VTM file with {num_blocks} blocks")
        
        local_blocks = pv.MultiBlock()
        local_results = []
        for i in range(rank, num_blocks, size):
            block = multiblock[i]
            if block:
                num_points = block.n_points
                scalar_name = list(block.point_data.keys())[0] if block.point_data else None
                mean_scalar = np.mean(block.point_data[scalar_name]) if scalar_name else 0.0
                local_results.append((i, num_points, mean_scalar))
                local_blocks.append(block)
                local_blocks.set_block_name(len(local_blocks) - 1, str(i))
        # No error if a rank has no blocks to process
        all_results = comm.gather(local_results, root=0)
        if rank == 0:
            print("Results from all processes:")
            for proc_id, results in enumerate(all_results):
                for block_id, num_points, mean_scalar in results:
                    print(f"Process {proc_id}, Block {block_id}: {num_points} points, Mean Scalar: {mean_scalar:.2f}")
        return local_blocks, all_results
    except Exception as e:
        print(f"Rank {rank}: Error reading VTM file: {str(e)}")
        return None, None

def compress_vtm(local_blocks, all_results, output_file):
    """Compress the MultiBlock dataset and save as a VTM file with ZLib level 9."""
    try:
        local_vtu_files = []
        local_block_info = []
        for i, block in enumerate(local_blocks):
            block_id = local_blocks.get_block_name(i)
            vtu_file = f"{output_file}_block_{block_id}_rank_{rank}.vtu"
            writer = vtk.vtkXMLUnstructuredGridWriter()
            writer.SetFileName(vtu_file)
            writer.SetInputData(block)
            writer.SetDataModeToBinary()
            writer.SetCompressorTypeToZLib()
            writer.SetCompressionLevel(9)
            writer.Write()
            local_vtu_files.append((block_id, vtu_file))
            # Also gather the block itself for final VTM assembly
            local_block_info.append((block_id, block))
        if rank == 0:
            print(f"Rank {rank} wrote {len(local_vtu_files)} VTU files")

        # Gather all block info (block_id, block) from all ranks to rank 0
        all_block_info = comm.gather(local_block_info, root=0)
        comm.Barrier()

        if rank == 0:
            output_multiblock = pv.MultiBlock()
            block_names = []
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
            print(f"Rank 0 collected {len(flat_blocks)} blocks for VTM")
            output_multiblock.save(output_file, binary=True)
            print(f"Compressed VTM file written to: {output_file}")

        comm.Barrier()
    except Exception as e:
        if rank == 0:
            print(f"Error compressing VTM file: {str(e)}")

def create_sample_vtm():
    """Create a sample VTM dataset for demonstration."""
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
    """Main async loop for Pyodide compatibility."""
    error_flag = 0
    input_file = create_sample_vtm() if rank == 0 else None
    input_file = comm.bcast(input_file, root=0)
    if input_file is None:
        error_flag = 1
    # Broadcast error flag to all ranks
    error_flag = comm.bcast(error_flag, root=0)
    if error_flag:
        print(f"Rank {rank}: Exiting due to error in file creation.")
        comm.Barrier()
        return

    print(f"Rank {rank}: before Barrier 1")
    comm.Barrier()
    print(f"Rank {rank}: after Barrier 1")

    local_blocks, all_results = read_vtm_parallel(input_file)
    print(f"Rank {rank}: local_blocks type: {type(local_blocks)}, len: {len(local_blocks) if local_blocks is not None else 'None'}")
    print(f"Rank {rank}: all_results type: {type(all_results)}, value: {all_results}")
    # Only treat as error if local_blocks is None, or if rank 0 and all_results is None
    if rank == 0:
        local_error = 0 if (local_blocks is not None and all_results is not None) else 1
    else:
        local_error = 0 if (local_blocks is not None) else 1
    # Gather error flags from all ranks
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
