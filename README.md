# PyVistaUtils

A collection of utilities for working with 3D mesh files using [PyVista](https://github.com/pyvista/pyvista), with support for parallel VTM reading and STL scaling.

## Features

- **Parallel VTM Reader**: Efficiently read and analyze `.vtm` (VTK MultiBlock) files in parallel using MPI.
- **STL Scaling Utility**: Scale STL mesh files by a uniform or non-uniform factor.

## Requirements

- Python 3.8+
- [PyVista](https://github.com/pyvista/pyvista)
- [mpi4py](https://github.com/mpi4py/mpi4py)
- [VTK](https://vtk.org/)
- See `requirements.txt` for the full list.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Parallel VTM Reader

Read and analyze a VTM file in parallel using MPI:

```bash
mpiexec -n 4 python read_vtm_parallel.py <path_to_file.vtm>
```

- Each MPI rank processes a subset of blocks.
- Statistics are gathered and printed by rank 0.

### 2. STL Scaling

Scale an STL file by a factor (uniform or non-uniform):

```bash
python scale_stl.py
```

Edit the configuration at the bottom of `scale_stl.py` to set input/output filenames and the scaling factor.

## Example

```python
from scale_stl import scale_stl_pyvista

scale_stl_pyvista('input.stl', 'output_scaled.stl', 2.0)
```

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.
