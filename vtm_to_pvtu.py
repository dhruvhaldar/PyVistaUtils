# Author: Dhruv Haldar
# Description: Convert a VTM file with multiple blocks into a PVTU file for parallel read/write.
# Created: 2024-06-09
import os
import pyvista as pv
from xml.etree.ElementTree import Element, SubElement, ElementTree
import numpy as np
from tqdm import tqdm

vtm_path = "/home/dhruv/Desktop/downsampled.vtm"
vtu_dir = "vtu_parts"
pvtu_path = "combined_output.pvtu"

os.makedirs(vtu_dir, exist_ok=True)

# Read multiblock once
print(f"Reading: {vtm_path}")
multiblock = pv.read(vtm_path, progress_bar=True)

vtu_files = []

print("Converting blocks to .vtu...")
for i, block in enumerate(tqdm(multiblock, desc="Processing blocks", unit="block")):
    if block is None:
        continue
    ugrid = block.cast_to_unstructured_grid()
    vtu_filename = f"block_{i}.vtu"
    vtu_path = os.path.join(vtu_dir, vtu_filename)
    ugrid.save(vtu_path)
    vtu_files.append(vtu_filename)

print(f"Saved {len(vtu_files)} .vtu files")

sample_vtu = pv.read(os.path.join(vtu_dir, vtu_files[0])) # Use first block to declare arrays in .pvtu

# Create the .pvtu XML structure
vtkfile = Element("VTKFile", type="PUnstructuredGrid", version="1.0", byte_order="LittleEndian")
punstructured = SubElement(vtkfile, "PUnstructuredGrid")

# Point data arrays
p_point_data = SubElement(punstructured, "PPointData", Scalars="velocity_magnitude")
for name, array in sample_vtu.point_data.items():
    dtype = "Float32" if array.dtype in [np.float32, np.float64] else "UInt8"
    ncomp = array.shape[1] if array.ndim > 1 else 1
    SubElement(p_point_data, "PDataArray", type=dtype, Name=name, NumberOfComponents=str(ncomp), format="appended")

SubElement(punstructured, "PCellData")  # Cell Data

# Points
ppoints = SubElement(punstructured, "PPoints")
SubElement(ppoints, "PDataArray", type="Float32", Name="Points", NumberOfComponents="3", format="appended")

# Add each piece
for fname in sorted(vtu_files):
    SubElement(punstructured, "Piece", Source=os.path.join(vtu_dir, fname))

# Write final .pvtu
ElementTree(vtkfile).write(pvtu_path, encoding="utf-8", xml_declaration=True)
print(f"Wrote combined .pvtu file: {pvtu_path}")
