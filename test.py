import pyvista as pv

vtu = pv.read("vtu_parts/block_0.vtu")
print(vtu.point_data.keys())  # What scalar/vector arrays are here?