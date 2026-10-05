'''
Read a mesh generated with generate_mesh/2d/cylinder/generate_cylinder_mesh.py (rectangle with a circular hole).
Same tags and names as read_mesh_square, but the geometry (box, centre and radius of the circle) is read from the mesh
itself, so that the same module works for both the 'unbounded' and the 'dfg' geometry.
'''
from fenics import *
import dolfin
import numpy as np

import load_mesh as lmsh
import runtime_arguments as rarg

# read the triangles
mvc = MeshValueCollection("size_t", lmsh.mesh, lmsh.mesh.topology().dim())
with XDMFFile((rarg.args.input_directory) + "/triangle_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
sf = dolfin.cpp.mesh.MeshFunctionSizet(lmsh.mesh, mvc)

# read the lines
mvc = MeshValueCollection("size_t", lmsh.mesh, lmsh.mesh.topology().dim() - 1)
with XDMFFile((rarg.args.input_directory) + "/line_mesh.xdmf") as infile:
    infile.read(mvc, "name_to_read")
mf = dolfin.cpp.mesh.MeshFunctionSizet(lmsh.mesh, mvc)

r_mesh = lmsh.mesh.hmin()

# geometry read from the mesh
coordinates = lmsh.mesh.coordinates()
x_min, y_min = coordinates.min(axis=0)
x_max, y_max = coordinates.max(axis=0)
L = x_max - x_min
h = y_max - y_min
circle_vertices = np.unique(np.concatenate(
    [facet.entities(0) for facet in facets(lmsh.mesh) if mf[facet] == 6]))
circle_points = coordinates[circle_vertices]
c_r = list(circle_points.mean(axis=0))
r = float(np.mean(np.linalg.norm(circle_points - c_r, axis=1)))

dx = Measure("dx", domain=lmsh.mesh, subdomain_data=sf, subdomain_id=1)
ds_l = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=2)
ds_r = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=3)
ds_t = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=4)
ds_b = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=5)
ds_circle = Measure("ds", domain=lmsh.mesh, subdomain_data=mf, subdomain_id=6)
ds_lr = ds_l + ds_r
ds_tb = ds_t + ds_b
ds_square = ds_lr + ds_tb
ds = ds_square + ds_circle

# boundaries as strings, with the same names as in read_mesh_square
boundary = 'on_boundary'
boundary_l = f'on_boundary && near(x[0], {x_min})'
boundary_r = f'on_boundary && near(x[0], {x_max})'
boundary_lr = f'on_boundary && (near(x[0], {x_min}) || near(x[0], {x_max}))'
boundary_tb = f'on_boundary && (near(x[1], {y_min}) || near(x[1], {y_max}))'
boundary_circle = f'on_boundary && sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) < {r + 0.25 * r}'
boundary_square = f'on_boundary && sqrt(pow(x[0] - {c_r[0]}, 2) + pow(x[1] - {c_r[1]}, 2)) > {r + 0.25 * r}'

# sanity checks of the tags: lengths of the boundaries and area of the domain
lengths = {name: assemble(Constant(1.0) * measure) for name, measure in
           [('l', ds_l), ('r', ds_r), ('t', ds_t), ('b', ds_b), ('circle', ds_circle)]}
exact = {'l': h, 'r': h, 't': L, 'b': L, 'circle': 2 * np.pi * r}
area = assemble(Constant(1.0) * dx)
print(f"Cylinder mesh: box = [{x_min:g}, {x_max:g}] x [{y_min:g}, {y_max:g}], c_r = [{c_r[0]:.4g}, {c_r[1]:.4g}], r = {r:.4g}, "
      f"cells = {lmsh.mesh.num_cells()}, h_min = {r_mesh:.3g}")
for name in lengths:
    rel = abs(lengths[name] - exact[name]) / exact[name]
    print(f"\tlength of boundary {name} = {lengths[name]:.6g}, should be {exact[name]:.6g}, relative error = {rel:.2e}")
    assert rel < 1e-2, f"boundary {name} is not tagged correctly"
print(f"\tarea = {area:.6g}, should be {L * h - np.pi * r ** 2:.6g}")
