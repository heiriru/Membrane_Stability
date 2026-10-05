'''
generate a mesh of a rectangle with a circular hole (diameter D = 1), refined around the cylinder and in its wake.
Lengths are in units of the cylinder diameter.

Two geometries are available:
- 'unbounded': large box [-20, 40] x [-20, 20], cylinder at the origin. Used for the linear stability analysis of the
               cylinder wake (reference: Hopf bifurcation at Re_c ~ 46.7, omega_c ~ 0.74).
- 'dfg':       Schaefer-Turek (DFG) benchmark channel [0, 22] x [0, 4.1], cylinder centred at (2, 2).
               Used to validate the steady Navier-Stokes solver (reference drag/lift at Re = 20).

The physical tags follow the convention of generate_mesh/2d/square:
1 = "Volume", 2 = "i" (inflow), 3 = "o" (outflow), 4 = "t" (top), 5 = "b" (bottom), 6 = "c" (circle)

run it with
python3 generate_cylinder_mesh.py [geometry] [resolution factor] [output directory]
example:
SOLUTION_PATH="solution_unbounded_medium"; rm -rf $SOLUTION_PATH; mkdir $SOLUTION_PATH; python3 generate_cylinder_mesh.py unbounded 1.0 $SOLUTION_PATH
'''

import argparse
import sys

import gmsh
import meshio

# add the path where to find the shared modules
module_path = '/home/fenics/shared/modules'
sys.path.append(module_path)

import input_output as io
import mesh as msh

parser = argparse.ArgumentParser()
parser.add_argument("geometry", choices=["unbounded", "dfg"])
parser.add_argument("resolution")
parser.add_argument("output_directory")
args = parser.parse_args()

# the resolution factor multiplies all mesh sizes: 1.0 = 'medium', 1.4 = 'coarse', 0.7 = 'fine'
f = float(args.resolution)
output_directory = io.add_trailing_slash(args.output_directory)
mesh_file = output_directory + "mesh.msh"

# CHANGE PARAMETERS HERE
r = 0.5
if args.geometry == "unbounded":
    x_min, x_max, y_min, y_max = -20.0, 40.0, -20.0, 20.0
    c_r = [0.0, 0.0]
    h_circle = 0.04 * f
    # (size, x_min, x_max, y_half_width around c_r[1], transition thickness)
    wake_boxes = [(0.25 * f, -3.0, 25.0, 3.0, 3.0), (0.6 * f, -6.0, x_max, 7.0, 6.0)]
    h_far = 2.5 * f
else:
    x_min, x_max, y_min, y_max = 0.0, 22.0, 0.0, 4.1
    c_r = [2.0, 2.0]
    h_circle = 0.025 * f
    wake_boxes = [(0.08 * f, 0.5, 8.0, 2.05, 1.0)]
    h_far = 0.2 * f
# CHANGE PARAMETERS HERE

print("geometry = ", args.geometry)
print("box = ", [x_min, x_max, y_min, y_max])
print("r = ", r, " c_r = ", c_r)
print("h_circle = ", h_circle, " h_far = ", h_far)
print(f'output_directory = "{output_directory}"')

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("cylinder")
occ = gmsh.model.occ
rectangle = occ.addRectangle(x_min, y_min, 0.0, x_max - x_min, y_max - y_min)
disk = occ.addDisk(c_r[0], c_r[1], 0.0, r, r)
occ.cut([(2, rectangle)], [(2, disk)])
occ.synchronize()

# sort the boundary curves according to their position
tol = 1e-6
curves = {"i": [], "o": [], "t": [], "b": [], "c": []}
for _, tag in gmsh.model.getEntities(1):
    xmin_c, ymin_c, _, xmax_c, ymax_c, _ = gmsh.model.getBoundingBox(1, tag)
    if abs(xmin_c - x_min) < tol and abs(xmax_c - x_min) < tol:
        curves["i"].append(tag)
    elif abs(xmin_c - x_max) < tol and abs(xmax_c - x_max) < tol:
        curves["o"].append(tag)
    elif abs(ymin_c - y_max) < tol and abs(ymax_c - y_max) < tol:
        curves["t"].append(tag)
    elif abs(ymin_c - y_min) < tol and abs(ymax_c - y_min) < tol:
        curves["b"].append(tag)
    else:
        curves["c"].append(tag)

surfaces = [tag for _, tag in gmsh.model.getEntities(2)]
gmsh.model.addPhysicalGroup(2, surfaces, 1, name="Volume")
for tag, name in zip([2, 3, 4, 5, 6], ["i", "o", "t", "b", "c"]):
    gmsh.model.addPhysicalGroup(1, curves[name], tag, name=name)

# mesh size fields: fine close to the cylinder, graded boxes in the wake, coarse far away
field = gmsh.model.mesh.field
f_dist = field.add("Distance")
field.setNumbers(f_dist, "CurvesList", curves["c"])
field.setNumber(f_dist, "Sampling", 400)
f_thr = field.add("Threshold")
field.setNumber(f_thr, "InField", f_dist)
field.setNumber(f_thr, "SizeMin", h_circle)
field.setNumber(f_thr, "SizeMax", h_far)
field.setNumber(f_thr, "DistMin", 0.1)
field.setNumber(f_thr, "DistMax", 0.6 * (x_max - x_min) if args.geometry == "dfg" else 15.0)
fields = [f_thr]
for size, bx_min, bx_max, half_width, thickness in wake_boxes:
    f_box = field.add("Box")
    field.setNumber(f_box, "VIn", size)
    field.setNumber(f_box, "VOut", h_far)
    field.setNumber(f_box, "XMin", bx_min)
    field.setNumber(f_box, "XMax", bx_max)
    field.setNumber(f_box, "YMin", c_r[1] - half_width)
    field.setNumber(f_box, "YMax", c_r[1] + half_width)
    field.setNumber(f_box, "Thickness", thickness)
    fields.append(f_box)
f_min = field.add("Min")
field.setNumbers(f_min, "FieldsList", fields)
field.setAsBackgroundMesh(f_min)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)

gmsh.model.mesh.generate(2)
gmsh.write(mesh_file)
gmsh.clear()
gmsh.finalize()

mesh_from_file = meshio.read(mesh_file)
line_mesh = msh.create_mesh(mesh_from_file, "line", prune_z=True)
meshio.write(output_directory + "line_mesh.xdmf", line_mesh)
triangle_mesh = msh.create_mesh(mesh_from_file, "triangle", prune_z=True)
meshio.write(output_directory + "triangle_mesh.xdmf", triangle_mesh)

mesh = msh.read_mesh(output_directory + "triangle_mesh.xdmf")
print(f"number of cells = {mesh.num_cells()}, number of vertices = {mesh.num_vertices()}, h_min = {mesh.hmin():.3g}")
