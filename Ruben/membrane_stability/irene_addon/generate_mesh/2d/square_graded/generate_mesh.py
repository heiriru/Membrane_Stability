'''
Square (or rectangular, --W) mesh with a circular hole (the protein inclusion, PI), graded from h_min at the PI to h_max far from it. Same
tags and output format as IRENE's generate_mesh/2d/square (surface 1; left 2, right 3, top 4, bottom 5, circle 6;
mesh_metadata.csv with L, h, r, c_r), so that IRENE's mesh.read.square reads it. Needed for the geometry of Ferraro &
Castellana, PRE (L = 100 r0), where a uniform mesh would be too large.

Run with (IRENE_ROOT defaults to /home/fenics/shared):
    python3 generate_mesh.py [output directory] --L 100 --r 1 --h_min 0.1 --h_max 3
'''
import argparse
import os
import sys

IRENE_ROOT = os.environ.get("IRENE_ROOT", "/home/fenics/shared")
sys.path.append(os.path.join(IRENE_ROOT, "modules"))

import gmsh

parser = argparse.ArgumentParser()
parser.add_argument("output_directory")
parser.add_argument("--L", type=float, default=100.0)
parser.add_argument("--W", type=float, default=None, help="width (y extent) of the rectangle (default L: square)")
parser.add_argument("--r", type=float, default=1.0)
parser.add_argument("--h_min", type=float, default=0.1)
parser.add_argument("--h_max", type=float, default=3.0)
parser.add_argument("--grading_length", type=float, default=None,
                    help="distance from the PI over which the size grows from h_min to h_max (default L/4)")
args = parser.parse_args()
# mesh.utils imports runtime_arguments_generate_mesh, which parses sys.argv: give it what it expects
sys.argv = [sys.argv[0], ".", args.output_directory]
import mesh.utils as msh

out = os.path.join(os.path.abspath(args.output_directory), "")
os.makedirs(out, exist_ok=True)
mesh_file = out + "mesh.msh"
L, r = args.L, args.r
W = args.W if args.W else L
c, cy = 0.5 * L, 0.5 * W

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("square")
geo = gmsh.model.geo
p = [geo.addPoint(0, 0, 0), geo.addPoint(L, 0, 0), geo.addPoint(L, W, 0), geo.addPoint(0, W, 0)]
bottom, right, top, left = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
pc = geo.addPoint(c, cy, 0)
q = [geo.addPoint(c + r, cy, 0), geo.addPoint(c, cy + r, 0), geo.addPoint(c - r, cy, 0), geo.addPoint(c, cy - r, 0)]
arcs = [geo.addCircleArc(q[i], pc, q[(i + 1) % 4]) for i in range(4)]
outer = geo.addCurveLoop([bottom, right, top, left])
hole = geo.addCurveLoop(arcs)
surface = geo.addPlaneSurface([outer, hole])
geo.synchronize()
gmsh.model.addPhysicalGroup(2, [surface], 1, name="Volume")
gmsh.model.addPhysicalGroup(1, [left], 2, name="l")
gmsh.model.addPhysicalGroup(1, [right], 3, name="r")
gmsh.model.addPhysicalGroup(1, [top], 4, name="t")
gmsh.model.addPhysicalGroup(1, [bottom], 5, name="b")
gmsh.model.addPhysicalGroup(1, arcs, 6, name="c")

field = gmsh.model.mesh.field
distance = field.add("Distance")
field.setNumbers(distance, "CurvesList", arcs)
field.setNumber(distance, "Sampling", 400)
threshold = field.add("Threshold")
field.setNumber(threshold, "InField", distance)
field.setNumber(threshold, "SizeMin", args.h_min)
field.setNumber(threshold, "SizeMax", args.h_max)
field.setNumber(threshold, "DistMin", 0.0)
field.setNumber(threshold, "DistMax", args.grading_length or 0.25 * min(L, W))
field.setAsBackgroundMesh(threshold)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.model.mesh.generate(2)
gmsh.write(mesh_file)

metadata = {"L": L, "h": W, "r": r, "c_r": [c, cy, 0], "resolution": args.h_min, "h_max": args.h_max,
            "file_format": "xdmf"}
msh.full_write(mesh_file, ["triangle", "line"], metadata, out, True)
gmsh.finalize()
