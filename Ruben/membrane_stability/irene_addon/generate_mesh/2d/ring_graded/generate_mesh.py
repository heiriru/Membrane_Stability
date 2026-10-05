'''
Ring mesh (protein circle of radius r inside an outer circle of radius R, both centred at the origin), graded from
h_min at the protein to h_max at the outer circle. Same tags and output format as IRENE's generate_mesh/2d/ring
(surface 1, circle r 2, circle R 3; mesh_metadata.csv with r, R, c_r, c_R), so that IRENE's mesh.read.ring reads it.
Needed for large R/r0, where IRENE's uniform ring mesh would be too large.

Run with (IRENE_ROOT defaults to /home/fenics/shared):
    python3 generate_mesh.py [output directory] --r 1 --R 10 --h_min 0.04 --h_max 0.6
'''
import argparse
import os
import sys

IRENE_ROOT = os.environ.get("IRENE_ROOT", "/home/fenics/shared")
sys.path.append(os.path.join(IRENE_ROOT, "modules"))

import gmsh

parser = argparse.ArgumentParser()
parser.add_argument("output_directory")
parser.add_argument("--r", type=float, default=1.0)
parser.add_argument("--R", type=float, default=10.0)
parser.add_argument("--h_min", type=float, default=0.04)
parser.add_argument("--h_max", type=float, default=0.6)
parser.add_argument("--grading_length", type=float, default=None,
                    help="distance from the protein over which the size grows from h_min to h_max (default (R-r)/2)")
args = parser.parse_args()
# mesh.utils imports runtime_arguments_generate_mesh, which parses sys.argv: give it what it expects
sys.argv = [sys.argv[0], ".", args.output_directory]
import mesh.utils as msh

out = os.path.join(os.path.abspath(args.output_directory), "")
os.makedirs(out, exist_ok=True)
mesh_file = out + "mesh.msh"

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("ring")
occ = gmsh.model.occ
outer = occ.addDisk(0, 0, 0, args.R, args.R)
inner = occ.addDisk(0, 0, 0, args.r, args.r)
occ.cut([(2, outer)], [(2, inner)])
occ.synchronize()
curves = [c for _, c in gmsh.model.getEntities(1)]
radius = {c: max(abs(v) for v in gmsh.model.getBoundingBox(1, c)[:2]) for c in curves}
inner_curves = [c for c in curves if radius[c] < 0.5 * (args.r + args.R)]
outer_curves = [c for c in curves if radius[c] >= 0.5 * (args.r + args.R)]
gmsh.model.addPhysicalGroup(2, [s for _, s in gmsh.model.getEntities(2)], 1, name="Volume")
gmsh.model.addPhysicalGroup(1, inner_curves, 2, name="Circle r")
gmsh.model.addPhysicalGroup(1, outer_curves, 3, name="Circle R")

field = gmsh.model.mesh.field
distance = field.add("Distance")
field.setNumbers(distance, "CurvesList", inner_curves)
field.setNumber(distance, "Sampling", 400)
threshold = field.add("Threshold")
field.setNumber(threshold, "InField", distance)
field.setNumber(threshold, "SizeMin", args.h_min)
field.setNumber(threshold, "SizeMax", args.h_max)
field.setNumber(threshold, "DistMin", 0.0)
field.setNumber(threshold, "DistMax", args.grading_length or 0.5 * (args.R - args.r))
field.setAsBackgroundMesh(threshold)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.model.mesh.generate(2)
gmsh.write(mesh_file)

metadata = {"r": args.r, "R": args.R, "c_r": [0, 0, 0], "c_R": [0, 0, 0], "resolution": args.h_min,
            "h_max": args.h_max, "file_format": "xdmf"}
msh.full_write(mesh_file, ["triangle", "line"], metadata, out, True)
gmsh.finalize()
