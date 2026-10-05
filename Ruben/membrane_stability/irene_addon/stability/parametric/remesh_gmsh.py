'''
New reference mesh for the parametric surface (param_remesh.py): IRENE's square with the circular PI hole (same
geometry, tags and output format as ../../generate_mesh/2d/square_graded), with the element size given by a background
field on the old triangulation (reference coordinates u' of the reparametrized surface, size per vertex).

    python3 remesh_gmsh.py [size.npz: u (n, 2), triangles (m, 3), size (n,), L, r, c] [output directory]
'''
import os
import sys

IRENE_ROOT = os.environ.get("IRENE_ROOT", "/home/fenics/shared")
sys.path.append(os.path.join(IRENE_ROOT, "modules"))

import gmsh
import numpy as np

d = np.load(sys.argv[1])
out = os.path.join(os.path.abspath(sys.argv[2]), "")
discrete = "--discrete" in sys.argv
sys.argv = [sys.argv[0], ".", out]
import mesh.utils as msh  # noqa: E402

os.makedirs(out, exist_ok=True)
L, r = float(d["L"]), float(d["r"])
c, cy = float(d["c"][0]), float(d["c"][1])
u, tri = d["u"], d["triangles"]

if discrete:
    # the given triangulation itself (new vertex coordinates, same topology and boundary tags)
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("square")
    gmsh.model.addDiscreteEntity(2, 1)
    nodes = np.arange(1, len(u) + 1)
    gmsh.model.mesh.addNodes(2, 1, nodes.tolist(), np.hstack([u, np.zeros((len(u), 1))]).ravel().tolist())
    gmsh.model.mesh.addElementsByType(1, 2, [], (tri + 1).ravel().tolist())
    gmsh.model.addPhysicalGroup(2, [1], 1, name="Volume")
    for tag, name in ((2, "l"), (3, "r"), (4, "t"), (5, "b"), (6, "c")):
        seg = d[f"lines_{tag}"]
        gmsh.model.addDiscreteEntity(1, tag)
        gmsh.model.mesh.addElementsByType(tag, 1, [], (seg + 1).ravel().tolist())
        gmsh.model.addPhysicalGroup(1, [tag], tag, name=name)
    mesh_file = out + "mesh.msh"
    gmsh.write(mesh_file)
    metadata = {"L": L, "h": L, "r": r, "c_r": [c, cy, 0], "resolution": 0.0, "h_max": 0.0, "file_format": "xdmf"}
    msh.full_write(mesh_file, ["triangle", "line"], metadata, out, True)
    gmsh.finalize()
    print("wrote (same triangulation)", out)
    sys.exit(0)
size = d["size"]

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("square")
geo = gmsh.model.geo
n_arc = int(d["n_arc"]) if "n_arc" in d.files else 0
p = [geo.addPoint(0, 0, 0), geo.addPoint(L, 0, 0), geo.addPoint(L, L, 0), geo.addPoint(0, L, 0)]
bottom, right, top, left = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
pc = geo.addPoint(c, cy, 0)
q = [geo.addPoint(c + r, cy, 0), geo.addPoint(c, cy + r, 0), geo.addPoint(c - r, cy, 0), geo.addPoint(c, cy - r, 0)]
arcs = [geo.addCircleArc(q[i], pc, q[(i + 1) % 4]) for i in range(4)]
outer = geo.addCurveLoop([bottom, right, top, left])
hole = geo.addCurveLoop(arcs)
surface = geo.addPlaneSurface([outer, hole])
geo.synchronize()
if n_arc:
    for a in arcs:
        gmsh.model.mesh.setTransfiniteCurve(a, n_arc + 1)
gmsh.model.addPhysicalGroup(2, [surface], 1, name="Volume")
gmsh.model.addPhysicalGroup(1, [left], 2, name="l")
gmsh.model.addPhysicalGroup(1, [right], 3, name="r")
gmsh.model.addPhysicalGroup(1, [top], 4, name="t")
gmsh.model.addPhysicalGroup(1, [bottom], 5, name="b")
gmsh.model.addPhysicalGroup(1, arcs, 6, name="c")

view = gmsh.view.add("size")
data = np.hstack([u[tri][:, :, 0], u[tri][:, :, 1], np.zeros((len(tri), 3)), size[tri]]).ravel().tolist()
gmsh.view.addListData(view, "ST", len(tri), data)
field = gmsh.model.mesh.field
bg = field.add("PostView")
field.setNumber(bg, "ViewTag", view)
field.setAsBackgroundMesh(bg)
gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.model.mesh.generate(2)
gmsh.model.mesh.optimize("Laplace2D")
mesh_file = out + "mesh.msh"
gmsh.write(mesh_file)
metadata = {"L": L, "h": L, "r": r, "c_r": [c, cy, 0], "resolution": float(size.min()), "h_max": float(size.max()),
            "file_format": "xdmf"}
msh.full_write(mesh_file, ["triangle", "line"], metadata, out, True)
gmsh.finalize()
print("wrote", out)
