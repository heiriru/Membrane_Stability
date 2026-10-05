'''
Remeshing of the parametric surface (param_problem.py) when the parametrization degenerates (steep walls, overhangs).

The surface is the piecewise-linear map X(u) on the current reference mesh. A remesh does three things:
  1. reparametrize: new reference coordinates u' of the current vertices from a discrete harmonic map of the surface
     onto the reference domain (mean-value weights, which are positive; log-polar coordinates around the PI, which
     keeps vertices out of the hole; the outer square and the PI rim are kept fixed, so the boundary conditions and
     IRENE's boundary geometry still hold).
     Unlike the quasi-Monge parametrization, where an overhang is squeezed onto almost no reference area, the harmonic
     map gives the folded wall a reference area of the order of its true area;
  2. build a new reference mesh (remesh_gmsh.py, same geometry and tags) whose element size, mapped to the surface, is
     the size of the original mesh (h_min at the PI, graded to h_max) and at most 0.4 / |kappa_max| on curved parts;
  3. write the transfer data: the old triangulation in the new coordinates u' and the nodal values of X (current and
     previous step), w, sigma, mu and of the tangential velocity as a 3D vector V = v^i e_i (gauge independent), which
     param_snap.py --transfer interpolates onto the new mesh.

    python3 param_remesh.py square_b [old mesh dir] [new mesh dir] --run [run dir with checkpoint.npz]
'''
import irene_paths  # noqa: F401

import argparse
import json
import os
import subprocess
import sys

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spla
from fenics import Function, VectorFunctionSpace, as_vector, project, vertex_to_dof_map

import runtime_arguments as rarg
import param_problem as pp
import differential_geometry.manifold.geometry as geo

parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--h_min", type=float, default=0.15)
parser.add_argument("--h_max", type=float, default=3.0)
parser.add_argument("--grading", type=float, default=25.0)
parser.add_argument("--curv_factor", type=float, default=0.4)
parser.add_argument("--max_vertices", type=int, default=9000)
parser.add_argument("--blend", type=float, default=1.0, help="fraction of the harmonic displacement applied")
parser.add_argument("--rho_fix", type=float, default=2.5,
                    help="vertices closer than this to the PI centre keep their reference coordinates (the rim "
                         "condition on the frame, E = [[1, 0, t r_x], [0, 1, t r_y]], assumes the horizontal projection there)")
parser.add_argument("--new_mesh", action="store_true",
                    help="generate a new gmsh mesh (default: same triangulation with the new reference coordinates, so the "
                         "surface is transferred exactly; a new mesh inherits the kinks of the old facets)")
parser.add_argument("--no_reparam", action="store_true", help="test: new mesh, old parametrization")
parser.add_argument("--identity", action="store_true", help="control: no reparametrization, no new mesh (transfer data on "
                                                            "the old mesh, to test the transfer)")
options = parser.parse_args(rarg.unknown_args)
new_dir = rarg.args.output_directory
R = np.load(os.path.join(options.run, "checkpoint.npz"), allow_pickle=True)
mesh = pp.mesh
u = mesh.coordinates().copy()
tri = mesh.cells().copy()
P1 = pp.P1
v2d = vertex_to_dof_map(P1)
P1v = VectorFunctionSpace(mesh, "P", 1, dim=3)


def nodal(expr):
    return project(expr, P1).vector().get_local()[v2d]


def fields(vec):
    pp.psi.vector()[:] = vec
    out = dict(X=np.array([nodal(pp.X[k]) for k in range(3)]).T, w=nodal(pp.w), sigma=nodal(pp.sigma),
               E=np.array([[nodal(pp.E[a, k]) for k in range(3)] for a in range(2)]).transpose(2, 0, 1),
               mu=nodal(pp.mu), K=nodal(geo.K(pp.E)),
               V=np.array([nodal(pp.v[0] * pp.E[0, k] + pp.v[1] * pp.E[1, k]) for k in range(3)]).T)
    return out


F_n = fields(R["psi_n"])
F_nm1 = fields(R["psi_nm1"])
X = F_n["X"]

# 1. harmonic reparametrization with mean-value weights on the surface
n = len(u)
rows, cols, vals = [], [], []
for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
    i, j, k = tri[:, a], tri[:, b], tri[:, c]
    eij, eik = X[j] - X[i], X[k] - X[i]
    lij, lik = np.linalg.norm(eij, axis=1), np.linalg.norm(eik, axis=1)
    cos = np.clip(np.sum(eij * eik, axis=1) / (lij * lik), -1, 1)
    t2 = np.tan(0.5 * np.arccos(cos))
    rows += [i, i]
    cols += [j, k]
    vals += [t2 / lij, t2 / lik]
W = sps.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
edges = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
ue, cnt = np.unique(edges, axis=0, return_counts=True)
bnd = np.unique(ue[cnt == 1].ravel())
bnd = np.union1d(bnd, np.flatnonzero(np.hypot(u[:, 0] - pp.c_r[0], u[:, 1] - pp.c_r[1]) < options.rho_fix))
inner = np.setdiff1d(np.arange(n), bnd)
Lap = (sps.diags(np.asarray(W.sum(axis=1)).ravel()) - W).tocsr()
# The domain has a hole (the PI), so a harmonic map of u itself is not guaranteed to be injective (it pushes vertices
# next to the rim into the hole). Instead s = ln(rho / r) and the angle theta around the PI centre are mapped
# harmonically (both are harmonic functions of u, so flat parts are reproduced); the rim is the straight line s = 0 and
# theta is multivalued: an edge that crosses the cut theta = +-pi carries a jump 2 pi.
c = np.array(pp.c_r)
rel = u - c
s_old = np.log(np.hypot(rel[:, 0], rel[:, 1]) / pp.r_pi)
th_old = np.arctan2(rel[:, 1], rel[:, 0])
Wc = W.tocoo()
kij = -np.round((th_old[Wc.col] - th_old[Wc.row]) / (2 * np.pi))
rhs_th = np.zeros(n)
np.add.at(rhs_th, Wc.row, 2 * np.pi * Wc.data * kij)
A_ii = Lap[inner][:, inner].tocsc()
A_ib = Lap[inner][:, bnd]
s_new, th_new = s_old.copy(), th_old.copy()
s_new[inner] = spla.spsolve(A_ii, -A_ib @ s_old[bnd])
th_new[inner] = spla.spsolve(A_ii, rhs_th[inner] - A_ib @ th_old[bnd])
u_new = c + pp.r_pi * np.exp(s_new)[:, None] * np.stack([np.cos(th_new), np.sin(th_new)], axis=1)
u_new[bnd] = u[bnd]
if options.identity or options.no_reparam:
    u_new = u.copy()


def signed_area(uu):
    p, q, r = uu[tri[:, 0]], uu[tri[:, 1]], uu[tri[:, 2]]
    return 0.5 * ((q[:, 0] - p[:, 0]) * (r[:, 1] - p[:, 1]) - (q[:, 1] - p[:, 1]) * (r[:, 0] - p[:, 0]))


s0 = np.sign(signed_area(u))
blend = options.blend
while np.any(np.sign(signed_area((1 - blend) * u + blend * u_new)) != s0) and blend > 0.05:
    blend *= 0.8
u_new = (1 - blend) * u + blend * u_new
print(f"reparametrization: blend = {blend:.3f}, inverted: {int(np.sum(np.sign(signed_area(u_new)) != s0))}", flush=True)


# 2. element size: target size on the surface / stretch of the map u' -> X
def stretch(uu):
    Du = np.stack([uu[tri[:, 1]] - uu[tri[:, 0]], uu[tri[:, 2]] - uu[tri[:, 0]]], axis=2)       # (m, 2, 2)
    DX = np.stack([X[tri[:, 1]] - X[tri[:, 0]], X[tri[:, 2]] - X[tri[:, 0]]], axis=2)           # (m, 3, 2)
    J = DX @ np.linalg.inv(Du)
    s = np.linalg.svd(J, compute_uv=False)
    return s[:, 0], s[:, 1]


s_old = stretch(u)
s_new = stretch(u_new)
lam = np.zeros(n)
np.maximum.at(lam, tri.ravel(), np.repeat(s_new[0], 3))
dist = np.maximum(np.linalg.norm(u_new - c, axis=1) - pp.r_pi, 0.0)
h3 = options.h_min + (options.h_max - options.h_min) * np.minimum(1.0, dist / options.grading)
H, K = F_n["mu"], F_n["K"]
kmax = np.abs(H) + np.sqrt(np.maximum(H ** 2 - K, 0.0))
h3 = np.minimum(h3, np.maximum(options.h_min, options.curv_factor / np.maximum(kmax, 1e-6)))
size = np.clip(h3 / lam, 0.02, options.h_max)
area = np.abs(signed_area(u_new))
n_est = np.sum(area / (0.433 * np.mean(size[tri], axis=1) ** 2)) / 2
if n_est > options.max_vertices:
    size = np.minimum(size * np.sqrt(n_est / options.max_vertices), options.h_max)
print(f"stretch (largest singular value / anisotropy): old {s_old[0].max():.2f} / {(s_old[0] / s_old[1]).max():.1f}, "
      f"new {s_new[0].max():.2f} / {(s_new[0] / s_new[1]).max():.1f}; estimated vertices {n_est:.0f}", flush=True)
os.makedirs(new_dir, exist_ok=True)
np.savez(os.path.join(new_dir, "size.npz"), u=u_new, triangles=tri, size=size, L=pp.rmsh.parameters["L"], r=pp.r_pi,
         c=c)
env = dict(os.environ)
if options.new_mesh and not options.identity:
  subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "remesh_gmsh.py"),
                  os.path.join(new_dir, "size.npz"), new_dir], check=True, env=env)

if not options.new_mesh:
    # same triangulation, new reference coordinates: write it with IRENE's mesh writer (gmsh discrete entities)
    from fenics import facets
    lines = {tag: [] for tag in (2, 3, 4, 5, 6)}
    for f in facets(mesh):
        tag = int(pp.rmsh.mf[f])
        if tag in lines:
            lines[tag].append(f.entities(0).copy())
    np.savez(os.path.join(new_dir, "same_mesh.npz"), u=u_new, triangles=tri,
             **{f"lines_{k}": np.array(v) for k, v in lines.items()}, L=pp.rmsh.parameters["L"], r=pp.r_pi, c=c)
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "remesh_gmsh.py"),
                    os.path.join(new_dir, "same_mesh.npz"), new_dir, "--discrete"], check=True, env=env)

# 3. transfer data. The frame of the solver, E_a = dX/du^a (a smooth nodal field), is carried over with the vertex
# Jacobian du/du' (least squares over the one-ring of each vertex): E'_b = (du^a/du'^b) E_a. Re-projecting grad X in the
# new coordinates instead averages the tangent planes across the sharp fold differently and gives a frame that the
# Newton iteration does not recover from.
nbr = [[] for _ in range(n)]
for a_, b_ in ue:
    nbr[a_].append(b_)
    nbr[b_].append(a_)
E_new = np.zeros_like(F_n["E"])
for i in range(n):
    du_new = u_new[nbr[i]] - u_new[i]
    du_old = u[nbr[i]] - u[i]
    Jt = np.linalg.lstsq(du_new, du_old, rcond=None)[0]           # du_old ~ du_new @ Jt,  Jt[b, a] = du^a / du'^b
    E_new[i] = Jt @ F_n["E"][i]
np.savez(os.path.join(new_dir, "transfer.npz"), u=u_new, triangles=tri, X_n=F_n["X"], X_nm1=F_nm1["X"], w=F_n["w"],
         E=E_new,
         sigma=F_n["sigma"], mu=F_n["mu"], V=F_n["V"], t=R["t"], dt=R["dt"], h=R["h"], h_prev=R["h_prev"],
         old_mesh=rarg.args.input_directory, run=options.run)
json.dump(dict(blend=blend, stretch_old=float(s_old[0].max()), stretch_new=float(s_new[0].max()),
               aniso_old=float((s_old[0] / s_old[1]).max()), aniso_new=float((s_new[0] / s_new[1]).max()),
               n_est=float(n_est), t=float(R["t"]), h=float(R["h"])),
          open(os.path.join(new_dir, "remesh.json"), "w"), indent=1)
print("transfer written to", new_dir, flush=True)
