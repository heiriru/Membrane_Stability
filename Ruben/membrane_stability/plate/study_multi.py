'''
Several PIs in the PRE box (L = 100 r0, inflow on the left, Gamma = sigma0 L^2 / kappa = 25, no in-plane friction:
self-adjoint, so the threshold is the generalized eigenvalue problem in the inflow velocity). Configurations:
    single PI (reference); pairs in tandem (along the flow) and side by side (across) at centre distances d;
    a row of PIs across the flow; a random cluster.
For each: threshold SL_c = v0_c L (the first mode), the next eigenvalues (in-phase / anti-phase splitting of the pair
modes), the drag on each PI, the heights h_k of the rigid PIs in the critical mode (relative phase), and the mode.

    python3 study_multi.py [out] [--configs ...]
'''
import argparse
import json
import os
import time

import numpy as np

import plate_lib as pl

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--L", type=float, default=100.0)
parser.add_argument("--gamma", type=float, default=25.0)
parser.add_argument("--nev", type=int, default=6)
parser.add_argument("--configs", default="all")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
L = args.L
c = L / 2
configs = {"single": [(c, c)]}
for d in (2.5, 3, 4, 6, 10, 15, 20, 30):
    configs[f"tandem_d{d:g}"] = [(c - d / 2, c), (c + d / 2, c)]
    configs[f"side_d{d:g}"] = [(c, c - d / 2), (c, c + d / 2)]
for n, d in ((3, 10), (5, 10), (5, 5)):
    configs[f"row{n}_d{d:g}"] = [(c, c + (k - (n - 1) / 2) * d) for k in range(n)]
    configs[f"file{n}_d{d:g}"] = [(c + (k - (n - 1) / 2) * d, c) for k in range(n)]
rng = np.random.default_rng(1)
pts = []
while len(pts) < 8:
    p = c + rng.uniform(-12, 12, 2)
    if all(np.hypot(*(p - q)) > 3.0 for q in pts):
        pts.append(p)
configs["cluster8"] = [tuple(p) for p in pts]
todo = list(configs) if args.configs == "all" else args.configs.split(",")
t0 = time.time()
results = {}
sigma0 = args.gamma / L ** 2
for name in todo:
    holes = configs[name]
    mdir = pl.make_mesh(os.path.join(args.out, f"mesh_{name}"), L, L, holes, h_min=0.15, h_max=3.0, grading=25.0)
    geo = pl.Geometry(mdir)
    base = pl.BaseFlow(geo)
    plate = pl.Plate(base, pi_bc="rigid")
    pairs = plate._eig(plate.a_m + plate.a_bend + sigma0 * plate.a_tens, -plate.a_flow, 0.0, args.nev)
    pos = sorted([p for p in pairs if p.eigenvalue.real > 0 and abs(p.eigenvalue.imag) < 1e-8 * abs(p.eigenvalue)],
                 key=lambda p: p.eigenvalue.real)
    from dolfin import assemble, dot, FacetNormal
    n = FacetNormal(geo.mesh)
    drags = [-assemble(dot(base.T1, n)[0] * geo.ds(pl.HOLE0 + k)) for k in range(len(holes))]
    modes = []
    x, y, tri = pl.p1_triangulation(geo)
    for k, p in enumerate(pos[:3]):
        z, _ = plate.z_field(p.x_real)
        z /= z[np.argmax(np.abs(z))]
        # heights of the PIs (rim values) in the normalized mode
        heights = [float(np.mean(z[np.hypot(x - hx, y - hy) < 1.0 + 1e-6])) for hx, hy in holes]
        modes.append(dict(SL=float(p.eigenvalue.real * L), PI_heights=heights,
                          centroid=[float(np.sum(x * z ** 2) / np.sum(z ** 2) - c),
                                    float(np.sum(y * z ** 2) / np.sum(z ** 2) - c)]))
        np.savez(os.path.join(args.out, f"mode_{name}_{k}.npz"), x=x, y=y, triangles=tri, z=z, holes=np.array(holes),
                 SL=p.eigenvalue.real * L)
    results[name] = dict(holes=[list(map(float, h)) for h in holes], SL_c=modes[0]["SL"] if modes else None,
                         modes=modes, drag_per_v0=[float(d) for d in drags], total_drag_per_v0=float(sum(drags)),
                         n_PI=len(holes))
    print(name, json.dumps(results[name]), f"[{time.time() - t0:.0f} s]", flush=True)
    json.dump(results, open(os.path.join(args.out, "multi.json"), "w"), indent=1)
print("... done.", flush=True)
