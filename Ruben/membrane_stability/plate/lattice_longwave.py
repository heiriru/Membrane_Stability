'''
Long-wave limit of the lattice instability (post-processing of study_lattice.py runs).

For a lattice of rigid PIs, z = const is neutral at q = 0; for small q along the flow the leading Bloch eigenvalue is
    zeta lambda(q) = -sigma_eff q^2 - i zeta c q + O(q^3),
with c ~ f0 / zeta the drift of the follower load and sigma_eff the effective long-wave tension of the lattice (base
tension plus the bending stiffness of the cells between the rigid PIs, minus the flow-induced part). The lattice
buckles collectively when sigma_eff(D) = 0: a long-wave, travelling undulation of the whole lattice. For each cell:
sigma_eff at a few drives, the drive D_lw at which it vanishes (bisection), and the drift speed there.

    python3 lattice_longwave.py [lattice run directory] [--extra_cells 40,56,80]
'''
import json
import os
import sys

import numpy as np

import plate_lib as pl

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--extra_cells", default="", help="cells not in the run (long-wave threshold only)")
args = parser.parse_args()
run = args.run
info = json.load(open(os.path.join(run, "lattice.json")))
proto = dict(info[0])
for Lc in [float(c) for c in args.extra_cells.split(",") if c]:
    if Lc not in {r["Lc"] for r in info}:
        info.append(dict(proto, Lc=Lc, D_c=None, phi=np.pi / Lc ** 2))
out = []
for r in info:
    Lc = r["Lc"]
    mdir = os.path.join(run, f"mesh_L{Lc:g}")
    if not os.path.exists(os.path.join(mdir, "geometry.npz")):
        pl.make_mesh(mdir, Lc, Lc, [(Lc / 2, Lc / 2)], h_min=0.1, h_max=min(1.5, Lc / 8), grading=min(10.0, Lc / 2),
                     periodic=True)
    geo = pl.Geometry(mdir)
    if r["drive"] == "force":
        base = pl.BaseFlow(geo, f0_unit=1.0)
    else:
        base = pl.BaseFlow(geo, b=1.0 / r["ell_b"] ** 2, V_unit=1.0)
    plate = pl.Plate(base, pi_bc="rigid", bloch=True)
    q = 0.01 * np.pi / Lc
    plate.set_q(q, 0.0)

    def s_eff(D):
        lam = plate.leading(D, r["sigma0"], nev=6)[0].eigenvalue
        return -lam.real / q ** 2, lam.imag / q

    s0, _ = s_eff(0.0)
    lo, hi = 0.0, (r["D_c"] or 1e-3) * 1.2
    while s_eff(hi)[0] > 0:
        lo, hi = hi, hi * 1.5
    for _ in range(25):
        mid = 0.5 * (lo + hi)
        if s_eff(mid)[0] > 0:
            lo = mid
        else:
            hi = mid
    D_lw = 0.5 * (lo + hi)
    _, drift = s_eff(D_lw)
    # q^4 coefficient at D_lw (sigma_eff = 0 there): zeta lambda = -K_eff q^4 + ...  -> effective bending rigidity of
    # the lattice for long waves (amplitude equation: zeta dH/dt = -sigma_eff H_xx - K_eff H_xxxx - zeta c H_x + ...)
    K_eff = []
    for fq in (0.1, 0.15):
        qq = fq * np.pi / Lc
        plate.set_q(qq, 0.0)
        K_eff.append(-plate.leading(D_lw, r["sigma0"], nev=6)[0].eigenvalue.real / qq ** 4)
    # is a finite-q mode unstable before the long-wave one? (coarse check at D_lw)
    finite = []
    for fx, fy in ((0.25, 0), (0.5, 0), (1, 0), (0, 0.5), (0, 1), (0.5, 0.5), (1, 1)):
        plate.set_q(fx * np.pi / Lc, fy * np.pi / Lc)
        finite.append(plate.leading(D_lw, r["sigma0"], nev=6)[0].eigenvalue.real)
    plate.set_q(q, 0.0)
    row = dict(Lc=Lc, phi=r["phi"], drive=r["drive"], sigma0=r["sigma0"], sigma_eff_at_rest=s0, D_lw=D_lw,
               U_lw=base.U * D_lw, drag_lw=base.F_drag * D_lw, D_c_grid=r["D_c"], drift_speed=-drift,
               U_c_grid=r.get("U_c") if r["D_c"] else None,
               drift_over_D=-drift / D_lw, max_finite_q_growth_at_D_lw=float(max(finite)), K_eff=K_eff)
    out.append(row)
    print(json.dumps(row), flush=True)
    json.dump(out, open(os.path.join(run, "longwave.json"), "w"), indent=1)
