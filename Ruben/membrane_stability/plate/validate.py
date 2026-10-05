'''
Checks of the standalone plate model (plate_lib.py):
 1. box L = 100 (PRE geometry), Gamma = 25: threshold vs IRENE's plate model / full FE (rigid force-free and clamped PI);
 2. periodic square array, uniform body force: drag vs Sangani & Acrivos (1982) for Stokes flow through a square array
    of cylinders, F / (eta U_s) = 4 pi / (-1/2 ln phi - 0.738 + phi - 0.887 phi^2 + 2.039 phi^3), U_s = superficial velocity
    (their flow is driven by a mean pressure gradient G, whose force on a cylinder includes the buoyancy-like part
    G pi r^2: F_SA = G L^2 = F_membrane / (1 - phi), with F_membrane = f0 x membrane area the drag of the body-force
    driven membrane on the PI);
 3. Bloch: q = 0 reproduces the periodic problem; lambda(q + 2 pi / L e_x) = lambda(q).
    python3 validate.py [out]
'''
import json
import os
import sys
import time

import numpy as np

import plate_lib as pl

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/runs/plate_validate"
os.makedirs(out, exist_ok=True)
res = {}
t0 = time.time()

# 1. box
m = pl.make_mesh(os.path.join(out, "box"), 100, 100, [(50, 50)], h_min=0.15, h_max=3.0)
geo = pl.Geometry(m)
base = pl.BaseFlow(geo)
for bcname in ("rigid", "clamped"):
    plate = pl.Plate(base, pi_bc=bcname)
    c = plate.drive_threshold(25 / 1e4)
    res[f"box_{bcname}_SLc"] = 100 * c.eigenvalue.real
    print(f"box, {bcname}: SL_c = {100 * c.eigenvalue.real:.4f}  (IRENE plate rigid mesh A 27.444, full FE 27.504; "
          f"clamped full FE 27.732)  [{time.time() - t0:.0f} s]", flush=True)
# temporal check at the threshold
plate = pl.Plate(base, pi_bc="rigid")
lam = plate.leading(res["box_rigid_SLc"] / 100, 25 / 1e4)[0].eigenvalue
res["box_rigid_lambda_at_threshold"] = [lam.real, lam.imag]
print(f"leading rate at the threshold: {lam}", flush=True)

# 2. periodic drag
rows = []
for Lc in (5.0, 10.0, 20.0, 40.0):
    mp = pl.make_mesh(os.path.join(out, f"cell{Lc:g}"), Lc, Lc, [(Lc / 2, Lc / 2)], h_min=0.1, h_max=min(1.0, Lc / 8),
                      periodic=True)
    g = pl.Geometry(mp)
    b = pl.BaseFlow(g, f0_unit=1.0)
    phi = g.phi
    Us = b.U * (1 - phi)
    F = b.F_drag / (1 - phi)
    sa = 4 * np.pi / (-0.5 * np.log(phi) - 0.738 + phi - 0.887 * phi ** 2 + 2.039 * phi ** 3)
    rows.append(dict(L=Lc, phi=phi, F_over_U=F / Us, sangani_acrivos=sa, rel=abs(F / Us - sa) / sa))
    print(f"cell L = {Lc}: phi = {phi:.4f}, F/(eta U_s) = {F / Us:.5f}, Sangani-Acrivos {sa:.5f}, "
          f"rel. diff {rows[-1]['rel']:.1e}  [{time.time() - t0:.0f} s]", flush=True)
res["periodic_drag"] = rows

# 3. Bloch checks on the L = 20 cell, uniform force drive (non-self-adjoint)
g = pl.Geometry(os.path.join(out, "cell20"))
b = pl.BaseFlow(g, f0_unit=1.0)
per = pl.Plate(b)
blo = pl.Plate(b, bloch=True)
D, s0 = 0.05, 1e-3
l_per = [p.eigenvalue for p in per.rates(D, s0, nev=4)][:2]
blo.set_q(0, 0)
l_q0 = [p.eigenvalue for p in blo.rates(D, s0, nev=6)][:4]
q = 0.4
blo.set_q(q, 0.1)
l_a = [p.eigenvalue for p in blo.rates(D, s0, nev=6)][:2]
blo.set_q(q + 2 * np.pi / 20, 0.1)
l_b = [p.eigenvalue for p in blo.rates(D, s0, nev=6)][:2]
print("periodic:", l_per, "\nBloch q = 0 (pairs):", l_q0, "\nBloch q:", l_a, "\nBloch q + G:", l_b, flush=True)
res["bloch"] = dict(periodic=[[l.real, l.imag] for l in l_per], q0=[[l.real, l.imag] for l in l_q0],
                    q=[[l.real, l.imag] for l in l_a], q_plus_G=[[l.real, l.imag] for l in l_b])
json.dump(res, open(os.path.join(out, "validate.json"), "w"), indent=1)
print("VALIDATE:", json.dumps(res), flush=True)
