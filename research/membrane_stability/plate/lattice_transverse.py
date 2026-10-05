'''
Long-wave Bloch coefficients of a lattice of anchored PIs in every direction (2D slope equation).

For a long-wave Bloch mode q = q (cos th, sin th) of the lattice (study_lattice.py, lattice_longwave.py: the 1D,
th = 0 case), the leading eigenvalue is
    zeta lambda(q, th) = -sigma(th) q^2 - K(th) q^4 - i zeta c(th) q + ...,
and for the 2D slope equation
    zeta H_t = div J(grad H) - K_ab..d d_a d_b d_c d_d H - zeta c H_x
the linear part requires sigma(th) = sigma_xx cos^2 + sigma_yy sin^2 and K(th) = K_xx cos^4 + K_m cos^2 sin^2
+ K_yy sin^4 (the lattice is symmetric under y -> -y; the flow along x breaks x -> -x and the x <-> y symmetry).
Re lambda is fitted with three small q per direction (q^2, q^4, q^6), at the long-wave threshold drive D_lw and at
rest.

    python3 lattice_transverse.py [lattice run directory] --cells 6,8,10,14,20
'''
import argparse
import json
import os

import numpy as np

import plate_lib as pl

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--cells", default="6,8,10,14,20")
parser.add_argument("--angles", default="0,30,45,60,90")
args = parser.parse_args()
lw = {r["Lc"]: r for r in json.load(open(os.path.join(args.run, "longwave.json")))}
out = []
for Lc in [float(c) for c in args.cells.split(",")]:
    r = lw[Lc]
    geo = pl.Geometry(os.path.join(args.run, f"mesh_L{Lc:g}"))
    base = pl.BaseFlow(geo, f0_unit=1.0) if r["drive"] == "force" else pl.BaseFlow(geo, b=1.0 / r["ell_b"] ** 2,
                                                                                       V_unit=1.0)
    plate = pl.Plate(base, pi_bc="rigid", bloch=True)
    row = dict(Lc=Lc, phi=r["phi"], D_lw=r["D_lw"], drives={})
    for dname, D in (("rest", 0.0), ("lw", r["D_lw"])):
        dirs = []
        for th in [float(a) for a in args.angles.split(",")]:
            fqs = np.array([0.03, 0.1, 0.15])
            lam = []
            for fq in fqs:
                q = fq * np.pi / Lc
                plate.set_q(q * np.cos(np.radians(th)), q * np.sin(np.radians(th)))
                lam.append(plate.leading(D, r["sigma0"], nev=6)[0].eigenvalue)
            lam = np.array(lam)
            qs = fqs * np.pi / Lc
            A = np.vstack([-qs ** 2, -qs ** 4, -qs ** 6]).T
            s, K, K6 = np.linalg.solve(A, lam.real)
            dirs.append(dict(angle=th, sigma=float(s), K=float(K), K6=float(K6), drift=float(-lam[0].imag / qs[0]),
                             lam=[[float(z.real), float(z.imag)] for z in lam]))
            print(f"Lc = {Lc:g}, {dname}: th = {th:4.0f}: sigma = {s:+.5e}, K = {K:+.4e}, drift = {dirs[-1]['drift']:+.4e}",
                  flush=True)
        th = np.radians([d["angle"] for d in dirs])
        c2, s2 = np.cos(th) ** 2, np.sin(th) ** 2
        sx, sy = np.linalg.lstsq(np.vstack([c2, s2]).T, [d["sigma"] for d in dirs], rcond=None)[0]
        Kx, Km, Ky = np.linalg.lstsq(np.vstack([c2 ** 2, c2 * s2, s2 ** 2]).T, [d["K"] for d in dirs], rcond=None)[0]
        row["drives"][dname] = dict(D=D, directions=dirs, sigma_xx=float(sx), sigma_yy=float(sy), K_xx=float(Kx),
                                    K_mixed=float(Km), K_yy=float(Ky))
        print(f"  {dname}: sigma_xx = {sx:+.5e}, sigma_yy = {sy:+.5e}; K_xx = {Kx:.4f}, K_mixed = {Km:.4f}, "
              f"K_yy = {Ky:.4f}", flush=True)
    out.append(row)
    json.dump(out, open(os.path.join(args.run, "transverse.json"), "w"), indent=1)
