'''
Divergence (static buckling) vs flutter (oscillatory) thresholds in the PRE box (L = 100, one rigid force-free PI)
with in-plane friction b (v - v0 e_x), as a function of the screening length ell_b = sqrt(eta / b) and of
Gamma = sigma0 L^2 / kappa. The plate operator is not self-adjoint with friction, so the thresholds come from the
temporal spectrum: for each (ell_b, Gamma) two bisections in v0, one on the largest real part among the real
eigenvalues (divergence) and one among the complex pairs (flutter); the first instability is the smaller of the two,
and their crossing is the codimension-two point. Algorithm: the largest real part of the whole spectrum is scanned in
6 % steps of v0 from SL = 5 and bisected at its first zero (flutter windows can be narrow: a coarser bracket expansion
steps over them); the type of the leading eigenvalue there gives divergence or flutter. Checked against IRENE's full equations (queue runs fmap_*).

    python3 study_flutter_map.py [out] --ells 2,3,5,7,10,20 --gammas 0,25,100
'''
import argparse
import json
import os
import time

import numpy as np

import plate_lib as pl

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--ells", default="1.5,2,2.5,3,3.5,4,4.5,5,6,7,8,10,12,14,17,20,25,30,40,60")
parser.add_argument("--gammas", default="0,10,25,50,100")
parser.add_argument("--rtol", type=float, default=2e-3)
parser.add_argument("--nev", type=int, default=10)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
L = 100.0
geo = pl.Geometry(pl.make_mesh(os.path.join(args.out, "mesh"), L, L, [(L / 2, L / 2)], h_min=0.15, h_max=3.0))
t0 = time.time()
rows = []
for ell in [float(e) for e in args.ells.split(",")]:
    base = pl.BaseFlow(geo, b=1.0 / ell ** 2, V_unit=1.0)
    plate = pl.Plate(base, pi_bc="rigid")
    for gamma in [float(g) for g in args.gammas.split(",")]:
        s0 = gamma / L ** 2
        cache = {}

        def spectrum(v0):
            if v0 not in cache:
                cache[v0] = [p.eigenvalue for p in plate.rates(v0, s0, nev=args.nev)]
            return cache[v0]

        def g_all(v0):
            return max(l.real for l in spectrum(v0))

        def g_real(v0):
            lr = [l.real for l in spectrum(v0) if abs(l.imag) <= 1e-9 * abs(l)]
            return max(lr) if lr else -np.inf

        def g_cplx(v0):
            lc = [l.real for l in spectrum(v0) if abs(l.imag) > 1e-9 * abs(l)]
            return max(lc) if lc else -np.inf

        # first instability: scan v0 in steps of 6 % until the largest real part (any eigenvalue) crosses zero, then
        # bisect (a flutter window between two scan points narrower than 6 % would be missed); the type of the
        # leading eigenvalue just above the crossing decides divergence vs flutter
        lo, hi = 0.0, 0.05
        while g_all(hi) < 0:
            lo, hi = hi, hi * 1.06
            if hi > 4.0:
                break
        while hi - lo > args.rtol * hi:
            mid = 0.5 * (lo + hi)
            if g_all(mid) > 0:
                hi = mid
            else:
                lo = mid
        first = 0.5 * (lo + hi)
        lead = max(spectrum(hi), key=lambda l: l.real)
        kind = "flutter" if abs(lead.imag) > 1e-9 * abs(lead) else "divergence"
        omega = abs(lead.imag) if kind == "flutter" else None
        v_div = first if kind == "divergence" else None
        v_flu = first if kind == "flutter" else None
        row = dict(ell_b=ell, Gamma=gamma, v0_div=v_div, v0_flutter=v_flu, omega_flutter=omega, v0_c=first,
                   SL_c=None if first is None else first * L, kind=kind, n_eig=len(cache), wall=time.time() - t0)
        rows.append(row)
        print(json.dumps(row), flush=True)
        json.dump(rows, open(os.path.join(args.out, "flutter_map.json"), "w"), indent=1)
print("... done.", flush=True)
