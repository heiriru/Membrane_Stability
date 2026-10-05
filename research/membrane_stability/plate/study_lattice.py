'''
Square lattice of PIs (one PI per periodic cell of side Lc, area fraction phi = pi r0^2 / Lc^2): flow-driven buckling
threshold as a function of the PI density, and Bloch-Floquet band structure (pattern selection).

The membrane is driven past the (anchored) PIs by a tangential force: --drive force (uniform force density f0, the
shear stress of an outer flow) or --drive friction (friction b (V - v) with an outer fluid moving at V, ell_b =
sqrt(eta / b)). The drive amplitude D is f0 or V; the mean membrane velocity U (relative to the PIs) and the drag per
PI are reported at the threshold. Perturbations: Bloch waves z = exp(i q.x) p(x), q in the Brillouin zone
[-pi/Lc, pi/Lc]^2; by the symmetries (y -> -y of the base flow, and lambda(-q) = conj lambda(q)) it suffices to scan
q_x, q_y in [0, pi/Lc]. The threshold is the smallest D at which max_q Re lambda(q, D) = 0 (bisection), the selected
pattern is q* (q = 0: all PIs buckle in phase; q_x = pi/Lc: alternating along the flow; q_y = pi/Lc: alternating
across), and Im lambda(q*) != 0 means a travelling / oscillating pattern.

    python3 study_lattice.py [out] --cells 6,10,20,40 --drive force --sigma0 0.0025
'''
import argparse
import json
import os
import time

import numpy as np

import plate_lib as pl

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--cells", default="6,8,10,14,20,28,40,56,80")
parser.add_argument("--drive", choices=["force", "friction"], default="force")
parser.add_argument("--ell_b", type=float, default=5.0)
parser.add_argument("--sigma0", type=float, default=0.0025)
parser.add_argument("--pi_bc", default="rigid")
parser.add_argument("--nq", type=int, default=5, help="q grid per direction in [0, pi/Lc]")
parser.add_argument("--rtol", type=float, default=3e-3)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
t0 = time.time()
results = []
prev = os.path.join(args.out, "lattice.json")
if os.path.exists(prev):                       # resumable: keep the cells already computed
    results = json.load(open(prev))
done_cells = {r["Lc"] for r in results}
for Lc in [float(c) for c in args.cells.split(",")]:
    if Lc in done_cells:
        continue
    mdir = pl.make_mesh(os.path.join(args.out, f"mesh_L{Lc:g}"), Lc, Lc, [(Lc / 2, Lc / 2)], h_min=0.1,
                        h_max=min(1.5, Lc / 8), grading=min(10.0, Lc / 2), periodic=True)
    geo = pl.Geometry(mdir)
    if args.drive == "force":
        base = pl.BaseFlow(geo, f0_unit=1.0)
    else:
        base = pl.BaseFlow(geo, b=1.0 / args.ell_b ** 2, V_unit=1.0)
    plate = pl.Plate(base, pi_bc=args.pi_bc, bloch=True)
    qs = np.linspace(0, np.pi / Lc, args.nq)
    cache = {}

    def lead(D, qx, qy):
        key = (round(D, 12), qx, qy)
        if key not in cache:
            plate.set_q(qx, qy)
            cache[key] = plate.leading(D, args.sigma0, nev=6)[0].eigenvalue
        return cache[key]

    def gmax(D, grid=False):
        pts = [(qx, qy) for qx in qs for qy in qs] if grid else \
            [(0, 0), (qs[-1], 0), (0, qs[-1]), (qs[-1], qs[-1]), (qs[len(qs) // 2], 0), (0, qs[len(qs) // 2])]
        vals = {p: lead(D, *p) for p in pts}
        best = max(vals, key=lambda p: vals[p].real)
        return vals[best].real, best, vals

    # bracket: start from a drive estimate (the uniform-compression scale) and expand
    D_lo, D_hi = 0.0, 1e-3
    while gmax(D_hi)[0] < 0:
        D_lo, D_hi = D_hi, D_hi * 2
        if D_hi > 1e4:
            break
    # refine with the full grid at the bracket ends
    while gmax(D_hi, grid=True)[0] < 0:
        D_lo, D_hi = D_hi, D_hi * 1.5
    while D_hi - D_lo > args.rtol * D_hi:
        mid = 0.5 * (D_lo + D_hi)
        if gmax(mid, grid=True)[0] > 0:
            D_hi = mid
        else:
            D_lo = mid
    # refine q* off the grid (pattern search in the zone at D_hi), then re-bisect D at that q
    _, q_star, band = gmax(D_hi, grid=True)
    step = qs[1] - qs[0]
    best = lead(D_hi, *q_star).real
    while step > (qs[1] - qs[0]) / 16:
        moved = False
        for dq in ((step, 0), (-step, 0), (0, step), (0, -step)):
            cand = (float(np.clip(q_star[0] + dq[0], 0, qs[-1])), float(np.clip(q_star[1] + dq[1], 0, qs[-1])))
            val = lead(D_hi, *cand).real
            if val > best:
                best, q_star, moved = val, cand, True
        if not moved:
            step *= 0.5
    while D_hi - D_lo > 0.3 * args.rtol * D_hi:
        mid = 0.5 * (D_lo + D_hi)
        if lead(mid, *q_star).real > 0 or gmax(mid)[0] > 0:
            D_hi = mid
        else:
            D_lo = mid
    Dc = 0.5 * (D_lo + D_hi)
    lam_star = lead(D_hi, *q_star)
    band = {p: lead(D_hi, *p) for p in [(qx, qy) for qx in qs for qy in qs]}
    band_grid = np.array([[band[(qx, qy)].real for qy in qs] for qx in qs])
    band_imag = np.array([[band[(qx, qy)].imag for qy in qs] for qx in qs])
    # mode at q*
    plate.set_q(*q_star)
    pair = plate.leading(D_hi, args.sigma0, nev=6)[0]
    # Bloch amplitude p = p_r + i p_i of the eigenvector (real and imaginary parts of the eigenvector of the real
    # formulation, xr / xi, for complex eigenvalues)
    zr, _ = plate.z_field(pair.x_real, 0)
    zi, _ = plate.z_field(pair.x_real, 1)
    zr2, _ = plate.z_field(pair.x_imag, 0)
    zi2, _ = plate.z_field(pair.x_imag, 1)
    x, y, tri = pl.p1_triangulation(geo)
    np.savez(os.path.join(args.out, f"mode_L{Lc:g}.npz"), x=x, y=y, triangles=tri, p_r=zr, p_i=zi, p_r_xi=zr2,
             p_i_xi=zi2, lam=np.array([pair.eigenvalue.real, pair.eigenvalue.imag]), q=np.array(q_star),
             qs=qs, band=band_grid, band_imag=band_imag, D=D_hi)
    row = dict(Lc=Lc, phi=geo.phi, drive=args.drive, ell_b=args.ell_b if args.drive == "friction" else None,
               sigma0=args.sigma0, D_c=Dc, U_c=base.U * Dc, drag_c=base.F_drag * Dc, U_per_D=base.U,
               drag_per_D=base.F_drag, q_star=[float(q_star[0]), float(q_star[1])],
               q_star_over_zone=[float(q_star[0] * Lc / np.pi), float(q_star[1] * Lc / np.pi)],
               lambda_star=[lam_star.real, lam_star.imag], travelling=bool(abs(lam_star.imag) > 1e-6 * abs(lam_star)
                                                                           + 1e-14),
               SL_cell=Lc * base.U * Dc, n_eig=len(cache), wall=time.time() - t0)
    results.append(row)
    print(json.dumps(row), flush=True)
    json.dump(results, open(os.path.join(args.out, "lattice.json"), "w"), indent=1)
print("... done.", flush=True)
