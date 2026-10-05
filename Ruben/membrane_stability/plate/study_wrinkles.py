'''
From a single dome to wrinkle patterns.

(a) Membrane sliding over a frictional support (no PI): rectangle [0, Lx] x [0, Ly], inflow U on the left, friction
    b v with a support at rest (V = 0). The base state is exact: v = U e_x, sigma(x) = sigma0 - b U (Lx - x) (the
    tension falls upstream and becomes compressive for x < Lx - sigma0 / (b U)). Local theory: in the compressed
    region the fastest growing wavelength is 2 pi sqrt(2 kappa / |sigma(x)|) and the shape drifts at g / zeta = -b U
    (upstream). The temporal eigenmodes at drives above the threshold show whether wrinkle trains (crests across or
    along the flow) form.
(b) The PRE box with a PI (no friction, Gamma = 25) at larger L = 200 and strong drive: leading modes at
    v0 = 1, 2, 4, 8 v0_c (count of unstable modes, their structure: domes vs wrinkle trains).

    python3 study_wrinkles.py [out]
'''
import json
import os
import sys
import time

import numpy as np

import plate_lib as pl

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/runs/wrinkles"
os.makedirs(out, exist_ok=True)
t0 = time.time()
res = {}

# (a) sliding membrane
Lx, Ly, ell_b, sigma0 = 200.0, 100.0, 10.0, 0.0025
b = 1.0 / ell_b ** 2
geo = pl.Geometry(pl.make_mesh(os.path.join(out, "mesh_slide"), Lx, Ly, [], h_max=1.0))
base = pl.BaseFlow(geo, b=b, V_unit=0.0)
plate = pl.Plate(base, pi_bc="rigid")
# threshold (temporal: the operator is not self-adjoint, drift)
lead = lambda U: plate.leading(U, sigma0, nev=8)[0].eigenvalue  # noqa: E731
lo, hi = 0.0, 0.01
while lead(hi).real < 0:
    lo, hi = hi, hi * 1.5
while hi - lo > 2e-3 * hi:
    mid = 0.5 * (lo + hi)
    lo, hi = (lo, mid) if lead(mid).real > 0 else (mid, hi)
U_c = 0.5 * (lo + hi)
print(f"(a) sliding membrane: U_c = {U_c:.5f} (compression at the inflow b U_c Lx - sigma0 = {b * U_c * Lx - sigma0:.4f})"
      f"  [{time.time() - t0:.0f} s]", flush=True)
x, y, tri = pl.p1_triangulation(geo)
slide = dict(U_c=U_c, b=b, Lx=Lx, Ly=Ly, sigma0=sigma0, cases=[])
for fac in (1.0, 1.5, 2.5, 4.0):
    U = fac * U_c
    s_in = b * U * Lx - sigma0
    pairs = plate.rates(U, sigma0, nev=12, target=0.0 if fac == 1.0 else 0.5 * (s_in ** 2 / 4))
    pairs = sorted(pairs, key=lambda p: -p.eigenvalue.real)
    modes = []
    for k, p in enumerate(pairs[:6]):
        zr, _ = plate.z_field(p.x_real)
        zi, _ = plate.z_field(p.x_imag)
        # dominant wavenumbers (FFT of the mode on a grid)
        import matplotlib.tri as mtri
        X, Y = np.meshgrid(np.linspace(0, Lx, 400), np.linspace(0, Ly, 200))
        Z = mtri.LinearTriInterpolator(mtri.Triangulation(x, y, tri), zr)(X, Y).filled(0.0)
        Fz = np.abs(np.fft.rfft2(Z))
        Fz[0, 0] = 0.0
        ky, kx = np.unravel_index(np.argmax(Fz), Fz.shape)
        kxv = 2 * np.pi * kx / Lx
        kyv = 2 * np.pi * min(ky, Z.shape[0] - ky) / Ly
        centroid_x = float(np.sum(x * zr ** 2) / np.sum(zr ** 2))
        modes.append(dict(lam=[p.eigenvalue.real, p.eigenvalue.imag], kx=kxv, ky=kyv, centroid_x=centroid_x))
        np.savez(os.path.join(out, f"slide_f{fac:g}_m{k}.npz"), x=x, y=y, triangles=tri, z_r=zr, z_i=zi,
                 lam=np.array([p.eigenvalue.real, p.eigenvalue.imag]))
    # local prediction at the inflow (most compressed): k* = sqrt(s / (2 kappa)), drift speed b U / zeta
    slide["cases"].append(dict(factor=fac, U=U, compression_inflow=s_in, k_star_inflow=float(np.sqrt(s_in / 2)),
                               wavelength_inflow=float(2 * np.pi / np.sqrt(s_in / 2)), drift=b * U,
                               n_unstable=int(sum(p.eigenvalue.real > 0 for p in pairs)), modes=modes))
    print(json.dumps(slide["cases"][-1]), f"[{time.time() - t0:.0f} s]", flush=True)
res["sliding"] = slide
json.dump(res, open(os.path.join(out, "wrinkles.json"), "w"), indent=1)

# (b) PRE box with a PI, L = 200, Gamma = 25 (sigma0 = 25 / L^2), strong drive
L = 200.0
s0 = 25 / L ** 2
geo = pl.Geometry(pl.make_mesh(os.path.join(out, "mesh_box200"), L, L, [(L / 2, L / 2)], h_min=0.15, h_max=3.0,
                               grading=50.0))
base = pl.BaseFlow(geo)
plate = pl.Plate(base, pi_bc="rigid")
c = plate.drive_threshold(s0)
v0c = c.eigenvalue.real
x, y, tri = pl.p1_triangulation(geo)
box = dict(L=L, v0_c=v0c, SL_c=v0c * L, cases=[])
print(f"(b) box L = 200: SL_c = {v0c * L:.3f}", flush=True)
for fac in (1.0, 2.0, 4.0, 8.0):
    # temporal spectrum at v0 = fac v0_c (self-adjoint: real); the unstable ones are the largest
    rates = sorted(plate.rates(fac * v0c, s0, nev=16, target=0.0 if fac == 1.0 else 1e-2), key=lambda p: -p.eigenvalue.real)
    modes = []
    for k, p in enumerate(rates[:6]):
        z, _ = plate.z_field(p.x_real)
        z /= z[np.argmax(np.abs(z))]
        # number of sign changes along the centreline upstream: lobes
        sel = (np.abs(y - L / 2) < 1.0) & (x < L / 2)
        order = np.argsort(x[sel])
        zl = z[sel][order]
        lobes = int(np.sum(np.abs(np.diff(np.sign(zl[np.abs(zl) > 0.05]))) > 0)) + 1
        modes.append(dict(lam=p.eigenvalue.real, lobes_upstream=lobes,
                          centroid=[float(np.sum(x * z ** 2) / np.sum(z ** 2) - L / 2),
                                    float(np.sum(y * z ** 2) / np.sum(z ** 2) - L / 2)]))
        np.savez(os.path.join(out, f"box200_f{fac:g}_m{k}.npz"), x=x, y=y, triangles=tri, z=z, lam=p.eigenvalue.real)
    box["cases"].append(dict(factor=fac, n_unstable=int(sum(p.eigenvalue.real > 0 for p in rates)), modes=modes))
    print(json.dumps(box["cases"][-1]), f"[{time.time() - t0:.0f} s]", flush=True)
res["box200"] = box
json.dump(res, open(os.path.join(out, "wrinkles.json"), "w"), indent=1)
print("... done.", flush=True)
