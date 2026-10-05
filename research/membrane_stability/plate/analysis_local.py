'''
Local (WKB) analysis of the flow-driven instability: convective vs absolute (Briggs-Bers / Huerre-Monkewitz).

Frozen-coefficient dispersion relation of the plate operator  zeta dz/dt = -kappa Delta^2 z + div(T grad z)
- (div T) . grad z  for a wave exp(i k x) along the flow, with local compression s = -T_xx > 0 and g = (div T)_x:
    zeta lambda = -kappa k^4 + s k^2 - i g k.
Scaling k = K sqrt(s/kappa), lambda = Lambda s^2/(kappa zeta):  Lambda = -K^4 + K^2 - i G K,  G = g sqrt(kappa) / s^1.5.
The term -i g k is a drift of the shape at speed g / zeta (towards larger tension). Temporal growth max_K Re Lambda =
1/4 for any G (convectively unstable wherever s > 0); the absolute growth rate (impulse response at a fixed point),
    sigma_abs(G) = lim_t log |int exp(Lambda(K) t) dK| / t,
is computed by high-precision quadrature (mpmath) and checked against the saddle points dLambda/dK = 0. The flow is
locally absolutely unstable where sigma_abs(G) > 0, i.e. G < G*.

Then the local fields of the box with in-plane friction (ell_b = 5, at the flutter threshold v0 = 0.2, and ell_b = 20
at its divergence threshold) are classified: compressed (s > 0), absolutely unstable (G < G*) or convectively
unstable only. Huerre-Monkewitz: a global oscillating mode (flutter) needs a region of local absolute instability of
sufficient size; without it the instability of the finite box is due to the boundaries.

    python3 analysis_local.py [out]
'''
import json
import os
import sys

import mpmath as mp
import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/runs/local"
os.makedirs(out, exist_ok=True)
mp.mp.dps = 60


def Lam(K, G):
    return -K ** 4 + K ** 2 - 1j * G * K


def impulse(G, t):
    f = lambda k: mp.exp(t * (-k ** 4 + k ** 2 - 1j * G * k))  # noqa: E731
    pts = list(np.linspace(-3, 3, 241))
    return mp.quad(f, [mp.mpf(p) for p in pts])


def sigma_abs_numeric(G, ts=tuple(np.linspace(40, 200, 9))):
    '''least-squares growth rate of the impulse response at x = 0 (the beating of two saddles averages out)'''
    vals = np.array([float(mp.log(abs(impulse(G, t)))) + 0.5 * np.log(t) for t in ts])
    return float(np.polyfit(ts, vals, 1)[0])


def saddles(G):
    roots = np.roots([-4.0, 0.0, 2.0, -1j * G])
    return [(complex(r), complex(Lam(r, G))) for r in roots]


def sigma_abs(G):
    '''absolute growth rate from the saddle points (dominant pair, connected to K = +-1/sqrt(2) at G = 0),
    checked against the impulse response'''
    # for G > 0 the real axis is deformed into the lower half plane (exp(-i G K t) decays there): only the saddles
    # with Im K <= 0 are crossed (the third one, on the positive imaginary axis, has Re Lambda > 0 but is not)
    return max(s[1].real for s in saddles(G) if s[0].imag <= 1e-12)


Gs = np.linspace(0.0, 1.6, 17)
rows = []
for G in Gs:
    sa = sigma_abs(G)
    num = sigma_abs_numeric(G)
    rows.append(dict(G=float(G), sigma_abs_saddle=sa, sigma_abs_numeric=num))
    print(f"G = {G:.3f}: sigma_abs saddle = {sa:+.5f}, impulse response = {num:+.5f}", flush=True)
lo, hi = 0.0, 5.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if sigma_abs(mid) > 0:
        lo = mid
    else:
        hi = mid
G_star = 0.5 * (lo + hi)
print(f"G* = {G_star:.5f}", flush=True)
res = dict(G_star=G_star, table=rows)

# local classification of the box with friction (needs dolfin)
try:
    import plate_lib as pl
    from dolfin import FunctionSpace, project
    L = 100.0
    geo = pl.Geometry(pl.make_mesh(os.path.join(out, "mesh"), L, L, [(L / 2, L / 2)], h_min=0.15, h_max=3.0))
    P1 = FunctionSpace(geo.mesh, "P", 1)
    xy = P1.tabulate_dof_coordinates()
    cases = []
    for ell, v0 in ((5.0, 0.1999), (20.0, 0.2583), (3.0, 0.155), (10.0, 0.244)):
        base = pl.BaseFlow(geo, b=1.0 / ell ** 2, V_unit=1.0)
        s0 = 25 / L ** 2
        Txx = s0 + v0 * project(base.T1[0, 0], P1).vector().get_local()
        gx = -v0 * project(base.f1[0], P1).vector().get_local()      # (div T)_x = -f_x
        s = -Txx
        comp = s > 0
        G = np.where(comp, np.abs(gx) / np.maximum(s, 1e-30) ** 1.5, np.inf)
        absolute = comp & (G < G_star)
        cell_area = np.full(len(s), 1.0)
        # areas from a P1 lumped mass
        from dolfin import TestFunction, assemble
        mass = assemble(TestFunction(P1) * geo.dx).get_local()
        row = dict(ell_b=ell, v0=v0, compressed_area=float(mass[comp].sum()), absolute_area=float(mass[absolute].sum()),
                   s_max=float(s.max()), G_min=float(G.min()), drift_upstream=bool(np.median(gx[comp]) < 0))
        if absolute.any():
            row["absolute_extent_x"] = [float(xy[absolute, 0].min() - L / 2), float(xy[absolute, 0].max() - L / 2)]
        cases.append(row)
        np.savez(os.path.join(out, f"local_ell{ell:g}.npz"), x=xy[:, 0] - L / 2, y=xy[:, 1] - L / 2, s=s, G=G, gx=gx,
                 triangles=pl.p1_triangulation(geo)[2])
        print(json.dumps(row), flush=True)
    res["box_cases"] = cases
except Exception as e:  # dolfin not available
    print("box classification skipped:", e)
json.dump(res, open(os.path.join(out, "local.json"), "w"), indent=1)
