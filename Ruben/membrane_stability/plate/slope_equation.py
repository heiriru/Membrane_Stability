'''
Long-wave (slope) equation of a lattice of anchored PIs with the coefficients computed from the full model.

For the mean height H(x, t) of the lattice along the flow (Bloch long-wave expansion, lattice_longwave.py, and the
homogenized nonlinear cell, nonlinear_cell.py):
    zeta H_t = d_x J(H_x) - K H_xxxx - zeta c H_x,     J(u) = sigma u + g3 u^3 + g5 u^5,
with zeta = 1, sigma = -eps sigma_rest (eps = D / D_lw - 1, the distance from the long-wave threshold; sigma_rest is the
Bloch effective tension at rest, so sigma = sigma_rest (1 - D / D_lw) in the linear approximation), K = K_eff and c the
drift speed of the long-wave mode at threshold (negative: the pattern travels upstream), and g3, g5 the cubic and
quintic coefficients of the cell problem at the threshold drive (rescaled to the Bloch normalization by
sigma_rest(Bloch) / sigma_rest(cell)).

Studies (all for zeta = 1, lengths in r0, time in zeta r0^4 / kappa):
  1. uniform-slope states: J(u) = 0 branches (supercritical: u_s ~ sqrt(eps); subcritical: a fold, hysteresis);
  2. periodic domain (the drift is then a Galilean shift and drops out): slope domains (terraces) and their
     coarsening, for the dense (Lc = 6) and the dilute (Lc = 8) lattice, plus a hysteresis run;
  3. finite lattice patch (H = H_x = 0 at both ends): the drift makes the instability convective. Absolute threshold
     eps_a (saddle point of the dispersion relation) and global threshold eps_g(L_p) of a patch of length L_p (linear
     eigenproblem); nonlinear run in a finite patch above eps_g.

    python3 slope_equation.py [out_dir]
'''
import json
import os
import sys
import time

import numpy as np
import scipy.linalg as sla
from scipy.optimize import brentq, fsolve

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "round5", "slope")
os.makedirs(out, exist_ok=True)
t0 = time.time()


def coefficients():
    lw = {r["Lc"]: r for r in json.load(open(os.path.join(ROOT, "results", "round3", "lat_force", "longwave.json")))}
    co = {}
    for Lc in (6.0, 8.0, 10.0, 14.0, 20.0):
        p = os.path.join(ROOT, "results", "round4", "lattice_nonlinear", f"nonlinear_cell_L{Lc:g}.json")
        d = json.load(open(p))
        rest = [r for r in d["runs"] if r["drive_name"] == "zero"][0]
        thr = [r for r in d["runs"] if r["drive_name"] == "lw"][0]
        b = lw[Lc]
        scale = b["sigma_eff_at_rest"] / rest["sigma_eff_fit"]
        us = np.array([r["u"] for r in thr["rows"]])
        Js = np.array([r["J"] for r in thr["rows"]])
        # nonlinear part of J at threshold: fit J = s u + g3 u^3 + g5 u^5 and keep g3, g5 (s, the static cell's small
        # offset from the dynamic threshold, is replaced by sigma(eps) below)
        A = np.vstack([us, us ** 3, us ** 5]).T
        s, g3, g5 = np.linalg.lstsq(A, Js, rcond=None)[0]
        co[Lc] = dict(Lc=Lc, phi=d["phi"], sigma_rest=b["sigma_eff_at_rest"], K=float(np.mean(b["K_eff"])),
                      c=b["drift_speed"], D_lw=b["D_lw"], g3=float(g3 * scale), g5=float(g5 * scale),
                      u_max_data=float(us.max()), scale=float(scale), s_static=float(s))
    return co


co = coefficients()
for Lc, c in co.items():
    print(f"Lc = {Lc:g}: phi = {c['phi']:.4f}, sigma_rest = {c['sigma_rest']:.4f}, K = {c['K']:.3f}, c = {c['c']:+.4f},"
          f" g3 = {c['g3']:+.4f}, g5 = {c['g5']:+.3f} (data up to u = {c['u_max_data']})", flush=True)


def J(u, c, eps):
    return -eps * c["sigma_rest"] * u + c["g3"] * u ** 3 + c["g5"] * u ** 5


# 1. uniform slopes: J(u)/u = 0  <=>  -eps sigma_rest + g3 u^2 + g5 u^4 = 0
def branches(c):
    res = []
    for eps in np.linspace(-0.3, 0.3, 121):
        a, b_, cc = c["g5"], c["g3"], -eps * c["sigma_rest"]
        roots = np.roots([a, b_, cc]) if a != 0 else np.array([-cc / b_])
        u2 = sorted(r.real for r in np.atleast_1d(roots) if abs(r.imag) < 1e-12 and r.real > 0)
        for x in u2:
            u = np.sqrt(x)
            stable = (-eps * c["sigma_rest"] + 3 * c["g3"] * u ** 2 + 5 * c["g5"] * u ** 4) > 0   # J'(u) > 0
            res.append(dict(eps=float(eps), u=float(u), stable=bool(stable)))
    return res


uniform = {Lc: branches(c) for Lc, c in co.items()}


# 2. periodic runs (pseudo-spectral in u = H_x, IMEX with a stabilizing linear term)
def run_periodic(c, eps_of_t, Lx=600.0, N=1024, t_end=4e5, dt=2.0, seed=1, n_out=200, u0=None):
    rng = np.random.default_rng(seed)
    x = np.arange(N) * Lx / N
    q = 2 * np.pi * np.fft.rfftfreq(N, Lx / N)
    u = 1e-3 * rng.standard_normal(N) if u0 is None else u0.copy()
    u -= u.mean()
    A_st = 2.0 * max(c["sigma_rest"], 0.05)
    nt = int(t_end / dt)
    every = max(1, nt // n_out)
    ts, Hs, us_out, ns = [], [], [], []
    for n in range(nt + 1):
        t = n * dt
        if n % every == 0:
            uh = np.fft.rfft(u)
            Hh = np.zeros_like(uh)
            Hh[1:] = uh[1:] / (1j * q[1:])
            H = np.fft.irfft(Hh, N)
            ts.append(t)
            Hs.append(H.astype(np.float32))
            us_out.append(u.astype(np.float32))
            s = np.sign(u)
            ns.append(int(np.count_nonzero(s != np.roll(s, 1))))
        eps = eps_of_t(t)
        Jh = np.fft.rfft(J(u, c, eps))
        uh = np.fft.rfft(u)
        rhs = uh + dt * (-q ** 2 * Jh + A_st * q ** 2 * uh)
        uh = rhs / (1 + dt * (c["K"] * q ** 4 + A_st * q ** 2))
        uh[0] = 0.0
        u = np.fft.irfft(uh, N)
        if not np.all(np.isfinite(u)) or np.abs(u).max() > 5:
            print("  blow-up at t =", t, flush=True)
            break
    return dict(x=x, t=np.array(ts), H=np.array(Hs), u=np.array(us_out), n_walls=np.array(ns))


runs = {}
for name, Lc, eps in (("dense_eps0.05", 6.0, 0.05), ("dense_eps0.2", 6.0, 0.2), ("dilute_eps0.05", 8.0, 0.05)):
    c = co[Lc]
    r = run_periodic(c, lambda t, e=eps: e)
    runs[name] = r
    np.savez_compressed(os.path.join(out, f"periodic_{name}.npz"), **r)
    print(f"periodic {name}: final max|u| = {np.abs(r['u'][-1]).max():.4f}, walls {r['n_walls'][[1, len(r['t']) // 4, -1]]}"
          f"  [{time.time() - t0:.0f} s]", flush=True)
# hysteresis (dilute, subcritical): ramp eps from +0.05 down to -0.15 and back, slowly
ramp = lambda t: 0.05 - 0.2 * min(t, 4e5) / 4e5 + 0.2 * max(t - 4e5, 0) / 4e5   # noqa: E731
r = run_periodic(co[8.0], ramp, t_end=8e5, u0=runs["dilute_eps0.05"]["u"][-1].astype(float))
r["eps"] = np.array([ramp(t) for t in r["t"]])
np.savez_compressed(os.path.join(out, "periodic_dilute_hysteresis.npz"), **r)
print(f"hysteresis run done  [{time.time() - t0:.0f} s]", flush=True)


# 3. finite patch: convective vs absolute
C_ABS = 1.622076   # absolute-instability drift of lambda = q^2 - q^4 - i C q (Briggs-Bers pinch, computed once)


def eps_absolute(c):
    '''scaling x = l x', t = tau t' with l = sqrt(K/a), tau = K/a^2 (a = -sigma > 0) maps the dispersion relation to
    lambda' = q^2 - q^4 - i C q with C = |c| sqrt(K) / a^(3/2): absolutely unstable for C < C_ABS'''
    a = (abs(c["c"]) * np.sqrt(c["K"]) / C_ABS) ** (2.0 / 3.0)
    return a / c["sigma_rest"]


def patch_operator(c, eps, Lp, n=None):
    '''finite differences for H on (0, Lp), H = H_x = 0 at both ends (ghost points)'''
    if n is None:
        n = int(Lp / 0.25) - 1
    h = Lp / (n + 1)
    sig = -eps * c["sigma_rest"]
    I = np.eye(n)
    D2 = (np.diag(-2 * np.ones(n)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)) / h ** 2
    D1 = (np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)) / (2 * h)
    D4 = (np.diag(6 * np.ones(n)) + np.diag(-4 * np.ones(n - 1), 1) + np.diag(-4 * np.ones(n - 1), -1)
          + np.diag(np.ones(n - 2), 2) + np.diag(np.ones(n - 2), -2)) / h ** 4
    D4[0, 0] += 1 / h ** 4          # clamped: H_{-1} = H_1
    D4[-1, -1] += 1 / h ** 4
    return sig * D2 - c["K"] * D4 - c["c"] * D1, h, I


def global_growth(c, eps, Lp):
    A, _, _ = patch_operator(c, eps, Lp)
    ev = np.linalg.eigvals(A)
    k = np.argmax(ev.real)
    return ev[k]


def eps_global(c, Lp):
    try:
        return brentq(lambda e: global_growth(c, e, Lp).real, 1e-4, 20.0, xtol=1e-3)
    except ValueError:
        return None


conv = {}
for Lc, c in co.items():
    ea = eps_absolute(c)
    rows = []
    for Lp in (50, 100, 200):   # longer patches: the eigenvalues of this strongly non-normal operator are polluted by
        # round-off (pseudospectra); the threshold of long patches tends to eps_a
        eg = eps_global(c, Lp)
        lam = global_growth(c, eg, Lp) if eg is not None else None
        rows.append(dict(Lp=Lp, eps_g=eg, omega_g=float(lam.imag) if lam is not None else None))
    conv[Lc] = dict(eps_a=ea, global_threshold=rows, periodic_threshold=0.0)
    print(f"Lc = {Lc:g}: absolute threshold eps_a = {ea}, global eps_g(L_p) = "
          f"{[(r['Lp'], None if r['eps_g'] is None else round(r['eps_g'], 4)) for r in rows]}  [{time.time() - t0:.0f} s]",
          flush=True)


# nonlinear run in a finite patch (dense lattice, L_p = 400): implicit Euler in H with Newton on the cubic/quintic
def run_patch(c, eps, Lp=400.0, n=570, dt=0.5, t_end=1e5, n_out=200, seed=2):
    A, h, I = patch_operator(c, eps, Lp, n)
    rng = np.random.default_rng(seed)
    H = 1e-3 * rng.standard_normal(n)
    # nonlinear flux d_x (g3 u^3 + g5 u^5) with u = H_x at half points (conservative)
    def nl(H):
        Hp = np.concatenate([[0.0], H, [0.0]])
        u = np.diff(Hp) / h
        f = c["g3"] * u ** 3 + c["g5"] * u ** 5
        return np.diff(f) / h
    M = I - dt * A
    lu = sla.lu_factor(M)
    nt = int(t_end / dt)
    every = max(1, nt // n_out)
    ts, Hs = [], []
    for k in range(nt + 1):
        if k % every == 0:
            ts.append(k * dt)
            Hs.append(H.copy())
        # semi-implicit: linear part implicit, nonlinear explicit (small slopes, dt checked against blow-up)
        H = sla.lu_solve(lu, H + dt * nl(H) + 1e-6 * np.sqrt(dt) * rng.standard_normal(n))
        if not np.all(np.isfinite(H)) or np.abs(H).max() > 1e4:
            break
    return dict(x=np.linspace(h, Lp - h, n), t=np.array(ts), H=np.array(Hs))


patch_runs = {}
for Lc in (6.0, 8.0):
    eg = [r["eps_g"] for r in conv[Lc]["global_threshold"] if r["Lp"] == 200][0]
    if eg is None:
        continue
    eps = 1.2 * eg
    name = f"L{Lc:g}_patch200_eps{eps:.3g}"
    r = run_patch(co[Lc], eps, Lp=200.0, n=799, dt=0.02, t_end=3e4)
    np.savez_compressed(os.path.join(out, f"patch_{name}.npz"), **r)
    patch_runs[name] = dict(Lc=Lc, eps=eps, final_max_H=float(np.abs(r["H"][-1]).max()))
    print(f"patch {name}: final max|H| = {np.abs(r['H'][-1]).max():.4f}  [{time.time() - t0:.0f} s]", flush=True)

json.dump(dict(coefficients=co, uniform_branches=uniform, convective=conv, patch_runs=patch_runs,
               periodic_runs={k: dict(final_max_u=float(np.abs(v["u"][-1]).max()),
                                      walls=[int(x) for x in v["n_walls"]]) for k, v in runs.items()}),
          open(os.path.join(out, "slope_equation.json"), "w"), indent=1, default=float)
print("done", flush=True)
