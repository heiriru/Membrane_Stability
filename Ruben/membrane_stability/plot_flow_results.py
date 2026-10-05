'''
Figures of the flow-driven stability results (results/flow_* and results/time_check_*), written to figures/flow_*.png.

run with:  python3 plot_flow_results.py
'''
import csv
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 160,
                     "savefig.bbox": "tight"})
MESH_LABEL = {"A": "mesh A (h 0.15-3)", "B": "mesh B (h 0.1-2)", "C": "mesh C (h 0.07-1.4)"}
COLORS = {"A": "#1f77b4", "B": "#ff7f0e", "C": "#2ca02c"}


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ("", "None") else np.nan for r in rows]) for k in rows[0]}


def load_run(name):
    path = os.path.join(RES, name, "critical_search.csv")
    if not os.path.exists(path):
        return None, None
    d = read_csv(path)
    order = np.argsort(d["v0"])
    d = {k: v[order] for k, v in d.items()}
    crit = os.path.join(RES, name, "critical.json")
    return d, (json.load(open(crit)) if os.path.exists(crit) else None)


def triangulation(d, r_hole=None, centre=None):
    if "triangles" in d:
        return mtri.Triangulation(d["x"], d["y"], d["triangles"])
    tri = mtri.Triangulation(d["x"], d["y"])
    if r_hole is not None:
        xc = d["x"][tri.triangles].mean(1)
        yc = d["y"][tri.triangles].mean(1)
        tri.set_mask(np.hypot(xc - centre[0], yc - centre[1]) < r_hole)
    return tri


# 1. PRE setup: leading eigenvalue vs SL, perfect (t = 0) and imperfect (t = -0.3) PI, and amplitude of the steady state
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
summary = {}
for m in "ABC":
    d, crit = load_run(f"flow_pre_t0_mesh{m}")
    if d is None:
        continue
    SL = d["v0"] * 100.0
    ax[0].plot(SL, d["lead_re"] * 1e6, "o-", color=COLORS[m], ms=4, label=f"t = 0, {MESH_LABEL[m]}")
    if crit:
        ax[0].axvline(crit["SL_c"], color=COLORS[m], ls=":", lw=1)
        summary[f"t0_mesh{m}"] = crit["SL_c"]
for m in "ABC":
    d, _ = load_run(f"flow_pre_t-0.3_mesh{m}")
    if d is None:
        continue
    SL = d["v0"] * 100.0
    ax[0].plot(SL, d["lead_re"] * 1e6, "s--", color=COLORS[m], ms=4, mfc="none", label=f"t = -0.3, {MESH_LABEL[m]}")
    ax[1].plot(SL, d["max_abs_z"], "s-", color=COLORS[m], ms=4, label=f"t = -0.3, {MESH_LABEL[m]}")
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xlabel("Scriven-Love number SL = eta v0 L / kappa")
ax[0].set_ylabel("leading eigenvalue  [1e-6 kappa / (zeta r0^4)]")
ax[0].set_title("PRE setup: slowest relaxation rate vs flow", fontsize=11)
ax[0].legend(fontsize=8)
ax[1].set_xlabel("SL")
ax[1].set_ylabel("max |z*|  [r0]")
ax[1].set_title("flow-induced deformation of the steady state (t = -0.3)", fontsize=11)
ax[1].legend(fontsize=8)
fig.savefig(os.path.join(FIG, "flow_pre_eigenvalue_vs_SL.png"))
plt.close(fig)

# 2. mechanism: tension, smallest principal stress, unstable mode at threshold; imperfect steady state
path = os.path.join(RES, "flow_pre_t0_meshA", "mechanism_at_threshold.npz")
if os.path.exists(path):
    d = np.load(path)
    tri = triangulation(d)
    fig, ax = plt.subplots(1, 4, figsize=(21, 4.6))
    items = [("sigma", "surface tension sigma", 0.01, "RdBu_r"),
             ("T_min", "smallest principal stress of sigma g + 2 eta d", 0.01, "RdBu_r"),
             ("mode", "unstable mode z' at SL_c", None, "RdBu_r"),
             ("vx", "velocity v^1", None, "viridis")]
    for a, (k, title, lim, cmap) in zip(ax, items):
        f = d[k]
        if cmap == "viridis":
            c = a.tripcolor(tri, f, shading="gouraud", cmap=cmap)
        else:
            l = lim or np.abs(f).max()
            c = a.tripcolor(tri, f, shading="gouraud", cmap=cmap, vmin=-l, vmax=l)
        a.set_aspect("equal")
        a.set_title(title, fontsize=10)
        a.set_xlabel("x / r0")
        fig.colorbar(c, ax=a, shrink=0.8)
    fig.suptitle(f"Mechanism at the threshold (t = 0, SL = {float(d['v0']) * 100:.2f}): the drag on the PI is carried by "
                 f"a tension gradient, the membrane upstream is compressed (sigma < 0, blue) and buckles", fontsize=11)
    fig.savefig(os.path.join(FIG, "flow_pre_mechanism.png"))
    plt.close(fig)

    # tension along the centre line
    fig, ax = plt.subplots(figsize=(7, 4))
    sel = np.abs(d["y"] - 50.0) < 0.6
    o = np.argsort(d["x"][sel])
    ax.plot(d["x"][sel][o], d["sigma"][sel][o], ".-", ms=3)
    ax.axhline(0, color="k", lw=0.8)
    ax.axhline(float(d["sigma_r"]), color="gray", ls="--", lw=1, label="sigma0 (outflow)")
    ax.set_ylim(-0.03, 0.03)
    ax.set_xlabel("x / r0  (y = L/2, flow from left to right)")
    ax.set_ylabel("sigma  [kappa / r0^2]")
    ax.set_title("tension along the centre line at SL_c")
    ax.legend()
    fig.savefig(os.path.join(FIG, "flow_pre_tension_centreline.png"))
    plt.close(fig)

path0 = os.path.join(RES, "flow_pre_t-0.3_meshA", "leading_modes.npz")
if os.path.exists(path0):
    d = np.load(path0)
    tri = triangulation(d, 1.0, (50, 50))
    keys = [k for k in ["0", "0.04", "0.08", "0.1", "0.16"] if f"base_{k}" in d]
    fig, ax = plt.subplots(2, len(keys), figsize=(4.2 * len(keys), 7.5))
    for k, key in enumerate(keys):
        for row, name in enumerate([f"base_{key}", key]):
            z = d[name]
            lim = np.abs(z).max()
            c = ax[row, k].tripcolor(tri, z, shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[row, k].set_aspect("equal")
            ax[row, k].set_title(("steady shape z*" if row == 0 else "slowest mode z'") + f", SL = {float(key) * 100:g}",
                                 fontsize=10)
            fig.colorbar(c, ax=ax[row, k], shrink=0.75)
    fig.suptitle("PRE setup with contact angle t = -0.3 (imperfect bifurcation): the flow-induced deformation is the "
                 "buckling mode, growing smoothly", fontsize=11)
    fig.savefig(os.path.join(FIG, "flow_pre_imperfect_shapes.png"))
    plt.close(fig)

# 3. threshold vs tension: SL_c vs Gamma = sigma0 L^2 / kappa (mesh A)
pts = []
for f in [os.path.join(RES, "flow_pre_t0_meshA", "critical.json")] + \
        glob.glob(os.path.join(RES, "flow_pre_t0_meshA_sigma*", "critical.json")):
    if os.path.exists(f):
        c = json.load(open(f))
        pts.append((c["sigma_r"] * c["L"] ** 2 / c["kappa"], c["SL_c"]))
pts = sorted(set(pts))
if len(pts) >= 2:
    pts = np.array(pts)
    fit = np.polyfit(pts[:, 0], pts[:, 1], 1)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(pts[:, 0], pts[:, 1], "o", ms=7, label="stability analysis (mesh A)")
    g = np.linspace(0, pts[:, 0].max() * 1.05, 50)
    ax.plot(g, np.polyval(fit, g), "--", label=f"linear fit: SL_c = {fit[1]:.2f} + {fit[0]:.3f} Gamma")
    ax.set_xlabel("Gamma = sigma0 L^2 / kappa")
    ax.set_ylabel("critical Scriven-Love number SL_c")
    ax.set_title("flow-driven buckling threshold vs membrane tension")
    ax.legend()
    fig.savefig(os.path.join(FIG, "flow_pre_SLc_vs_tension.png"))
    plt.close(fig)
    json.dump(dict(points=pts.tolist(), fit=fit.tolist()), open(os.path.join(RES, "flow_pre_SLc_vs_tension.json"), "w"),
              indent=1)

# 4. time-dependent cross-check
runs = sorted(glob.glob(os.path.join(RES, "time_check_*", "time_check.json")))
if runs:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for f in runs:
        c = json.load(open(f))
        ts = read_csv(os.path.join(os.path.dirname(f), "time_series.csv"))
        l, = ax.semilogy(ts["t"] * 1e-3, ts["amplitude"], lw=2,
                         label=f"SL = {c['SL']:.0f}: time stepping, fitted rate {c['fitted_rate']:+.3e}")
        ax.semilogy(ts["t"] * 1e-3, ts["amplitude"][0] * np.exp(c["eigenvalue_re"] * ts["t"]), "k--", lw=1)
        ax.plot([], [], "k--", label="exp(lambda t), lambda from the eigenvalue problem" if f == runs[0] else None)
    ax.set_xlabel("t  [1e3 zeta r0^4 / kappa]")
    ax.set_ylabel("rms |z - z*|")
    ax.set_title("nonlinear time stepping (BDF2) vs eigenvalue, PRE setup t = 0")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(FIG, "flow_time_check.png"))
    plt.close(fig)

# 5. IRENE example (square_a, rho = 1): bending waves slow down and collide at zero (divergence)
path = os.path.join(RES, "flow_irene_example_scan", "scan_spectra.npz")
if os.path.exists(path):
    sp = np.load(path)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    for key in sorted(sp.files, key=float):
        lam = sp[key]
        v0 = float(key)
        near = lam[lam.real > -0.05 * np.abs(lam) - 1e-8]
        osc = near[np.abs(near.imag) > 1e-6 * np.abs(near)]
        ax[0].plot(np.full(len(osc), v0), np.abs(osc.imag), "o", color="C0", ms=4)
        real = lam[(np.abs(lam.imag) < 1e-6 * np.abs(lam) + 1e-12) & (np.abs(lam) < 150)]
        ax[1].plot(np.full(len(real), v0), real.real, "s", color="C3", ms=4)
        cplx = lam[(np.abs(lam.imag) > 1e-6 * np.abs(lam)) & (np.abs(lam.real) > 5)]
        ax[1].plot(np.full(len(cplx), v0), cplx.real, "^", color="C2", ms=4)
    _, crit = load_run("flow_critical_rho1_eta1_m003")
    for a in ax:
        if crit:
            a.axvline(crit["v0_c"], color="k", ls=":", label=f"v0_c = {crit['v0_c']:.3f}")
        a.set_xlabel("inflow velocity v0")
        a.legend()
    ax[0].set_ylabel("|Im lambda| of the bending waves")
    ax[0].set_title("bending-wave frequencies (IRENE example, coarse mesh)", fontsize=10)
    ax[1].plot([], [], "s", color="C3", label="real eigenvalues")
    ax[1].plot([], [], "^", color="C2", label="Re of complex (|Re| > 5)")
    ax[1].legend(fontsize=8)
    ax[1].set_ylabel("Re lambda")
    ax[1].set_title("collision at 0 -> real pair +/- s (divergence)", fontsize=10)
    fig.savefig(os.path.join(FIG, "flow_irene_example_collision.png"))
    plt.close(fig)

# 6. weak growth of the bending waves at small v0 (numerical, converges to 0 with h^3) and threshold vs mesh / alpha
rows = []
for name, h in [("flow_critical_rho1_eta1_m003", 0.03), ("flow_critical_rho1_eta1_m0018", 0.018),
                ("flow_critical_rho1_eta1_m0012", 0.012)]:
    d, crit = load_run(name)
    if d is None:
        continue
    for v0 in (2.0, 8.0):
        k = np.where(np.isclose(d["v0"], v0))[0]
        if len(k):
            rows.append((h, v0, d["lead_re"][k[0]], crit["v0_c"] if crit else np.nan))
if rows:
    rows = np.array(rows)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    for v0, mk in ((2.0, "o"), (8.0, "s")):
        s = rows[rows[:, 1] == v0]
        ax[0].loglog(s[:, 0], s[:, 2], mk + "-", label=f"v0 = {v0:g}")
        if len(s) >= 2:
            p = np.polyfit(np.log(s[:, 0]), np.log(s[:, 2]), 1)[0]
            ax[0].text(s[-1, 0], s[-1, 2], f"  ~ h^{p:.1f}", fontsize=9)
    ax[0].set_xlabel("mesh resolution h")
    ax[0].set_ylabel("largest Re lambda of the bending waves")
    ax[0].set_title("weak oscillatory growth at small v0: numerical (-> 0 with h)")
    ax[0].legend()
    s = rows[rows[:, 1] == 8.0]
    ax[1].plot(s[:, 0], s[:, 3], "o-", label="alpha = 100 (IRENE default)")
    for a, mk in (("1e3", "^"), ("1e4", "v")):
        _, c = load_run(f"flow_critical_rho1_eta1_alpha{a}_m003")
        if c:
            ax[1].plot([0.03], [c["v0_c"]], mk, ms=8, label=f"alpha = {a}")
    ax[1].set_xlabel("mesh resolution h")
    ax[1].set_ylabel("critical inflow v0_c")
    ax[1].set_title("IRENE example: threshold depends on mesh and penalty alpha")
    ax[1].legend()
    fig.savefig(os.path.join(FIG, "flow_irene_example_convergence.png"))
    plt.close(fig)

json.dump(summary, open(os.path.join(RES, "flow_summary.json"), "w"), indent=1)
print("figures written to", FIG)
