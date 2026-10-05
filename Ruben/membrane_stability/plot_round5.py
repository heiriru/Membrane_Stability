'''
Figures of the fifth study (results/round5, collected by collect_round5.sh):
    r5_flutter.png        nonlinear flutter: saturated wave emitted by the PI, Landau coefficient of the Hopf bifurcation
    r5_slope_equation.png long-wave slope equation of the lattice: terraces, hysteresis, convective vs absolute
    r5_proteins_dyn.png   IRENE + mobile proteins, nonlinear time stepping (energy check, patterns)
    r5_peclet.png         protein demixing and buckling thresholds vs the Peclet number (mobility M)
    r5_fold_parametric.png post-snap fold beyond the Monge gauge (parametric surface) vs the Monge run
    python3 plot_round5.py
'''
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(HERE, "results", "round5")
R4 = os.path.join(HERE, "results", "round4")
FIG = os.path.join(HERE, "figures")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY, INK = "#8a8985", "#2b2a27"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 1.8, "legend.frameon": False})
summary = {}


def rows(p):
    if not os.path.exists(p):
        return []
    out = list(csv.DictReader(open(p)))
    for r in out:
        for k in r:
            try:
                r[k] = float(r[k])
            except (ValueError, TypeError):
                pass
    return out


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------------------------------------ flutter
def envelope(t, z, omega):
    '''amplitude sqrt(z^2 + (dz/dt / omega)^2) of a nearly harmonic signal'''
    dz = np.gradient(z, t)
    return np.sqrt(z ** 2 + (dz / omega) ** 2)


def flutter_analysis(run):
    rr = rows(os.path.join(run, "time_series.csv"))
    info = json.load(open(os.path.join(run, "flutter.json"))) if os.path.exists(os.path.join(run, "flutter.json")) \
        else None
    if not rr:
        return None
    t = np.array([r["t"] for r in rr])
    T = info["period"] if info else 7.8915e5
    om = 2 * np.pi / T
    probes = [k for k in rr[0] if k.startswith("z_at_")]
    A = np.mean([envelope(t, np.array([r[k] for r in rr]), om) for k in probes if k in ("z_at_-15", "z_at_-8", "z_at_-30")],
                axis=0)
    return dict(t=t, T=T, om=om, A=A, rows=rr, info=info)


def fig_flutter():
    runs = {}
    for f, lab in (("1.05_k", "1.05 v0_c"), ("1.10_k", "1.10 v0_c"), ("1.20_k", "1.20 v0_c"),
                   ("0.97_k", "0.97 v0_c, kick 2 r0"), ("0.97_from", "0.97 v0_c, from the 1.05 limit cycle")):
        p = glob.glob(os.path.join(R5, f"flut_v{f}*"))
        if p:
            a = flutter_analysis(p[0])
            if a:
                runs[lab] = (a, p[0])
    if not runs:
        return
    fig = plt.figure(figsize=(14, 8))
    a0, a1, a2 = (fig.add_subplot(2, 3, k) for k in (1, 2, 3))
    res = {}
    for k, (lab, (a, path)) in enumerate(runs.items()):
        tt = a["t"] / a["T"]
        a0.plot(tt, [r["z_at_-15"] for r in a["rows"]], "-", color=C[k], lw=1.2, label=lab)
        a1.semilogy(tt, np.maximum(a["A"], 1e-8), "-", color=C[k], label=lab)
        # growth rate of the envelope vs A^2 (Landau): d ln A / dt = s + l A^2, fitted after the initial transient
        lnA = np.log(np.maximum(a["A"], 1e-12))
        g = np.gradient(lnA, a["t"])
        sel = (tt > 0.6) & (a["A"] > 1e-3)
        if sel.sum() > 6:
            # smooth over one period
            n = max(3, int(len(tt) / max(tt[-1], 1)))
            gs = np.convolve(g, np.ones(n) / n, mode="same")
            a2.plot(a["A"][sel] ** 2, gs[sel] * a["T"], ".", color=C[k], ms=3, label=lab)
            m = sel & (np.arange(len(tt)) > n) & (np.arange(len(tt)) < len(tt) - n)
            if m.sum() > 6:
                fit = np.polyfit(a["A"][m] ** 2, gs[m], 1)
                xs = np.linspace(0, (a["A"][m] ** 2).max(), 20)
                a2.plot(xs, np.polyval(fit, xs) * a["T"], "-", color=C[k], lw=1)
                late = a["A"][tt > tt[-1] - 1.5]
                res[lab] = dict(landau_l=float(fit[0]), growth_s=float(fit[1]), A_sat=float(late.mean()),
                                A_sat_spread=float(late.std()), t_end_over_T=float(tt[-1]))
    a0.set_xlabel("t / T (flutter period)")
    a0.set_ylabel("z(x = −15 r0) / r0")
    a0.set_title("(a) height upstream of the PI", loc="left")
    a0.legend(fontsize=7)
    a1.set_xlabel("t / T")
    a1.set_ylabel("envelope amplitude A / r0")
    a1.set_title("(b) growth and saturation", loc="left")
    a2.axhline(0, color=INK, lw=0.7)
    a2.set_xlabel("A² / r0²")
    a2.set_ylabel("T d ln A / dt")
    a2.set_title("(c) Landau: d ln A/dt = s + l A² (l < 0: supercritical)", loc="left", fontsize=8)
    a2.legend(fontsize=7)
    # kymograph and snapshots of the most strongly driven run
    lab = "1.10 v0_c" if "1.10 v0_c" in runs else list(runs)[0]
    a, path = runs[lab]
    sp = os.path.join(path, "snapshots.npz")
    if os.path.exists(sp):
        d = np.load(sp)
        tri = mtri.Triangulation(d["x"] - 50, d["y"] - 50, d["triangles"])
        xs = np.linspace(-49, 49, 300)
        K = np.array([mtri.LinearTriInterpolator(tri, z)(xs, 0 * xs).filled(np.nan) for z in d["z"]])
        a3 = fig.add_subplot(2, 3, 4)
        lim = np.nanmax(np.abs(K))
        a3.grid(False)
        a3.pcolormesh(xs, d["t"] / a["T"], K, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        a3.set_xlabel("x / r0 (flow →), centreline")
        a3.set_ylabel("t / T")
        a3.set_title(f"(d) space-time diagram, {lab}", loc="left")
        a3.grid(False)
        for kk, frac in enumerate((0.75, 1.0)):
            i = int(frac * (len(d["t"]) - 1))
            ax = fig.add_subplot(2, 3, 5 + kk)
            lim2 = np.abs(d["z"][i]).max()
            pc = ax.tripcolor(tri, d["z"][i], shading="gouraud", cmap="RdBu_r", vmin=-lim2, vmax=lim2)
            ax.add_patch(plt.Circle((0, 0), 1, color=INK))
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_title(f"({'ef'[kk]}) z at t/T = {d['t'][i] / a['T']:.2f}", loc="left")
            fig.colorbar(pc, ax=ax, shrink=0.75)
    fig.tight_layout()
    save(fig, "r5_flutter.png")
    summary["flutter"] = res


# ----------------------------------------------------------------------------------------------------- slope equation
def fig_slope():
    p = os.path.join(R5, "slope", "slope_equation.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    co = d["coefficients"]
    fig = plt.figure(figsize=(14, 8))
    a0 = fig.add_subplot(2, 3, 1)
    for k, (Lc, br) in enumerate(d["uniform_branches"].items()):
        st = [b for b in br if b["stable"]]
        un = [b for b in br if not b["stable"]]
        lab = f"Lc = {float(Lc):g} (φ = {co[Lc]['phi']:.3f})"
        a0.plot([b["eps"] for b in st], [b["u"] for b in st], ".", color=C[k], ms=3, label=lab)
        a0.plot([b["eps"] for b in un], [b["u"] for b in un], ".", color=C[k], ms=1, alpha=0.4)
    a0.axvline(0, color=INK, lw=0.7)
    a0.set_ylim(0, 0.4)
    a0.set_xlabel("ε = D / D_lw − 1")
    a0.set_ylabel("uniform slope u of the lattice")
    a0.set_title("(a) slope states: stable (dots), unstable (faint)", loc="left")
    a0.legend(fontsize=6)
    for k, (name, ttl) in enumerate((("dense_eps0.05", "(b) dense (φ = 8.7 %), ε = 0.05: terraces"),
                                     ("dilute_eps0.05", "(c) dilute (φ = 4.9 %), ε = 0.05: steep terraces"))):
        q = os.path.join(R5, "slope", f"periodic_{name}.npz")
        if not os.path.exists(q):
            continue
        r = np.load(q)
        ax = fig.add_subplot(2, 3, 2 + k)
        lim = np.abs(r["H"]).max()
        ax.grid(False)
        ax.pcolormesh(r["x"], r["t"] / 1e3, r["H"], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        ax.set_xlabel("x / r0")
        ax.set_ylabel("t / 10³")
        ax.set_title(ttl, loc="left", fontsize=8)
        ax.grid(False)
        ins = ax.inset_axes([0.55, 0.06, 0.42, 0.3])
        ins.plot(r["x"], r["H"][-1], color=INK, lw=0.8)
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_title("H(x), final", fontsize=6)
    q = os.path.join(R5, "slope", "periodic_dilute_hysteresis.npz")
    if os.path.exists(q):
        r = np.load(q)
        a3 = fig.add_subplot(2, 3, 4)
        mu = np.abs(r["u"]).max(axis=1)
        half = len(r["t"]) // 2
        a3.plot(r["eps"][:half], mu[:half], "-", color=C[1], label="drive decreasing")
        a3.plot(r["eps"][half:], mu[half:], "--", color=C[0], label="drive increasing")
        a3.axvline(0, color=INK, lw=0.7)
        a3.set_xlabel("ε")
        a3.set_xticks([-0.15, -0.1, -0.05, 0, 0.05])
        a3.set_ylabel("max |u|")
        a3.set_title("(d) dilute lattice: hysteresis (fold at ε ≈ −0.045)", loc="left")
        a3.legend(fontsize=7)
    a4 = fig.add_subplot(2, 3, 5)
    phis = [co[L]["phi"] for L in d["convective"]]
    a4.semilogx(phis, [d["convective"][L]["eps_a"] for L in d["convective"]], "-o", color=INK, label="absolute ε_a")
    for k, Lp in enumerate((50, 100, 200)):
        a4.semilogx(phis, [[g["eps_g"] for g in d["convective"][L]["global_threshold"] if g["Lp"] == Lp][0]
                           for L in d["convective"]], "-o", color=C[k], ms=3, label=f"finite patch, L_p = {Lp} r0")
    a4.axhline(0, color=GRAY, lw=0.8, ls=":")
    a4.set_xlabel("area fraction φ")
    a4.set_ylabel("ε at onset")
    a4.set_title("(e) the lattice instability is convective up to ε ≈ 1", loc="left", fontsize=8)
    a4.legend(fontsize=7)
    pr = sorted(glob.glob(os.path.join(R5, "slope", "patch_*.npz")))
    if pr:
        r = np.load(pr[0])
        a5 = fig.add_subplot(2, 3, 6)
        lim = np.abs(r["H"]).max()
        a5.grid(False)
        a5.pcolormesh(r["x"], r["t"] / 1e3, r["H"], cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
        a5.set_xlabel("x / r0 (membrane flows →, pattern drifts ←)")
        a5.set_ylabel("t / 10³")
        a5.set_title(f"(f) finite patch: {os.path.basename(pr[0])[6:-4]}", loc="left", fontsize=8)
        a5.grid(False)
    fig.tight_layout()
    save(fig, "r5_slope_equation.png")
    summary["slope"] = dict(eps_a={L: d["convective"][L]["eps_a"] for L in d["convective"]},
                            periodic=d.get("periodic_runs"), patch=d.get("patch_runs"))


# ------------------------------------------------------------------------------------------------- protein dynamics
def fig_proteins_dyn():
    names = []
    for cands in (("phidyn_rest",), ("phidyn_flow3", "phidyn_flow"), ("phidyn_source3", "phidyn_source")):
        for n in cands:
            if os.path.exists(os.path.join(R5, n, "snapshots.npz")):
                names.append(n)
                break
    if not names:
        return
    fig = plt.figure(figsize=(14, 8))
    a0 = fig.add_subplot(2, 3, 1)
    a1 = fig.add_subplot(2, 3, 2)
    a2 = fig.add_subplot(2, 3, 3)
    res = {}
    for k, n in enumerate(names):
        rr = rows(os.path.join(R5, n, "time_series.csv"))
        t = np.array([r["t"] for r in rr])
        G = np.array([r["G"] for r in rr])
        a0.plot(t / 1e3, G - G[0], "-", color=C[k], label=n.split("_")[1].rstrip("23"))
        a1.plot(t / 1e3, [max(abs(r["phi_min"]), abs(r["phi_max"])) for r in rr], "-", color=C[k], label=n.split("_")[1].rstrip("23"))
        a2.plot(t / 1e3, [max(abs(r["z_min"]), abs(r["z_max"])) for r in rr], "-", color=C[k], label=n.split("_")[1].rstrip("23"))
        res[n] = dict(dG_monotone=bool(np.all(np.diff(G) <= 1e-9)) if n == "phidyn_rest" else None,
                      N_drift=float(rr[-1]["N"] - rr[0]["N"]), t_end=float(t[-1]),
                      phi_range=[rr[-1]["phi_min"], rr[-1]["phi_max"]], z_range=[rr[-1]["z_min"], rr[-1]["z_max"]])
    a0.set_xlabel("t / 10³")
    a0.set_ylabel("G(t) − G(0)")
    a0.set_title("(a) free energy (must decrease at rest)", loc="left")
    a0.legend(fontsize=7)
    a1.set_xlabel("t / 10³")
    a1.set_ylabel("max |φ|")
    a1.set_title("(b) protein demixing amplitude", loc="left")
    a2.set_xlabel("t / 10³")
    a2.set_ylabel("max |z| / r0")
    a2.set_title("(c) membrane deformation", loc="left")
    for k, n in enumerate(names):
        d = np.load(os.path.join(R5, n, "snapshots.npz"))
        ax = fig.add_subplot(2, 3, 4 + k)
        tri = mtri.Triangulation(d["x"] - 50, d["y"] - 50, d["triangles"])
        ph = d["phi"][-1]
        lim = np.abs(ph).max()
        pc = ax.tripcolor(tri, ph, shading="gouraud", cmap="PuOr_r", vmin=-lim, vmax=lim)
        zz = d["z"][-1]
        zl = np.abs(zz).max()
        if zl > 0:
            ax.tricontour(tri, zz, levels=np.linspace(-zl, zl, 9)[1:-1], colors=[INK], linewidths=0.5)
        ax.add_patch(plt.Circle((0, 0), 1, color=INK))
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(f"({'def'[k]}) {n.split('_')[1].rstrip('23')}: φ (colour), z (lines), t = {d['t'][-1]:.3g}", loc="left",
                     fontsize=8)
        fig.colorbar(pc, ax=ax, shrink=0.7)
    fig.tight_layout()
    save(fig, "r5_proteins_dyn.png")
    summary["proteins_dyn"] = res


# ------------------------------------------------------------------------------------------------------------ Peclet
def fig_peclet():
    Ms = [1, 10, 100, 1000]
    have = [M for M in Ms if os.path.exists(os.path.join(R5, f"pa_M{M}_chic", "phi_stability.json"))
            or os.path.exists(os.path.join(R5, f"pa_M{M}_v0c", "phi_stability.json"))]
    if not have:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    res = {}
    for k, M in enumerate(have):
        p = os.path.join(R5, f"pa_M{M}_chic", "phi_stability.json")
        if os.path.exists(p):
            d = json.load(open(p))
            cc = [r for r in d["chi_c"] if r.get("chi_c") is not None]
            ax[0].plot([r["SL"] for r in cc], [r["chi_c"] for r in cc], "-o", color=C[k], ms=3, label=f"M = {M}")
            res.setdefault(M, {})["chi_c"] = d["chi_c"]
        p = os.path.join(R5, f"pa_M{M}_v0c", "phi_stability.json")
        if os.path.exists(p):
            d = json.load(open(p))
            pb = os.path.join(R5, f"pa_M{M}_v0c_b", "phi_stability.json")
            if os.path.exists(pb):
                have_chi = {r["chi"] for r in d["v0_c"]}
                d["v0_c"] = d["v0_c"] + [r for r in json.load(open(pb))["v0_c"] if r["chi"] not in have_chi]
            vv = sorted([r for r in d["v0_c"] if r.get("v0_c") is not None], key=lambda r: r["chi"])
            ax[1].plot([r["chi"] for r in vv], [r["SL_c"] for r in vv], "-o", color=C[k], ms=3, label=f"M = {M}")
            res.setdefault(M, {})["v0_c"] = d["v0_c"]
    xe = np.linspace(0.02, 1.0, 50)
    ax[1].plot(xe, 27.50 * xe / (xe + 0.25), "--", color=GRAY, lw=1, label="proteins in equilibrium")
    ax[1].axhline(27.5, color=GRAY, ls=":", lw=1)
    ax[0].set_xlabel("SL = η v0 L / κ")
    ax[0].set_ylabel("protein threshold χ_c")
    ax[0].set_title("(a) demixing threshold vs flow, by mobility", loc="left")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("χ")
    ax[1].set_ylabel("buckling threshold SL_c")
    ax[1].set_title("(b) buckling threshold vs χ, by mobility", loc="left")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "r5_peclet.png")
    summary["peclet"] = res


# ------------------------------------------------------------------------------------------------ parametric fold
def fig_fold():
    p = [q for q in (os.path.join(R5, "param_eq"), os.path.join(R5, "param_g05"), os.path.join(R5, "param_ff_v0.97r")) if os.path.isdir(q)]
    if not p or not os.path.exists(os.path.join(p[0], "time_series.csv")):
        return
    rr = rows(os.path.join(p[0], "time_series.csv"))
    mo = rows(os.path.join(R4, "snapff_v0.97", "time_series.csv"))
    fig = plt.figure(figsize=(14, 8))
    a0, a1, a2 = (fig.add_subplot(2, 3, k) for k in (1, 2, 3))
    t0 = rr[0]["t"]
    for a, key, lab in ((a0, "h", "PI height h / r0"), (a1, "z_min", "z_min / r0")):
        a.plot([(r["t"] - t0) / 1e3 for r in mo if r["t"] >= t0], [r[key] for r in mo if r["t"] >= t0], "--",
               color=GRAY, label="Monge (round 4)")
        a.plot([(r["t"] - t0) / 1e3 for r in rr], [r[key] for r in rr], "-", color=C[0], label="parametric surface")
        a.set_xlabel("t − t_start [10³]")
        a.set_ylabel(lab)
        a.legend(fontsize=7)
    a0.set_title("(a) PI height: parametric vs Monge", loc="left")
    a1.set_title("(b) pit depth", loc="left")
    a2.plot([(r["t"] - t0) / 1e3 for r in rr], [r["min_nz"] for r in rr], "-", color=C[0], label="min n_z (parametric)")
    a2.plot([(r["t"] - t0) / 1e3 for r in mo if r["t"] >= t0], [1 / np.sqrt(1 + r["max_slope"] ** 2) for r in mo
                                                               if r["t"] >= t0], "--", color=GRAY, label="Monge")
    a2.axhline(0, color=INK, lw=0.7)
    a2.set_xlabel("t − t_start [10³]")
    a2.set_ylabel("min n_z (0: vertical wall, < 0: overhang)")
    a2.set_title("(c) steepest wall", loc="left")
    a2.legend(fontsize=7)
    sp = os.path.join(p[0], "snapshots.npz")
    if os.path.exists(sp):
        d = np.load(sp)
        X = d["X"]
        u = d["u"]
        a3 = fig.add_subplot(2, 3, 4)
        sel = np.abs(u[:, 1] - 50) < 0.8
        for k, i in enumerate(np.linspace(0, len(d["t"]) - 1, 5).astype(int)):
            o = np.argsort(u[sel, 0])
            a3.plot(X[i][sel][o, 0] - 50, X[i][sel][o, 2], "-", color=plt.cm.viridis(k / 4), lw=1.3,
                    label=f"t − t0 = {(d['t'][i] - d['t'][0]) / 1e3:.2f}e3")
        a3.set_aspect("equal")
        a3.set_xlabel("x / r0 (centreline)")
        a3.set_ylabel("z / r0")
        a3.set_title("(d) centreline profiles (true aspect ratio)", loc="left")
        a3.legend(fontsize=6)
        a4 = fig.add_subplot(2, 3, 5)
        i = len(d["t"]) - 1
        tri = mtri.Triangulation(X[i][:, 0] - 50, X[i][:, 1] - 50, d["triangles"])
        pc = a4.tripcolor(tri, X[i][:, 2], shading="flat", cmap="RdBu_r")
        a4.set_aspect("equal")
        a4.grid(False)
        a4.set_xlim(-35, 35)
        a4.set_ylim(-35, 35)
        a4.set_title("(e) final surface seen from above", loc="left")
        fig.colorbar(pc, ax=a4, shrink=0.75)
        a5 = fig.add_subplot(2, 3, 6)
        drift = np.hypot(X[i][:, 0] - u[:, 0], X[i][:, 1] - u[:, 1])
        pc = a5.tripcolor(mtri.Triangulation(u[:, 0] - 50, u[:, 1] - 50, d["triangles"]), drift, shading="gouraud",
                          cmap="viridis")
        a5.set_aspect("equal")
        a5.grid(False)
        a5.set_xlim(-35, 35)
        a5.set_ylim(-35, 35)
        a5.set_title("(f) horizontal drift of the parametrization", loc="left")
        fig.colorbar(pc, ax=a5, shrink=0.75)
    fig.tight_layout()
    save(fig, "r5_fold_parametric.png")
    summary["fold"] = dict(final=rr[-1], n=len(rr))


def fig_tricritical():
    files = {6.0: os.path.join(R5, "lat_fine", "nonlinear_cell_L6.json"), 8.0: os.path.join(R5, "lat_fine", "nonlinear_cell_L8.json")}
    for L in (6.5, 7.0, 7.5):
        files[L] = os.path.join(R5, "lat_tc", f"nonlinear_cell_L{L:g}.json")
    files = {L: f for L, f in files.items() if os.path.exists(f)}
    if len(files) < 3:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    phis, gs = [], []
    for k, L in enumerate(sorted(files)):
        d = json.load(open(files[L]))
        r = [x for x in d["runs"] if x["drive_name"] == "lw"][0]
        u = np.array([x["u"] for x in r["rows"]])
        Ju = np.array([x["J_over_u"] for x in r["rows"]])
        ax[0].plot(u, Ju, "-o", ms=3, color=C[k], label=f"Lc = {L:g} (φ = {d['phi']:.3f})")
        m = u <= 0.1
        A = np.vstack([np.ones(m.sum()), u[m] ** 2, u[m] ** 4]).T
        g = np.linalg.lstsq(A, Ju[m], rcond=None)[0][1]
        phis.append(d["phi"])
        gs.append(g)
    ax[0].axhline(0, color=INK, lw=0.7)
    ax[0].set_xlabel("slope u of the tilted lattice")
    ax[0].set_ylabel("J(u)/u at the threshold drive [κ/r0²]")
    ax[0].set_title("(a) homogenized cell at the long-wave threshold", loc="left")
    ax[0].legend(fontsize=7)
    o = np.argsort(phis)
    phis, gs = np.array(phis)[o], np.array(gs)[o]
    ax[1].plot(phis, gs, "-o", color=INK)
    ax[1].axhline(0, color=GRAY, lw=0.8)
    i = np.where(np.diff(np.sign(gs)) != 0)[0]
    res = {}
    if len(i):
        i = i[-1]
        ps = phis[i] - gs[i] * (phis[i + 1] - phis[i]) / (gs[i + 1] - gs[i])
        ax[1].axvline(ps, color=C[1], ls="--", lw=1)
        ax[1].text(ps, ax[1].get_ylim()[1] * 0.8, f"  φ* ≈ {ps:.3f}", color=C[1])
        res["phi_star"] = float(ps)
        res["Lc_star"] = float(np.sqrt(np.pi / ps))
    ax[1].set_xlabel("area fraction φ")
    ax[1].set_ylabel("cubic coefficient g (fit u ≤ 0.1)")
    ax[1].set_title("(b) tricritical density: g < 0 snap, g > 0 terraces", loc="left")
    fig.tight_layout()
    save(fig, "r5_tricritical.png")
    res["g"] = dict(zip([float(x) for x in phis], [float(x) for x in gs]))
    summary["tricritical"] = res


if __name__ == "__main__":
    for f in (fig_flutter, fig_slope, fig_proteins_dyn, fig_peclet, fig_fold, fig_tricritical):
        try:
            f()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{f.__name__}: skipped ({e})")
    json.dump(summary, open(os.path.join(HERE, "results", "round5_summary.json"), "w"), indent=1, default=float)
