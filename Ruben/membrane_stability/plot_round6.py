'''
Figures of the sixth study (results/round6, collected by collect_round6.sh):
    r6_slope2d.png     2D slope equation of the lattice: anisotropic coefficients, terraces from noise
    r6_codim2.png      codimension-two point of static buckling and travelling protein instability: neutral curves,
                       unfolding and normal-form coefficients
    r6_wake.png        protein source at the PI as an imperfection of the snap (fold shift vs injection rate)
    r6_tube.png        the fold followed beyond the overhang with remeshing (parametric surface)
    python3 plot_round6.py
'''
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.join(HERE, "results", "round6")
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


# ------------------------------------------------------------------------------------------------- 2D slope equation
def fig_slope2d():
    cf = os.path.join(R6, "slope2d", "coefficients_2d.json")
    if not os.path.exists(cf):
        return
    co = json.load(open(cf))
    tr = {r["Lc"]: r for r in json.load(open(os.path.join(R6, "lattice_2d", "transverse.json")))}
    runs = [(n, os.path.join(R6, "slope2d", f"periodic2d_{n}.npz")) for n in ("dense_eps0.05", "dense_eps0.2",
                                                                            "dilute_eps0.05")]
    runs = [(n, p) for n, p in runs if os.path.exists(p)]
    fig = plt.figure(figsize=(13, 7.6))
    gs = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.45)
    # a: sigma(theta) at rest and at threshold
    ax = fig.add_subplot(gs[0, 0])
    for k, (Lc, lab) in enumerate(((6.0, "φ = 8.7 %"), (8.0, "φ = 4.9 %"))):
        if Lc not in tr:
            continue
        for dn, ls in (("rest", ":"), ("lw", "-")):
            d = tr[Lc]["drives"][dn]["directions"]
            ax.plot([x["angle"] for x in d], [x["sigma"] for x in d], ls, marker="o", ms=3, color=C[k],
                    label=f"{lab}, {'rest' if dn == 'rest' else 'threshold'}")
    ax.set_xlabel("direction θ of the undulation [deg]\n(0: along the flow)")
    ax.set_ylabel("effective tension σ(θ) [κ/r0²]")
    ax.set_title("a  long-wave tension of the lattice", loc="left", fontsize=9)
    ax.legend(fontsize=7)
    # b: cell fluxes vs tilt direction at threshold (J_y / u_y)
    ax = fig.add_subplot(gs[0, 1])
    for k, Lc in enumerate((6.0, 8.0)):
        for j, ang in enumerate((30, 45, 60, 90)):
            p = os.path.join(R6, "lattice_2d", f"nonlinear_cell_L{Lc:g}_a{ang}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            r = [x for x in d["runs"] if x["drive_name"] == "lw"][0]["rows"]
            r = [x for x in r if x["u"] <= 0.2]
            ax.plot([x["ux"] for x in r], [x["Jy"] / x["uy"] for x in r], "-o", ms=2.5, color=C[k], alpha=0.4 + 0.15 * j,
                    label=f"Lc = {Lc:g}" if ang == 90 else None)
    ax.set_xlabel("slope along the flow u_x")
    ax.set_ylabel("transverse stiffness J_y / u_y")
    ax.set_title("b  tilted cells at the threshold drive\n(30°, 45°, 60°, 90°; light → dark)", loc="left", fontsize=9)
    ax.legend(fontsize=7)
    # c: terrace transverse stiffness
    ax = fig.add_subplot(gs[0, 2])
    for k, (Lc, c) in enumerate(sorted(co.items())):
        eps = [float(e) for e in c["terrace"]]
        ax.plot(eps, [c["terrace"][e]["D_yy"] for e in c["terrace"]], "-o", color=C[k], label=f"Lc = {float(Lc):g}")
    ax.axhline(0, color=GRAY, lw=0.8)
    ax.set_xlabel("distance from threshold ε")
    ax.set_ylabel("D_yy on a terrace (> 0: no zigzag)")
    ax.set_title("c  transverse stability of terraces", loc="left", fontsize=9)
    ax.legend(fontsize=7)
    summary["slope2d_coefficients"] = co
    # d: statistics vs time
    ax = fig.add_subplot(gs[0, 3])
    for k, (n, p) in enumerate(runs):
        r = np.load(p)
        st = r["stats"]
        ax.semilogx(st[1:, 0], st[1:, 3], color=C[k], label=n.replace("_", " "))
        summary[f"slope2d_{n}"] = dict(t_end=float(st[-1, 0]), mean_abs_ux=float(st[-1, 1]),
                                       mean_abs_uy=float(st[-1, 2]), walls_per_line=float(st[-1, 3]),
                                       max_ux=float(st[-1, 4]), max_uy=float(st[-1, 5]))
    ax.set_xlabel("t [ζ r0⁴/κ]")
    ax.set_ylabel("terrace walls per line along x")
    ax.set_title("d  coarsening", loc="left", fontsize=9)
    ax.legend(fontsize=7)
    # e-h: height maps
    for k, (n, p) in enumerate(runs[:3]):
        r = np.load(p)
        ax = fig.add_subplot(gs[1, k])
        idx = [len(r["t"]) // 10, -1][1]
        H = r["H"][idx]
        x = r["x"]
        im = ax.imshow(H.T, origin="lower", extent=[x[0], x[-1], x[0], x[-1]], cmap="RdBu_r",
                       vmin=-np.abs(H).max(), vmax=np.abs(H).max())
        ax.set_title(f"{'efg'[k]}  {n.replace('_', ' ')}, t = {r['t'][idx]:.2g}", loc="left", fontsize=9)
        ax.set_xlabel("x [r0] (flow →)")
        if k == 0:
            ax.set_ylabel("y [r0]")
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.046).set_label("H [r0]", fontsize=7)
    lp = os.path.join(R6, "slope2d", "periodic2d_dense_eps0.2_local.npz")
    ax = fig.add_subplot(gs[1, 3])
    if os.path.exists(lp):
        r = np.load(lp)
        ax.plot(r["t"], r["y_front"], "o", ms=3, color=C[0], label="lateral front (envelope 10⁻⁴)")
        tt = np.array([0, r["t"].max()])
        ax.plot(tt, r["y_front"][0] + float(r["v_y_linear"]) * tt, "--", color=C[1],
                label=f"linear spreading 2√(σ_yy λ_max) = {float(r['v_y_linear']):.3f}")
        summary["slope2d_front"] = dict(v_measured=float(r["v_y_measured"]), v_linear=float(r["v_y_linear"]))
        ax.set_xlabel("t [ζ r0⁴/κ]")
        ax.set_ylabel("half-width across the flow [r0]")
        ax.set_title(f"h  a terrace patch invades sideways\n(measured {float(r['v_y_measured']):.3f} r0 per unit time)",
                     loc="left", fontsize=9)
        ax.legend(fontsize=6.5, loc="lower right")
    save(fig, "r6_slope2d.png")


# ------------------------------------------------------------------------------------------------------- codim-2
def _crit(path, key):
    if not os.path.exists(path):
        return None
    return json.load(open(path)).get(key)


def fig_codim2():
    gp = os.path.join(R6, "c2_M10_grid", "codim2_grid.json")
    if not os.path.exists(gp):
        return
    grid = json.load(open(gp))["grid"]
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0))
    ax = axs[0]
    kinds = {"stable": (GRAY, "o"), "static": (C[0], "s"), "oscillatory": (C[1], "D")}
    seen = set()
    for g in grid:
        lam = g["eigenvalues"][0]
        kind = "stable" if lam[0] < 0 else ("static" if abs(lam[1]) < 1e-7 else "oscillatory")
        col, mk = kinds[kind]
        ax.plot(g["chi"], g["v0"] * 100, mk, color=col, ms=6, alpha=0.6, label=None if kind in seen else
                {"stable": "flat state stable", "static": "static buckling (real λ > 0)",
                 "oscillatory": "travelling mode (complex λ, Re > 0)"}[kind])
        seen.add(kind)
    crit = {}
    # static onsets (pitchfork coefficient a, normalized max z = 1) and Hopf onsets (Re d)
    for chi in ("0.1", "0.075"):
        p = os.path.join(R6, f"imp_M10_chi{chi}_wnl", "imperfection.json")
        if os.path.exists(p):
            d = json.load(open(p))
            crit[f"static_chi{chi}"] = dict(SL=d["SL_c"], a=d["a"])
            ax.plot(float(chi), d["SL_c"], "s", ms=11, mfc="none", mec=C[0], mew=2)
            ax.annotate(f"a = {d['a']:+.1e}\n(subcritical)" if d["a"] > 0 else f"a = {d['a']:+.1e}",
                        (float(chi), d["SL_c"]), xytext=(4, -22), textcoords="offset points", fontsize=7, color=C[0])
    for chi, sl in (("0", 17.83), ("0.05", 19.0)):
        p = os.path.join(R6, f"c2_hopf_chi{chi}", "codim2_normal_form.json")
        if os.path.exists(p):
            d = json.load(open(p))
            red = d["coefficients"]["1.0"]["d"][0]
            crit[f"hopf_chi{chi}"] = dict(SL=d["SL"], Re_d=red, omega=d["omega"])
            ax.plot(float(chi), d["SL"], "D", ms=11, mfc="none", mec=C[1], mew=2)
            ax.annotate(f"Re d = {red:+.1e}\n(subcritical)" if red > 0 else f"Re d = {red:+.1e}",
                        (float(chi), d["SL"]), xytext=(-60, 8), textcoords="offset points", fontsize=7, color=C[1])
    for p in sorted(glob.glob(os.path.join(R6, "c2_M10_nf", "codim2_bt_*.json"))):
        d = json.load(open(p))
        cf = d["coefficients"]["1.0"]
        crit[os.path.basename(p)] = dict(chi=d["chi"], SL=d["SL"], a=cf.get("a_eff", cf["a"]), b=cf.get("b_eff", cf["b"]))
        ax.plot(d["chi"], d["SL"], "*", ms=13, color=INK)
    ax.plot([], [], "*", ms=11, color=INK, label="near the merger (Bogdanov–Takens): b < 0, a ≈ 0")
    summary["codim2_criticality"] = crit
    summary["codim2_grid"] = grid
    ax.set_xlabel("χ (protein interaction)")
    ax.set_ylabel("SL = v0 L")
    ax.set_title("a  M = 10, clamped PI: the two instabilities and\ntheir weakly nonlinear coefficients",
                 loc="left", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower left")
    ax = axs[1]
    for k, (name, lab) in enumerate((("c2dyn_static", "χ = 0.10, SL = 16 (static sector)"),
                                     ("c2dyn_hopf", "χ = 0.05, SL = 20 (oscillatory sector)"))):
        rr = rows(os.path.join(R6, name, "time_series.csv"))
        if not rr:
            continue
        t = np.array([r["t"] for r in rr])
        zm = np.array([max(abs(r["z_min"]), abs(r["z_max"])) for r in rr])
        ax.semilogy(t / 1e6, np.maximum(zm, 1e-8), color=C[k], label=lab)
        summary[name] = dict(t_end=float(t[-1]), z_final=float(zm[-1]),
                             phi_final=float(max(abs(rr[-1]["phi_min"]), abs(rr[-1]["phi_max"]))))
    ax.axhline(0.14, color=GRAY, ls=":", lw=1)
    ax.text(0.02, 0.17, "small-amplitude saturation if supercritical", fontsize=7, color=GRAY)
    ax.set_xlabel("t [10⁶ ζ r0⁴/κ]")
    ax.set_ylabel("max |z| [r0]")
    ax.set_title("b  time stepping: exponential growth, then runaway", loc="left", fontsize=9)
    ax.legend(fontsize=7)
    ax = axs[2]
    sp = os.path.join(R6, "c2dyn_static", "snapshots.npz")
    if os.path.exists(sp):
        import matplotlib.tri as mtri
        d = np.load(sp)
        tr = mtri.Triangulation(d["x"], d["y"], d["triangles"])
        z, ph = d["z"][-1], d["phi"][-1]
        zl = np.abs(z).max()
        im = ax.tripcolor(tr, z, cmap="RdBu_r", vmin=-zl, vmax=zl, shading="gouraud")
        ax.tricontour(tr, ph, levels=[-0.1, -0.03, 0.03, 0.1], colors=[INK], linewidths=0.6)
        plt.colorbar(im, ax=ax, fraction=0.046, label="z [r0]")
        ax.set_xlim(20, 80)
        ax.set_ylim(20, 80)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_title(f"c  static sector at t = {d['t'][-1] / 1e6:.2f}·10⁶: deep pit,\nproteins (contours of φ) collected in it",
                     loc="left", fontsize=9)
        ax.set_xlabel("x [r0] (flow →)")
        ax.set_ylabel("y [r0]")
    fig.tight_layout()
    save(fig, "r6_codim2.png")


# ---------------------------------------------------------------------------------------------------------- wake
def fig_wake():
    p = os.path.join(R6, "imp_M1", "imperfection.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p))
    pb = os.path.join(R6, "imp_M1_b", "imperfection.json")
    if os.path.exists(pb):
        d["runs"] = d["runs"] + [r for r in json.load(open(pb))["runs"] if r.get("rows")]
    summary["wake"] = {k: v for k, v in d.items() if k != "runs"}
    summary["wake"]["folds"] = [{k: v for k, v in r.items() if k != "rows"} for r in d["runs"]]
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axs[0]
    for k, r in enumerate(d["runs"]):
        rr = r["rows"]
        ax.plot([x["v0"] / d["v0_c"] for x in rr], [x["z_max"] for x in rr], color=C[k], label=f"Q = {r['Q']:g}")
        ax.plot(r["v_fold"] / d["v0_c"], r["z_at_fold"], "o", color=C[k])
    ax.axvline(1, color=GRAY, lw=0.8)
    ax.set_xlabel("v0 / v0_c")
    ax.set_ylabel("largest deflection z [r0]")
    ax.set_title("a  steady states with a protein source at the PI\n(dot: fold = snap)", loc="left", fontsize=9)
    ax.legend(fontsize=7)
    ax = axs[1]
    Q = np.array([r["Q"] for r in d["runs"]])
    sh = np.array([r["one_minus_vfold"] for r in d["runs"]])
    pr = [r["predicted_one_minus_vfold"] for r in d["runs"]]
    for r in d["runs"]:
        ax.annotate(f"{r['one_minus_vfold']:.3f}", (r["Q"], r["one_minus_vfold"]), xytext=(5, -10),
                    textcoords="offset points", fontsize=7)
    if len(Q):
        ax.loglog(Q, sh, "o", color=C[0], label="continuation")
        if all(x is not None for x in pr):
            qq = np.geomspace(Q.min() / 2, Q.max() * 2, 50)
            k = pr[0] / Q[0] ** (2 / 3)
            ax.loglog(qq, k * qq ** (2 / 3), "-", color=C[1], label="weakly nonlinear (Koiter, ∝ Q^(2/3))")
    from matplotlib.ticker import ScalarFormatter, NullFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("injection rate Q")
    ax.set_ylabel("1 − v_fold / v0_c")
    ax.set_title("b  the snap moves earlier, it is not smoothed", loc="left", fontsize=9)
    ax.legend(fontsize=7)
    save(fig, "r6_wake.png")


# ---------------------------------------------------------------------------------------------------------- tube
def fig_tube():
    segs = sorted(glob.glob(os.path.join(R6, "tube", "seg*")), key=lambda s: int(s.split("seg")[-1]))
    base = rows(os.path.join(HERE, "results", "round5", "param_eq", "time_series.csv"))
    if not segs:
        return
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    ts = [(base, "before remeshing (round 5)")] + [(rows(os.path.join(s, "time_series.csv")), os.path.basename(s))
                                                   for s in segs]
    t0 = base[0]["t"] if base else 0.0
    for k, (rr, lab) in enumerate(ts):
        if not rr:
            continue
        t = np.array([r["t"] for r in rr]) - t0
        axs[0].plot(t / 1e3, [r["z_min"] for r in rr], color=C[k % len(C)], label=lab)
        axs[0].plot(t / 1e3, [r["h"] for r in rr], "--", color=C[k % len(C)])
        axs[1].plot(t / 1e3, [r["min_nz"] for r in rr], color=C[k % len(C)])
        axs[2].semilogy(t / 1e3, [r["stretch_max"] for r in rr], color=C[k % len(C)])
    axs[0].set_ylabel("z_min (solid), PI height h (dashed) [r0]")
    axs[1].set_ylabel("min n_z (< 0: overhang)")
    axs[2].set_ylabel("largest area stretch of the parametrization")
    for ax, tl in zip(axs, ("a  depth and PI height", "b  overhang", "c  parametrization (reset by remeshing)")):
        ax.set_xlabel("t − t0 [10³ ζ r0⁴/κ]")
        ax.set_title(tl, loc="left", fontsize=9)
    axs[0].legend(fontsize=6)
    save(fig, "r6_tube.png")


if __name__ == "__main__":
    for f in (fig_slope2d, fig_codim2, fig_wake):
        f()
    json.dump(summary, open(os.path.join(HERE, "results", "round6_summary.json"), "w"), indent=1, default=str)
