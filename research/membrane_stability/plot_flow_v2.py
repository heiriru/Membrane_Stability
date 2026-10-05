'''
Figures of the second flow study (physical PI conditions, plate model, domain / friction dependence, bifurcation
diagram, adjoint sensitivity) and of the ring (h, alpha) stability diagram and PRE Fig. 5 in physical units.

Reads results/flow2/<run>/ (flow) and results/ring_R*_tan*/ (ring); missing runs are skipped.
    python3 plot_flow_v2.py
'''
import csv
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.tri as mtri
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
F2 = os.path.join(RES, "flow2")
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
# categorical slots (validated palette, light mode): identity is also carried by markers and legends
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8985"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False})
summary = {}


def load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def plate(name, bc=None):
    bc = bc or name.split("_")[-1]
    return load_json(os.path.join(F2, name, f"plate_model_{bc}.json"))


def crit(name):
    return load_json(os.path.join(F2, name, "critical.json"))


def plate_curve(name):
    d = plate(name)
    if d is None:
        return None, None
    return np.array([t["Gamma"] for t in d["thresholds"]]), np.array([t["SL_c"] for t in d["thresholds"]])


def plate_at(name, gamma):
    g, s = plate_curve(name)
    if g is None:
        return None
    return float(np.interp(gamma, g, s))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------------------------------------------------------
# 1. threshold vs tension: PI conditions, plate model vs full FE, IRENE's condition for comparison
def fig_tension():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rows = {}
    for k, (name, lab, ls) in enumerate([("plate_C_rigid", "plate model, rigid force-free PI", "-"),
                                         ("plate_C_clamped", "plate model, clamped PI", "--")]):
        g, s = plate_curve(name)
        if g is not None:
            ax.plot(g, s, ls, color=C[k], label=lab)
            rows[name] = dict(zip(map(float, g), map(float, s)))
    full = []
    for sig in ("0", "0.001", "0.005", "0.01"):
        c = crit(f"crit_C_sigma{sig}_free")
        if c:
            full.append((c["Gamma"], c["SL_c"]))
    for name in ("crit_C_free", "crit_B_free", "crit_A_free_verify"):
        c = crit(name)
        if c:
            full.append((c["Gamma"], c["SL_c"]))
    if full:
        full = np.array(sorted(full))
        ax.plot(full[:, 0], full[:, 1], "o", color=C[0], ms=7, mec="white", mew=1.5, label="full FE, force-free PI")
    fullc = [crit(n) for n in ("crit_C_clamped", "crit_B_clamped", "crit_A_clamped_verify")]
    fullc = [(c["Gamma"], c["SL_c"]) for c in fullc if c]
    if fullc:
        fullc = np.array(fullc)
        ax.plot(fullc[:, 0], fullc[:, 1], "s", color=C[1], ms=7, mec="white", mew=1.5, label="full FE, clamped PI")
    old = load_json(os.path.join(RES, "flow_pre_SLc_vs_tension.json"))
    old_pts = []
    for sig in ("0", "0.001", "0.005", "0.01"):
        c = load_json(os.path.join(RES, f"flow_pre_t0_meshA_sigma{sig}", "critical.json"))
        if c:
            old_pts.append((c["sigma_r"] * c["L"] ** 2 / c["kappa"], c["SL_c"]))
    c = load_json(os.path.join(RES, "flow_pre_t0_meshA", "critical.json"))
    if c:
        old_pts.append((25.0, c["SL_c"]))
    if old_pts:
        old_pts = np.array(sorted(old_pts))
        ax.plot(old_pts[:, 0], old_pts[:, 1], "^:", color=GRAY, ms=6, lw=1,
                label="IRENE's square_b PI (no force balance)")
    ax.set_xlabel(r"$\Gamma = \sigma_0 L^2/\kappa$")
    ax.set_ylabel(r"$SL_c = \eta v_{0,c} L/\kappa$")
    ax.set_title("Buckling threshold vs tension (L = 100 r0)", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(loc="lower right")
    save(fig, "flow2_threshold_vs_tension.png")
    summary["tension"] = dict(plate=rows, full_free=np.asarray(full).tolist() if len(full) else [],
                              full_clamped=np.asarray(fullc).tolist() if len(fullc) else [],
                              irene_bc=np.asarray(old_pts).tolist() if len(old_pts) else [])


# ---------------------------------------------------------------------------------------------------------------------
# 2. domain: box size L, PI radius, width W, outer boundary
def fig_domain():
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
    out = {}
    # (a) L at fixed Gamma = 25 and at fixed sigma0 = 0.0025
    Ls = [50.0, 100.0, 200.0]
    pl = [plate_at(n, 25.0) for n in ("plate_L50_rigid", "plate_A_rigid", "plate_L200_rigid")]
    ok = [(L, s) for L, s in zip(Ls, pl) if s is not None]
    if ok:
        ax[0].plot(*zip(*ok), "-", color=C[0], label=r"plate, fixed $\Gamma$ = 25")
    fullG = [(L, crit(n)["SL_c"]) for L, n in zip(Ls, ("crit_L50_G25_free", "crit_A_free_verify", "crit_L200_G25_free"))
             if crit(n)]
    if fullG:
        ax[0].plot(*zip(*fullG), "o", color=C[0], ms=7, mec="white", mew=1.5, label=r"full FE, fixed $\Gamma$ = 25")
    pls = [plate_at(n, g) for n, g in (("plate_L50_rigid", 6.25), ("plate_A_rigid", 25.0), ("plate_L200_rigid", 100.0))]
    oks = [(L, s) for L, s in zip(Ls, pls) if s is not None]
    if oks:
        ax[0].plot(*zip(*oks), "--", color=C[1], label=r"plate, fixed $\sigma_0$ (Γ = 6.25, 25, 100)")
    fulls = [(L, crit(n)["SL_c"]) for L, n in zip(Ls, ("crit_L50_free", "crit_A_free_verify", "crit_L200_free")) if crit(n)]
    if fulls:
        ax[0].plot(*zip(*fulls), "s", color=C[1], ms=7, mec="white", mew=1.5, label=r"full FE, fixed $\sigma_0$")
    ax[0].set_xscale("log")
    ax[0].set_xticks(Ls, ["50", "100", "200"])
    ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[0].set_xlabel("box size L / r0")
    ax[0].set_ylabel(r"$SL_c$")
    ax[0].set_title("(a) box size", loc="left")
    ax[0].legend(fontsize=7)
    out["L"] = dict(plate_fixed_Gamma=ok, full_fixed_Gamma=fullG, plate_fixed_sigma=oks, full_fixed_sigma=fulls)
    # (b) PI radius at Gamma = 25
    rs = [0.5, 1.0, 2.0]
    pr = [(r, plate_at(n, 25.0)) for r, n in zip(rs, ("plate_r0.5_rigid", "plate_A_rigid", "plate_r2_rigid"))]
    pr = [p for p in pr if p[1] is not None]
    if pr:
        ax[1].plot(*zip(*pr), "-", color=C[0], label="plate model")
    fr = [(r, crit(n)["SL_c"]) for r, n in zip(rs, ("crit_r0.5_free", "crit_A_free_verify", "crit_r2_free")) if crit(n)]
    if fr:
        ax[1].plot(*zip(*fr), "o", color=C[0], ms=7, mec="white", mew=1.5, label="full FE")
    ax[1].set_xscale("log")
    ax[1].set_xticks(rs, ["0.5", "1", "2"])
    ax[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax[1].set_xlabel("PI radius (L = 100)")
    ax[1].set_title("(b) PI radius, Γ = 25", loc="left")
    ax[1].legend()
    out["r0"] = dict(plate=pr, full=fr)
    # (c) width and outer boundary condition
    cases = [("W = 50", "plate_W50_rigid", "crit_W50_free"), ("W = 100", "plate_A_rigid", "crit_A_free_verify"),
             ("W = 200", "plate_W200_rigid", "crit_W200_free"),
             ("free walls", "plate_A_zouter1_rigid", "crit_A_zouter1_free"),
             ("free walls\n+ outflow", "plate_A_zouter2_rigid", "crit_A_zouter2_free")]
    rows = []
    for k, (lab, pn, cn) in enumerate(cases):
        p, c = plate_at(pn, 25.0), crit(cn)
        if p is not None:
            ax[2].plot(k, p, "_", color=C[0], ms=18, mew=2.5, label="plate model" if k == 0 else None)
        if c:
            ax[2].plot(k, c["SL_c"], "o", color=C[0], ms=7, mec="white", mew=1.5, label="full FE" if k == 0 else None)
        rows.append((lab.replace("\n", " "), p, c["SL_c"] if c else None))
    ax[2].set_xticks(range(len(cases)), [c[0] for c in cases], fontsize=7)
    ax[2].set_title("(c) width W and outer edges, L = 100, Γ = 25", loc="left")
    ax[2].legend()
    out["outer"] = rows
    for a in ax:
        a.set_ylim(bottom=0)
    save(fig, "flow2_domain.png")
    summary["domain"] = out


# ---------------------------------------------------------------------------------------------------------------------
# 3. in-plane friction (screening length ell_b)
def fig_friction():
    ells = [2, 5, 10, 20, 50]
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    out = {}
    for k, (key, L, pn0, cn0) in enumerate([("A", 100, "plate_A_rigid", "crit_A_free_verify"),
                                            ("L200", 200, "plate_L200_rigid", "crit_L200_free")]):
        # v0_c at the physical tension sigma0 = 0.0025 (Gamma = 0.0025 L^2): interpolate the Gamma list
        pv = []
        for e in ells:
            d = plate(f"plate_{key}_ell{e}_rigid")
            if d:
                g = np.array([t["Gamma"] for t in d["thresholds"]])
                v = np.array([t["v0_c"] for t in d["thresholds"]])
                pv.append((e, float(np.interp(0.0025 * L ** 2, g, v))))
        if pv:
            ax.plot(*zip(*pv), "-", color=C[k], label=f"plate model (divergence), L = {L}")
        fv, osc = [], []
        for e in ells:
            c = crit(f"crit_{key}_ell{e}_free")
            if not c:
                continue
            fv.append((e, c["v0_c"]))
            rows = list(csv.DictReader(open(os.path.join(F2, f"crit_{key}_ell{e}_free", "critical_search.csv"))))
            above = min([r for r in rows if float(r["lead_re"]) > 0], key=lambda r: float(r["v0"]))
            osc.append(float(above["lead_im"]) > 1e-3 * abs(float(above["lead_re"])) + 1e-12)
        for (e, v), o in zip(fv, osc):
            ax.plot(e, v, "D" if o else "os"[k], color=C[k], ms=7, mfc="white" if o else C[k], mec=C[k] if o else "white",
                    mew=1.5)
        if fv:
            ax.plot([], [], "os"[k], color=C[k], ms=7, mec="white", mew=1.5, label=f"full FE, L = {L}: divergence")
            if any(osc):
                ax.plot([], [], "D", color=C[k], mfc="white", ms=7, mew=1.5, label=f"full FE, L = {L}: flutter (Hopf)")
        c0 = crit(cn0) if cn0 != "crit_L200_free" else crit("crit_L200_free")
        if c0:
            ax.axhline(c0["v0_c"], color=C[k], ls=":", lw=1.2, label=f"no friction, L = {L}")
        out[f"L{L}"] = dict(plate=pv, full=fv, oscillatory=osc, no_friction=c0["v0_c"] if c0 else None)
    ax.set_xscale("log")
    ax.set_xticks(ells, [str(e) for e in ells])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel(r"screening length $\ell_b = \sqrt{\eta/b}$ / r0")
    ax.set_ylabel(r"$v_{0,c}$  [$\kappa/(\eta r_0)$]")
    ax.set_title(r"Threshold with in-plane friction ($\sigma_0 = 0.0025\,\kappa/r_0^2$)", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7)
    save(fig, "flow2_friction.png")
    summary["friction"] = out


# ---------------------------------------------------------------------------------------------------------------------
# 4. plate model: energy balance at the threshold, flow-induced stress, critical mode
def fig_plate():
    d = plate("plate_C_rigid")
    if d is None:
        return
    fig = plt.figure(figsize=(12, 3.8))
    a0 = fig.add_subplot(1, 3, 1)
    g = np.array([t["Gamma"] for t in d["thresholds"]])
    parts = {k: np.array([t["energy_parts"][k] / t["energy_parts"]["bending"] for t in d["thresholds"]])
             for k in ("tension", "flow_sigma", "flow_viscous")}
    a0.axhline(1.0, color=GRAY, lw=1)
    a0.text(g[-1], 1.03, "bending = 1", ha="right", va="bottom", fontsize=7, color=GRAY)
    for k, (key, lab) in enumerate([("tension", r"tension $\sigma_0|\nabla z|^2$"),
                                    ("flow_sigma", r"flow: tension drop $\sigma_1$"),
                                    ("flow_viscous", r"flow: viscous stress $2\eta d$")]):
        a0.plot(g, parts[key], "-o", color=C[k], ms=4, label=lab)
    a0.axhline(0, color="k", lw=0.8)
    a0.set_xlabel(r"$\Gamma$")
    a0.set_ylabel("energy / bending energy at threshold")
    a0.set_title("(a) energy balance of the critical mode", loc="left")
    a0.legend(fontsize=7)
    T1 = np.load(os.path.join(F2, "plate_C_rigid", "plate_T1_rigid.npz")) if os.path.exists(
        os.path.join(F2, "plate_C_rigid", "plate_T1_rigid.npz")) else None
    L = d["L"]
    zoom = 15.0
    if T1 is not None:
        a1 = fig.add_subplot(1, 3, 2)
        tri = mtri.Triangulation(T1["x"] - L / 2, T1["y"] - L / 2, T1["triangles"])
        v = T1["sigma1"]
        lim = np.percentile(np.abs(v[(np.abs(T1["x"] - L / 2) < zoom) & (np.abs(T1["y"] - L / 2) < zoom)]), 98)
        cs = a1.tripcolor(tri, v, shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
        a1.add_patch(plt.Circle((0, 0), d["r"], color="k"))
        a1.set_xlim(-zoom, zoom)
        a1.set_ylim(-zoom, zoom)
        a1.set_aspect("equal")
        a1.grid(False)
        fig.colorbar(cs, ax=a1, shrink=0.85, label=r"$\sigma_1 = \partial\sigma/\partial v_0$")
        a1.set_title("(b) flow-induced tension (flow to the right)", loc="left")
    mp = os.path.join(F2, "plate_C_rigid", "plate_mode_rigid_G25.npz")
    if os.path.exists(mp):
        m = np.load(mp)
        a2 = fig.add_subplot(1, 3, 3)
        x, y, z = m["x"] - L / 2, m["y"] - L / 2, m["z"]
        z = z / z[np.argmax(np.abs(z))]
        sel = (np.abs(x) < 2.5 * zoom) & (np.abs(y) < 2.5 * zoom)
        tri = mtri.Triangulation(x[sel], y[sel])
        cs = a2.tricontourf(tri, z[sel], levels=np.linspace(-1, 1, 21), cmap="RdBu_r")
        a2.add_patch(plt.Circle((0, 0), d["r"], color="k"))
        a2.set_xlim(-2 * zoom, 2 * zoom)
        a2.set_ylim(-2 * zoom, 2 * zoom)
        a2.set_aspect("equal")
        a2.grid(False)
        fig.colorbar(cs, ax=a2, shrink=0.85, label="critical mode z' (max = 1)")
        a2.set_title(f"(c) critical mode, Γ = 25, SL_c = {float(m['v0_c']) * L:.1f}", loc="left")
    fig.tight_layout()
    save(fig, "flow2_plate_model.png")
    summary["plate_energy"] = dict(Gamma=g.tolist(), **{k: v.tolist() for k, v in parts.items()},
                                   azimuthal=[t["azimuthal_spectrum"][:5] for t in d["thresholds"]])


# ---------------------------------------------------------------------------------------------------------------------
# 5. bifurcation diagram (amplitude-controlled continuation; subcritical pitchfork, Koiter imperfection sensitivity)
def read_rows(p):
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in r:
            if k != "kind":
                r[k] = float(r[k])
    return rows


def fig_branches():
    amp = {}
    for p in glob.glob(os.path.join(F2, "amp_A_t*", "branches.csv")):
        rows = [r for r in read_rows(p) if r["kind"] == "amplitude"]
        if rows:
            amp[rows[0]["t"]] = sorted(rows, key=lambda r: r["A"])
    nat = os.path.join(F2, "branch_A_clamped", "branches.csv")
    nat = [r for r in read_rows(nat) if r["kind"] == "imperfect"] if os.path.exists(nat) else []
    if not amp and not nat:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ts = sorted(set(amp) | {r["t"] for r in nat}, key=abs)
    folds = []
    for k, t in enumerate(ts):
        col = C[k % len(C)]
        lab = f"t = {t:g}" + (" (perfect)" if t == 0 else "")
        if t in amp:
            rr = amp[t]
            x = np.array([r["v0_over_v0c"] for r in rr])
            A = np.array([r["A"] for r in rr])
            st = np.array([r["lead_re"] < 0 for r in rr])
            ax[0].plot(x, A, "-", color=col, lw=1.2, label=lab)
            ax[0].plot(x[st], A[st], "o", color=col, ms=4)
            ax[0].plot(x[~st], A[~st], "o", color=col, mfc="white", ms=4)
            if t != 0 and np.any(np.diff(x) < 0):
                i = int(np.argmax(x))
                folds.append((abs(t), 1 - x[i], A[i]))
        nn = sorted([r for r in nat if r["t"] == t and r["A"] > 0], key=lambda r: r["v0"])
        if nn and t not in amp:
            ax[0].plot([r["v0_over_v0c"] for r in nn], [r["A"] for r in nn], ":", color=col, lw=1.2, label=lab)
    ax[0].plot([0, 1.0], [0, 0], "-", color="k", lw=1.5)
    ax[0].plot([1.0, 1.15], [0, 0], "--", color="k", lw=1.2)
    ax[0].plot([], [], "o", color=GRAY, ms=4, label="stable")
    ax[0].plot([], [], "o", color=GRAY, mfc="white", ms=4, label="unstable")
    ax[0].axvline(1.0, color=GRAY, lw=0.8)
    ax[0].set_xlim(0.5, 1.15)
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlabel(r"$v_0 / v_{0,c}$")
    ax[0].set_ylabel(r"amplitude $A$ (projection on the critical mode, $\approx$ max $z$ / r0)")
    ax[0].set_title("(a) steady branches, clamped PI (mesh A)", loc="left")
    ax[0].legend(fontsize=7, loc="upper right")
    out = dict(folds=folds)
    if 0.0 in amp:
        rr = amp[0.0]
        A = np.array([r["A"] for r in rr])
        x = np.array([r["v0_over_v0c"] for r in rr])
        small = A <= 1.0
        if small.sum() >= 3:
            c2 = np.polyfit(A[small] ** 2, 1 - x[small], 1)[0]
            out["t0_coefficient_c2"] = float(c2)   # v0/v0_c = 1 - c2 A^2
    if len(folds) >= 2:
        f = np.array(sorted(folds))
        ax[1].loglog(f[:, 0], f[:, 1], "o", color=C[0], ms=7, mec="white", mew=1.5, label="fold of the branch")
        fit = np.polyfit(np.log(f[:, 0]), np.log(f[:, 1]), 1)
        tt = np.logspace(np.log10(f[:, 0].min()), np.log10(f[:, 0].max()), 50)
        ax[1].loglog(tt, np.exp(np.polyval(fit, np.log(tt))), "-", color=C[0], lw=1.2, label=f"fit, slope {fit[0]:.2f}")
        ax[1].loglog(tt, f[-1, 1] * (tt / f[-1, 0]) ** (2 / 3), ":", color=GRAY, lw=1.2, label="Koiter, slope 2/3")
        ax[1].set_xlabel("|t| (contact slope of the PI)")
        ax[1].set_ylabel(r"$1 - v_{0,\mathrm{fold}} / v_{0,c}$")
        ax[1].set_title("(b) imperfection sensitivity", loc="left")
        ax[1].legend()
        out["fold_slope"] = float(fit[0])
    fig.tight_layout()
    save(fig, "flow2_bifurcation.png")
    summary["bifurcation"] = out


# ---------------------------------------------------------------------------------------------------------------------
# 6. adjoint sensitivity
def fig_sensitivity():
    p = os.path.join(F2, "sens_A_clamped", "sensitivity.npz")
    if not os.path.exists(p):
        return
    d = np.load(p)
    js = load_json(os.path.join(F2, "sens_A_clamped", "sensitivity.json"))
    L = 100.0
    x, y = d["x"] - L / 2, d["y"] - L / 2
    tri = mtri.Triangulation(x, y, d["triangles"])
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    zoom = 25.0
    xz = d["x_z"] / np.abs(d["x_z"]).max()
    yz = d["y_z"] / np.abs(d["y_z"]).max() * np.sign(np.dot(d["x_z"], d["y_z"]))
    for a, f, title, cmap, sym in [(ax[0], xz, "(a) direct mode z'", "RdBu_r", True),
                                   (ax[1], d["wavemaker"] / d["wavemaker"].max(),
                                    "(b) wavemaker |z'| |z'+|", "viridis", False)]:
        cs = a.tripcolor(tri, f, shading="gouraud", cmap=cmap, vmin=-1 if sym else 0, vmax=1)
        fig.colorbar(cs, ax=a, shrink=0.8)
        a.set_title(title, loc="left")
    # threshold shift per unit in-plane point force: d v0_c = Phi . e,  Phi = -S_v / (d lambda / d v0)
    Phi = -np.stack([d["S_vx"], d["S_vy"]]) / float(d["dlam_dv0"])
    mag = np.hypot(*Phi)
    cs = ax[2].tripcolor(tri, mag, shading="gouraud", cmap="viridis")
    fig.colorbar(cs, ax=ax[2], shrink=0.8, label=r"$|\partial v_{0,c}/\partial f|$ per unit point force")
    g = np.linspace(-zoom, zoom, 17)
    X, Y = np.meshgrid(g, g)
    interp = [mtri.LinearTriInterpolator(tri, c) for c in Phi]
    U, V = [np.ma.filled(i(X, Y), np.nan) for i in interp]
    ax[2].quiver(X, Y, U, V, color="white", scale=None, width=0.004)
    ax[2].set_title("(c) force direction that raises the threshold", loc="left")
    for a in ax:
        a.add_patch(plt.Circle((0, 0), 1.0, color="k"))
        a.set_xlim(-zoom, zoom)
        a.set_ylim(-zoom, zoom)
        a.set_aspect("equal")
        a.grid(False)
    if js:
        fig.suptitle(f"Adjoint sensitivity at the threshold (clamped PI, mesh A, v0 = {js['v0']}); adjoint check: "
                     f"predicted vs direct d lambda rel. error {js.get('check_rel_error', float('nan')):.1e}",
                     fontsize=9, x=0.01, ha="left")
        summary["sensitivity"] = js
    fig.tight_layout()
    save(fig, "flow2_sensitivity.png")


# ---------------------------------------------------------------------------------------------------------------------
# 7. ring: (h, alpha) stability diagram and PRE Fig. 5 in physical units
def ring_runs():
    runs = {}
    for p in glob.glob(os.path.join(RES, "ring_R*_tan*", "force_displacement.csv")):
        name = os.path.basename(os.path.dirname(p))
        if "finemesh" in name:
            continue
        R = float(name.split("_")[1][1:])
        tan = float(name.split("_tan")[1])
        rows = list(csv.DictReader(open(p)))
        data = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
        runs[(R, tan)] = data
    return runs


def force_extrema(h, F):
    '''heights where dF/dh changes sign (force-control stability boundary); up / down branch'''
    dF = np.gradient(F, h)
    up = [h[i] for i in range(len(h) - 1) if h[i] >= 0 and dF[i] < 0 <= dF[i + 1]]
    down = [h[i + 1] for i in range(len(h) - 1) if h[i + 1] <= 0 and dF[i] >= 0 > dF[i + 1]]
    return (min(up) if up else None), (max(down) if down else None)


def fig_ring():
    runs = ring_runs()
    if not runs:
        return
    Rs = sorted({k[0] for k in runs})
    fig, ax = plt.subplots(1, len(Rs), figsize=(3.6 * len(Rs), 3.8), sharey=False)
    ax = np.atleast_1d(ax)
    table = []
    for a, R in zip(ax, Rs):
        tans = sorted(t for (r, t) in runs if r == R)
        for t in tans:
            d = runs[(R, t)]
            h = d["h"]
            lam = d["lambda_max"]
            up, down = force_extrema(h, d["F_virtual"])
            a.plot([t, t], [h.min(), h.max()], "-", color="#e4e3df", lw=7, solid_capstyle="butt")
            lo = down if down is not None else h.min()
            hi = up if up is not None else h.max()
            a.plot([t, t], [lo, hi], "-", color="#9c9a93", lw=7, solid_capstyle="butt")
            if up is not None:
                a.plot(t, up, "v", color=C[1], ms=7)
            if down is not None:
                a.plot(t, down, "^", color=C[0], ms=7)
            h0 = float(np.interp(0.0, d["F_virtual"][::-1], h[::-1])) if d["F_virtual"][0] > 0 > d["F_virtual"][-1] \
                else None
            if h0 is not None:
                a.plot(t, h0, "o", color="k", ms=4)
            ms = sorted({int(m) for m in d["m_max"]})
            table.append(dict(R=R, tan_alpha=t, h_min=float(h.min()), h_max=float(h.max()), h_F0=h0,
                              h_force_extremum_up=up, h_force_extremum_down=down,
                              max_lambda_displacement=float(lam.max()), leading_m=ms,
                              lambda_m1_max=float(np.nanmax(d["lambda_m1"])) if "lambda_m1" in d else None))
        a.set_xlabel(r"$\tan\alpha$")
        a.set_xlim(-0.25, 2.25)
        a.set_title(f"R = {R:g} r0", loc="left")
    ax[0].set_ylabel("PI height h / r0")
    ax[0].plot([], [], "-", color="#9c9a93", lw=7, label="stable under displacement and force control")
    ax[0].plot([], [], "-", color="#e4e3df", lw=7, label="stable under displacement control only")
    ax[0].plot([], [], "v", color=C[1], label="force extremum (up): force control unstable above")
    ax[0].plot([], [], "^", color=C[0], label="force extremum (down): unstable below")
    ax[0].plot([], [], "o", color="k", ms=4, label="zero force (free PI)")
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1), fontsize=7)
    fig.tight_layout()
    save(fig, "ring_stability_diagram.png")
    summary["ring_diagram"] = table
    # PRE Fig. 5 in physical units: r0 = 10 nm, kappa = 10 kT (T = 300 K), sigma0 = 0.0025 kappa / r0^2 = 1e-6 N/m
    kT = 1.380649e-23 * 300
    kappa, r0 = 10 * kT, 10e-9
    F_unit = kappa / r0 * 1e12   # pN
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for k, R in enumerate(sorted(R for (R, t) in runs if t == 0.5)):
        d = runs[(R, 0.5)]
        ax.plot(d["h"] * r0 * 1e9, d["F_virtual"] * F_unit, "-", color=C[k], label=f"R = {R:g} r0 = {R * 10:g} nm")
    f_tether = 2 * np.pi * np.sqrt(2 * kappa * 0.0025 * kappa / r0 ** 2) * 1e12
    for s in (1, -1):
        ax.axhline(s * f_tether, color=GRAY, ls=":", lw=1.2)
    ax.text(ax.get_xlim()[0], f_tether, r" tether force $2\pi\sqrt{2\kappa\sigma}$", va="bottom", fontsize=7,
            color=GRAY)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("PI displacement h [nm]")
    ax.set_ylabel("force on the PI  F = -dE/dh  [pN]")
    ax.set_title(r"Force-displacement, tan $\alpha$ = 0.5 ($\kappa$ = 10 kT, $r_0$ = 10 nm, $\sigma$ = 1e-6 N/m)",
                 loc="left", fontsize=8)
    ax.legend(fontsize=7)
    save(fig, "ring_force_physical_units.png")
    summary["ring_units"] = dict(kappa_J=kappa, r0_m=r0, force_unit_pN=F_unit, tether_force_pN=f_tether)


for f in (fig_tension, fig_domain, fig_friction, fig_plate, fig_branches, fig_sensitivity, fig_ring):
    try:
        f()
    except Exception as e:  # keep going with the other figures while runs are incomplete
        print(f"{f.__name__}: skipped ({type(e).__name__}: {e})")
json.dump(summary, open(os.path.join(RES, "flow2_summary.json"), "w"), indent=1, default=float)
print("wrote results/flow2_summary.json")
