'''
Figures of the third study (results/round3, collected by collect_round3.sh):
    r3_landau.png        weakly nonlinear theory vs continuation (perfect branch, Koiter fold law)
    r3_snap.png          time stepping beyond the fold: collapse into a deep pit
    r3_forcefree.png     force-free PI with contact angle: steady branches vs clamped and vs IRENE's square_b
    r3_flutter_map.png   divergence / flutter in (ell_b, Gamma); local absolute instability
    r3_lattice.png       lattice of PIs: threshold vs density, Bloch band, selected pattern
    r3_multi.png         several PIs in the box: thresholds, collective modes
    r3_wrinkles.png      sliding membrane (wrinkle trains) and strongly driven box
    r3_tube.png          ring beyond the force maximum (axisymmetric tube), force tables
    r3_mobile.png        mobile curvature-coupled proteins with flow
Missing inputs: that figure is skipped. Also writes results/round3_summary.json.
    python3 plot_round3.py
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
RES = os.path.join(HERE, "results")
R3 = os.path.join(RES, "round3")
F2 = os.path.join(RES, "flow2")
FIG = os.path.join(HERE, "figures")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8985"
INK = "#2b2a27"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False})
summary = {}
kT = 1.380649e-23 * 300
KAPPA, R0 = 10 * kT, 10e-9
F_UNIT = KAPPA / R0 * 1e12            # pN
V_UNIT = KAPPA / (1e-8 * R0) * 1e6    # um/s for eta = 1e-8 Pa m s


def jl(p):
    return json.load(open(p)) if os.path.exists(p) else None


def rows(p, numeric=True):
    if not os.path.exists(p):
        return []
    out = list(csv.DictReader(open(p)))
    if numeric:
        for r in out:
            for k, v in r.items():
                try:
                    r[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return out


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def top_view(ax, x, y, tri, z, half, title=None, holes=(), r=1.0, cmap="RdBu_r", lim=None):
    t = mtri.Triangulation(x, y, tri)
    lim = lim or float(np.abs(z).max()) or 1.0
    pc = ax.tripcolor(t, z, shading="gouraud", cmap=cmap, vmin=-lim, vmax=lim)
    for hx, hy in holes:
        ax.add_patch(plt.Circle((hx, hy), r, color=INK))
    ax.set_aspect("equal")
    ax.grid(False)
    if half is not None:
        cx, cy, hw = half
        ax.set_xlim(cx - hw, cx + hw)
        ax.set_ylim(cy - hw, cy + hw)
    if title:
        ax.set_title(title, loc="left", fontsize=8)
    return pc


# ---------------------------------------------------------------------------------------------------------------------
def fig_landau():
    lan = jl(os.path.join(R3, "landau_A_clamped", "landau.json"))
    if lan is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    br = [r for r in rows(os.path.join(F2, "gif_amp_t0", "branches.csv")) if r["kind"] == "amplitude"]
    br += [r for r in rows(os.path.join(R3, "post_amp_t0", "branches.csv")) if r["kind"] == "amplitude"]
    br = sorted({round(r["A"], 6): r for r in br}.values(), key=lambda r: r["A"])
    A = np.array([r["A"] for r in br])
    x = np.array([r["v0_over_v0c"] for r in br])
    st = np.array([r["lead_re"] < 0 for r in br])
    c = lan["coef_perfect_branch"]
    ax[0].plot(x, A, "o", color=C[0], mfc="white", ms=4, label="continuation (all unstable)")
    AA = np.linspace(0, A.max(), 200)
    ax[0].plot(1 - c * AA ** 2, AA, "-", color=C[1], lw=1.5,
               label=fr"Landau: $v_0/v_{{0,c}} = 1 - {c:.5f}\,A^2$")
    ax[0].axvline(1, color=GRAY, lw=0.8)
    ax[0].set_xlabel(r"$v_0 / v_{0,c}$")
    ax[0].set_ylabel("amplitude A")
    ax[0].set_title("(a) perfect PI (t = 0): subcritical pitchfork", loc="left")
    ax[0].legend(fontsize=7)
    ax[0].set_xlim(0.45, 1.05)
    # folds: clamped continuation (flow2 amp_A_t*), Landau prediction, force-free branches
    folds = []
    for p in glob.glob(os.path.join(F2, "amp_A_t-*", "branches.csv")):
        rr = [r for r in rows(p) if r["kind"] == "amplitude"]
        if rr:
            xs = np.array([r["v0_over_v0c"] for r in rr])
            folds.append((abs(rr[0]["t"]), 1 - xs.max()))
    folds.sort()
    f = np.array(folds)
    tt = np.logspace(-2.2, -0.4, 50)
    pref = lan["koiter_prefactor"]
    ax[1].loglog(f[:, 0], f[:, 1], "o", color=C[0], ms=7, mec="white", mew=1.5, label="clamped PI: continuation")
    ax[1].loglog(tt, pref * tt ** (2 / 3), "-", color=C[1], lw=1.5,
                 label=fr"theory: ${pref:.3f}\,|t|^{{2/3}}$ (no fit)")
    ff = []
    for p in glob.glob(os.path.join(R3, "ff_t*", "forcefree_branch.csv")):
        rr = rows(p)
        vs = np.array([r["v0_over_v0c"] for r in rr])
        i = int(np.argmax(vs)) if len(vs) else 0
        # the fold: the branch turned back (interior maximum of v0), or it became vertical (last steps change v0 by
        # less than 2e-4: the continuation in v0 creeps along the fold)
        if 0 < i and (vs[-1] < vs[i] or np.ptp(vs[-5:]) < 2e-4):
            t = float(p.split("ff_t")[-1].split("/")[0])
            ff.append((abs(t), 1 - vs[i]))
    if ff:
        ff = np.array(sorted(ff))
        ax[1].loglog(ff[:, 0], ff[:, 1], "s", color=C[2], ms=7, mec="white", mew=1.5, label="force-free PI")
        if len(ff) >= 2:
            fit = np.polyfit(np.log(ff[:, 0]), np.log(ff[:, 1]), 1)
            summary["forcefree_fold_slope"] = float(fit[0])
    ax[1].set_xlabel("|t| (contact slope of the PI)")
    ax[1].set_ylabel(r"$1 - v_{0,\mathrm{fold}} / v_{0,c}$")
    ax[1].set_title("(b) imperfection sensitivity (Koiter)", loc="left")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "r3_landau.png")
    summary["landau"] = dict(coef=c, N3_parts=lan["N3_parts"], koiter_prefactor=pref,
                             folds_clamped=folds, fold_predictions=lan["fold_predictions"],
                             folds_forcefree=ff.tolist() if len(ff) else [])


# ---------------------------------------------------------------------------------------------------------------------
def fig_snap():
    ts = rows(os.path.join(R3, "snap_t-0.3_v0.92", "time_series.csv"))
    if not ts:
        return
    fig = plt.figure(figsize=(12, 7))
    ax0 = fig.add_subplot(2, 3, 1)
    ax1 = fig.add_subplot(2, 3, 2)
    for k, f in enumerate(("0.92", "0.97")):
        tsk = rows(os.path.join(R3, f"snap_t-0.3_v{f}", "time_series.csv"))
        if not tsk:
            continue
        t = np.array([r["t"] for r in tsk])
        ax0.plot(t / 1e5, [r["max_abs_z"] for r in tsk], "-", color=C[k], label=f"v0 = {f} v0_c")
        ax1.semilogy(t / 1e5, [max(r["max_slope"], 1e-3) for r in tsk], "-", color=C[k], label=f"v0 = {f} v0_c")
    ax0.set_xlabel(r"time [$10^5\,\zeta r_0^4/\kappa$]")
    ax0.set_ylabel("max |z| / r0")
    ax0.set_title("(a) depth of the pit", loc="left")
    ax0.legend(fontsize=7)
    ax1.set_xlabel(r"time [$10^5\,\zeta r_0^4/\kappa$]")
    ax1.set_ylabel("max |grad z| (wall slope)")
    ax1.axhline(1, color=GRAY, lw=0.8)
    ax1.set_title("(b) steepest wall (Monge breaks down)", loc="left")
    d = np.load(os.path.join(R3, "snap_t-0.3_v0.92", "snapshots.npz"))
    x, y = d["x"] - 50, d["y"] - 50
    tri = mtri.Triangulation(x, y, d["triangles"])
    xc, yc = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
    tri.set_mask(np.hypot(xc, yc) < 1)
    sel = [0, len(d["t"]) // 3, 2 * len(d["t"]) // 3, len(d["t"]) - 1]
    lim = np.abs(d["z"][-1]).max()
    ax = fig.add_subplot(2, 3, 3)
    xs = np.linspace(-45, 45, 400)
    for k, i in enumerate(sel):
        zl = mtri.LinearTriInterpolator(tri, d["z"][i])(xs, 0 * xs)
        ax.plot(xs, zl, "-", color=C[k], lw=1.5, label=f"t = {d['t'][i]:.3g}")
    ax.add_patch(plt.Rectangle((-1, -0.5), 2, 1, color=INK))
    ax.set_xlabel("x / r0 (flow to the right)")
    ax.set_ylabel("z / r0 on the centreline")
    ax.set_title("(c) centreline profiles (PI at x = 0)", loc="left")
    ax.legend(fontsize=7)
    for k, i in enumerate(sel[1:]):
        a = fig.add_subplot(2, 3, 4 + k)
        pc = a.tripcolor(tri, d["z"][i], shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
        a.add_patch(plt.Circle((0, 0), 1, color=INK))
        a.set_xlim(-40, 40)
        a.set_ylim(-40, 40)
        a.set_aspect("equal")
        a.grid(False)
        a.set_title(f"({'def'[k]}) z at t = {d['t'][i]:.3g}", loc="left")
    fig.colorbar(pc, ax=fig.axes[-3:], shrink=0.8, label="z / r0")
    save(fig, "r3_snap.png")
    last = ts[-1]
    summary["snap"] = dict(v092=dict(t_end=last["t"], max_abs_z=last["max_abs_z"], max_slope=last["max_slope"]),
                           v097=jl(os.path.join(R3, "snap_t-0.3_v0.97", "snap.json"))["final"])


# ---------------------------------------------------------------------------------------------------------------------
def fig_forcefree():
    runs = {p.split("ff_t")[-1].split("/")[0]: rows(os.path.join(p, "forcefree_branch.csv"), True)
            for p in glob.glob(os.path.join(R3, "ff_t*"))}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3))
    for k, t in enumerate(sorted(runs, key=lambda s: abs(float(s)))):
        rr = runs[t]
        ax[0].plot([r["v0_over_v0c"] for r in rr], [r["h"] for r in rr], "-o", ms=3, color=C[k], lw=1.3,
                   label=f"force-free PI, t = {t}")
        ax[1].plot([r["SL"] for r in rr], [r["max_abs_z"] for r in rr], "-o", ms=3, color=C[k], lw=1.3,
                   label=f"force-free PI, t = {t}")
    # clamped PI (height 0) branch and IRENE's square_b (no force balance), t = -0.3
    cl = [r for r in rows(os.path.join(F2, "gif_amp_t-0.3_dense", "branches.csv")) if r["kind"] == "amplitude"]
    if cl:
        ax[1].plot([r["SL"] for r in cl], [r["max_abs_z"] for r in cl], "--", color=GRAY, lw=1.3,
                   label="clamped PI (h = 0), t = -0.3")
    ir = rows(os.path.join(RES, "flow_pre_t-0.3_meshA", "critical_search.csv"))
    if ir:
        ir = sorted(ir, key=lambda r: r["v0"])
        ax[1].plot([100 * r["v0"] for r in ir], [r["max_abs_z"] for r in ir], ":", color=INK, lw=1.5,
                   label="IRENE square_b (no force balance), t = -0.3")
    ax[0].set_xlabel(r"$v_0 / v_{0,c}$ (force-free $v_{0,c}$ = 0.2750)")
    ax[0].set_ylabel("PI height h / r0")
    ax[0].set_title("(a) height of the force-free PI along the branch", loc="left")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel(r"$SL = \eta v_0 L / \kappa$")
    ax[1].set_ylabel("max |z| / r0")
    ax[1].set_title("(b) deformation vs flow: PI conditions compared", loc="left")
    ax[1].legend(fontsize=7)
    ax[1].set_ylim(0, 12)
    fig.tight_layout()
    save(fig, "r3_forcefree.png")
    folds = {}
    for t, rr in runs.items():
        vs = np.array([r["v0_over_v0c"] for r in rr])
        i = int(np.argmax(vs))
        folds[t] = dict(fold_found=bool(0 < i and (vs[-1] < vs[i] or np.ptp(vs[-5:]) < 2e-4)),
                        turned_back=bool(vs[-1] < vs[i]), v_fold=float(vs[i]),
                        h_fold=rr[i]["h"], A_fold=rr[i]["A"], h_rest=rr[0]["h"], n=len(rr))
    summary["forcefree"] = folds


# ---------------------------------------------------------------------------------------------------------------------
def fig_flutter_map():
    rr = []
    for p in sorted(glob.glob(os.path.join(R3, "fmap_plate_g*", "flutter_map.json"))):
        rr += jl(p) or []
    if not rr:
        return
    loc = jl(os.path.join(R3, "local", "local.json"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    gammas = sorted({r["Gamma"] for r in rr})
    codim2 = []
    for k, g in enumerate(gammas):
        s = sorted([r for r in rr if r["Gamma"] == g], key=lambda r: r["ell_b"])
        ell = np.array([r["ell_b"] for r in s])
        sl = np.array([r["SL_c"] for r in s])
        fl = np.array([r["kind"] == "flutter" for r in s])
        ax[0].semilogx(ell, sl, "-", color=C[k], lw=1.3, label=f"Γ = {g:g}")
        ax[0].semilogx(ell[~fl], sl[~fl], "o", color=C[k], ms=5)
        ax[0].semilogx(ell[fl], sl[fl], "s", color=C[k], ms=5, mfc="white")
        for i in range(len(ell) - 1):
            if fl[i] != fl[i + 1]:
                codim2.append((g, float(np.sqrt(ell[i] * ell[i + 1]))))
    ax[0].plot([], [], "o", color=GRAY, label="divergence")
    ax[0].plot([], [], "s", color=GRAY, mfc="white", label="flutter")
    for p in glob.glob(os.path.join(R3, "fmap_G*", "critical.json")) + \
            glob.glob(os.path.join(F2, "crit_A_ell*_free", "critical.json")):
        d = jl(p)
        if d and d.get("ell_b"):
            ax[0].semilogx(d["ell_b"], d["SL_c"], "x", color=INK, ms=7)
    ax[0].plot([], [], "x", color=INK, label="IRENE full equations")
    ax[0].set_xlabel(r"screening length $\ell_b = \sqrt{\eta/b}$ / r0")
    ax[0].set_ylabel(r"threshold $SL_c$ (first instability)")
    ax[0].set_title("(a) first instability with in-plane friction (plate model)", loc="left")
    ax[0].legend(fontsize=6.5, ncol=2)
    # regime diagram
    for r in rr:
        mk = "s" if r["kind"] == "flutter" else "o"
        col = C[1] if r["kind"] == "flutter" else C[0]
        ax[1].semilogx(r["ell_b"], r["Gamma"], mk, color=col, ms=6)
    if codim2:
        cc = np.array(codim2)
        ax[1].semilogx(cc[:, 1], cc[:, 0], "|", color=INK, ms=14, mew=1.5)
    ax[1].plot([], [], "o", color=C[0], label="divergence first")
    ax[1].plot([], [], "s", color=C[1], label="flutter first")
    ax[1].plot([], [], "|", color=INK, ms=10, mew=1.5, label="change of type (codimension two)")
    ax[1].set_xlabel(r"$\ell_b$ / r0")
    ax[1].set_ylabel(r"$\Gamma = \sigma_0 L^2/\kappa$")
    ax[1].set_title("(b) first instability", loc="left")
    ax[1].legend(fontsize=7)
    if loc:
        G = np.array([r["G"] for r in loc["table"]])
        ax[2].plot(G, [r["sigma_abs_saddle"] for r in loc["table"]], "-", color=C[0], label="saddle point")
        ax[2].plot(G, [r["sigma_abs_numeric"] for r in loc["table"]], "o", color=C[1], ms=4, label="impulse response")
        ax[2].axhline(0, color=INK, lw=0.8)
        ax[2].axvline(loc["G_star"], color=GRAY, ls="--", lw=1)
        ax[2].text(loc["G_star"], 0.2, f" G* = {loc['G_star']:.3f}", color=GRAY, fontsize=8)
        ax[2].set_xlabel(r"drift parameter $G = g\sqrt{\kappa}/s^{3/2}$")
        ax[2].set_ylabel(r"absolute growth rate [$s^2/(\kappa\zeta)$]")
        ax[2].set_title("(c) local analysis: absolute if G < G*", loc="left")
        ax[2].legend(fontsize=7)
        summary["local"] = dict(G_star=loc["G_star"], box_cases=loc.get("box_cases"))
    fig.tight_layout()
    save(fig, "r3_flutter_map.png")
    summary["flutter_map"] = dict(rows=rr, codim2=codim2)


# ---------------------------------------------------------------------------------------------------------------------
def fig_lattice():
    lat = {}
    for p in glob.glob(os.path.join(R3, "lat_*", "lattice.json")):
        lat[os.path.basename(os.path.dirname(p))] = jl(p)
    if not lat:
        return
    fig = plt.figure(figsize=(14, 8))
    a0 = fig.add_subplot(2, 3, 1)
    a1 = fig.add_subplot(2, 3, 2)
    a2 = fig.add_subplot(2, 3, 3)
    names = {"lat_force": r"uniform force, $\sigma_0$ = 1e-6 N/m", "lat_fric5": r"friction $\ell_b$ = 5",
             "lat_fric20": r"friction $\ell_b$ = 20", "lat_force_s1e-5": r"uniform force, $\sigma_0$ = 1e-5 N/m",
             "lat_force_s1e-7": r"uniform force, $\sigma_0$ = 1e-7 N/m"}
    for k, (name, rr) in enumerate(sorted(lat.items())):
        phi = np.array([r["phi"] for r in rr])
        a0.loglog(phi, [r["U_c"] for r in rr], "-o", color=C[k], ms=4, lw=1.3, label=names.get(name, name))
        a1.semilogx(phi, [r["drag_c"] for r in rr], "-o", color=C[k], ms=4, lw=1.3, label=names.get(name, name))
        a2.semilogx(phi, [np.hypot(*r["q_star_over_zone"]) for r in rr], "-o", color=C[k], ms=4, lw=1.3,
                    label=names.get(name, name))
    a0.axhline(0.2750, color=GRAY, ls=":", lw=1)
    a0.text(a0.get_xlim()[0], 0.2750, " single PI in the PRE box (L = 100)", color=GRAY, fontsize=7, va="bottom")
    a0.set_xlabel(r"area fraction of PIs $\phi = \pi r_0^2 / L_c^2$")
    a0.set_ylabel(r"mean membrane velocity at threshold $U_c$ [$\kappa/(\eta r_0)$]")
    a0.set_title("(a) threshold vs protein density", loc="left")
    a0.legend(fontsize=6.5)
    a1.set_xlabel(r"$\phi$")
    a1.set_ylabel(r"drag per PI at threshold [$\kappa / r_0$]")
    a1.set_title("(b) anchoring force per PI at threshold", loc="left")
    a2.set_xlabel(r"$\phi$")
    a2.set_ylabel(r"$|q^*| L_c / \pi$ (0: in phase, 1: zone edge)")
    a2.set_title("(c) selected Bloch wavevector", loc="left")
    # band structure and pattern for one cell
    for col, name in enumerate(("lat_force",)):
        mp = sorted(glob.glob(os.path.join(R3, name, "mode_L*.npz")), key=lambda p: float(p.split("_L")[-1][:-4]))
        if not mp:
            continue
        d = np.load(mp[len(mp) // 2])
        Lc = float(mp[len(mp) // 2].split("_L")[-1][:-4])
        a3 = fig.add_subplot(2, 3, 4)
        qs = d["qs"] * Lc / np.pi
        band = d["band"].copy()
        band[0, 0] = np.nan       # q = 0: the neutral uniform lift (lambda = 0) is excluded there
        lim = np.nanmax(np.abs(band))
        pc = a3.pcolormesh(qs, qs, band.T, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        a3.plot(*(np.array(d["q"]) * Lc / np.pi), "*", color=INK, ms=10)
        a3.set_xlabel(r"$q_x L_c / \pi$ (along the flow)")
        a3.set_ylabel(r"$q_y L_c / \pi$")
        a3.set_title(f"(d) Re λ(q) at threshold, L_c = {Lc:g}", loc="left")
        fig.colorbar(pc, ax=a3, shrink=0.8)
        # pattern over 4 x 4 cells: z = Re[exp(i q.x) (p_r + i p_i)]
        a4 = fig.add_subplot(2, 3, 5)
        tri = mtri.Triangulation(d["x"], d["y"], d["triangles"])
        q = np.array(d["q"])
        nc = int(min(16, max(4, np.ceil(2 * np.pi / max(np.hypot(*q), 1e-9) / Lc))))
        X, Y = np.meshgrid(np.linspace(0, nc * Lc, 480), np.linspace(0, nc * Lc / 2, 240))
        xm, ym = np.mod(X, Lc), np.mod(Y, Lc)
        pr = mtri.LinearTriInterpolator(tri, d["p_r"])(xm, ym).filled(np.nan)
        pi_ = mtri.LinearTriInterpolator(tri, d["p_i"])(xm, ym).filled(np.nan)
        ph = q[0] * X + q[1] * Y
        Z = np.cos(ph) * pr - np.sin(ph) * pi_
        lim = np.nanmax(np.abs(Z))
        a4.pcolormesh(X, Y, Z, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
        for i in range(nc):
            for j in range(nc // 2):
                a4.add_patch(plt.Circle((Lc * (i + 0.5), Lc * (j + 0.5)), 1.0, color=INK))
        a4.set_aspect("equal")
        a4.grid(False)
        a4.set_title(f"(e) critical pattern over {nc} x {nc // 2} cells (flow →, travels upstream)", loc="left",
                     fontsize=8)
        a5 = fig.add_subplot(2, 3, 6)
        pc = top_view(a5, d["x"], d["y"], d["triangles"], d["p_r"], None, "(f) Bloch amplitude Re p in one cell")
        a5.add_patch(plt.Circle((Lc / 2, Lc / 2), 1.0, color=INK))
    fig.tight_layout()
    save(fig, "r3_lattice.png")
    summary["lattice"] = lat


# ---------------------------------------------------------------------------------------------------------------------
def fig_multi():
    m = jl(os.path.join(R3, "multi", "multi.json"))
    if m is None:
        return
    fig = plt.figure(figsize=(14, 8))
    a0 = fig.add_subplot(2, 3, 1)
    a1 = fig.add_subplot(2, 3, 2)
    for k, kind in enumerate(("tandem", "side")):
        ks = sorted([n for n in m if n.startswith(kind + "_d")], key=lambda n: float(n.split("_d")[1]))
        d = [float(n.split("_d")[1]) for n in ks]
        a0.plot(d, [m[n]["SL_c"] for n in ks], "-o", color=C[k], ms=4,
                label={"tandem": "pair along the flow", "side": "pair across the flow"}[kind])
    a0.axhline(m["single"]["SL_c"], color=GRAY, ls=":", lw=1)
    a0.text(30, m["single"]["SL_c"], "single PI", color=GRAY, fontsize=7, va="bottom", ha="right")
    a0.set_xlabel("centre distance d / r0")
    a0.set_ylabel(r"$SL_c$")
    a0.set_title("(a) two PIs: threshold vs distance", loc="left")
    a0.legend(fontsize=7)
    names = list(m)
    tot = np.array([m[n]["total_drag_per_v0"] * m[n]["SL_c"] / 100 for n in names])
    a1.bar(range(len(names)), tot, color=[C[0] if n.startswith("tandem") else C[1] if n.startswith("side") else
                                          C[2] if n.startswith(("row", "file")) else C[3] for n in names])
    a1.set_xticks(range(len(names)))
    a1.set_xticklabels(names, rotation=90, fontsize=6)
    a1.set_ylabel(r"total drag on the PIs at threshold [$\kappa/r_0$]")
    a1.set_title("(b) total anchoring force at threshold", loc="left")
    a1.axhline(1.0, color=GRAY, lw=0.8)
    for k, n in enumerate(("side_d10", "row5_d10", "cluster8", "tandem_d20")):
        p = os.path.join(R3, "multi", f"mode_{n}_0.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p)
        a = fig.add_subplot(2, 3, 3 + k) if k == 0 else fig.add_subplot(2, 3, 3 + k)
        top_view(a, d["x"], d["y"], d["triangles"], d["z"], (50, 50, 35),
                 f"({'cdef'[k]}) {n}: SL_c = {float(d['SL']):.2f}", holes=[tuple(h) for h in d["holes"]], lim=1.0)
    fig.tight_layout()
    save(fig, "r3_multi.png")
    summary["multi"] = {n: dict(SL_c=v["SL_c"], total_drag_at_threshold=v["total_drag_per_v0"] * v["SL_c"] / 100,
                                PI_heights=v["modes"][0]["PI_heights"]) for n, v in m.items()}


# ---------------------------------------------------------------------------------------------------------------------
def fig_wrinkles():
    w = jl(os.path.join(R3, "wrinkles", "wrinkles.json"))
    if w is None:
        return
    fig = plt.figure(figsize=(14, 7.5))
    sl = w.get("sliding")
    k = 0
    if sl:
        for i, case in enumerate(sl["cases"][-3:]):
            p = os.path.join(R3, "wrinkles", f"slide_f{case['factor']:g}_m0.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            a = fig.add_subplot(2, 3, 1 + i)
            top_view(a, d["x"], d["y"], d["triangles"], d["z_r"], None,
                     f"({'abc'[i]}) sliding membrane, U = {case['factor']:g} U_c: {case['n_unstable']} unstable, "
                     f"λ* local = {case['wavelength_inflow']:.0f} r0")
            a.set_xlabel("x / r0 (flow →; tension falls upstream)")
    bx = w.get("box200")
    if bx:
        for i, case in enumerate(bx["cases"][1:4]):
            p = os.path.join(R3, "wrinkles", f"box200_f{case['factor']:g}_m0.npz")
            if not os.path.exists(p):
                continue
            d = np.load(p)
            a = fig.add_subplot(2, 3, 4 + i)
            top_view(a, d["x"], d["y"], d["triangles"], d["z"], (100, 100, 70),
                     f"({'def'[i]}) box L = 200, v0 = {case['factor']:g} v0_c: {case['n_unstable']} unstable modes",
                     holes=[(100, 100)], lim=1.0)
    fig.tight_layout()
    save(fig, "r3_wrinkles.png")
    summary["wrinkles"] = w


# ---------------------------------------------------------------------------------------------------------------------
def fig_tube():
    runs = {}
    for p in glob.glob(os.path.join(R3, "tube", "tube_R*_tan*")):
        n = os.path.basename(p)
        R = float(n.split("_R")[1].split("_")[0])
        t = float(n.split("_tan")[1])
        runs[(R, t)] = (rows(os.path.join(p, "branch.csv")), jl(os.path.join(p, "info.json")), p)
    if not runs:
        return
    fig = plt.figure(figsize=(14, 8))
    a0 = fig.add_subplot(2, 3, 1)
    a1 = fig.add_subplot(2, 3, 2)
    a2 = fig.add_subplot(2, 3, 3)
    for k, R in enumerate(sorted({r for r, t in runs})):
        if (R, 0.5) not in runs:
            continue
        rr, info, path = runs[(R, 0.5)]
        h = np.array([r["h"] for r in rr])
        F = -np.array([r["F"] for r in rr])
        a0.plot(h / R, F * F_UNIT, "-", color=C[k], lw=1.5, label=f"R = {R:g} r0 = {10 * R:g} nm")
        ov = np.array([r["overhang"] in (True, "True", 1.0) for r in rr])
        a0.plot(h[ov] / R, F[ov] * F_UNIT, ".", color=C[k], ms=2)
        ir = rows(os.path.join(RES, f"ring_R{R:g}_tan0.5", "force_displacement.csv"))
        if ir:
            a0.plot([r["h"] / R for r in ir], [-r["F_virtual"] * F_UNIT for r in ir], "x", color=C[k], ms=3)
    f0 = 2 * np.pi * np.sqrt(2 * 0.0025) * F_UNIT
    a0.axhline(f0, color=INK, ls=":", lw=1)
    a0.text(0, f0, f"tether force {f0:.2f} pN", fontsize=7, va="bottom")
    a0.plot([], [], "x", color=GRAY, label="IRENE (Monge), virtual work")
    a0.plot([], [], ".", color=GRAY, label="overhanging shapes")
    a0.set_xlabel("PI displacement h / R")
    a0.set_ylabel("pulling force [pN]")
    a0.set_ylim(-3 * f0, None)
    a0.set_title(r"(a) force vs displacement, tan $\alpha$ = 0.5", loc="left")
    a0.legend(fontsize=6.5)
    # profiles for R = 30
    for (R, a, lab) in ((30, a1, "(b)"), (100, a2, "(c)")):
        if (R, 0.5) not in runs:
            continue
        pf = np.load(os.path.join(runs[(R, 0.5)][2], "profiles.npz"))
        keys = sorted([kk for kk in pf.files if float(kk) >= 0], key=float)
        pick = keys[::max(1, len(keys) // 7)]
        for j, kk in enumerate(pick):
            psi, u, r, z, gam = pf[kk]
            a.plot(r, z, "-", color=plt.get_cmap("viridis")(j / max(1, len(pick) - 1)), lw=1.2)
            a.plot(-r, z, "-", color=plt.get_cmap("viridis")(j / max(1, len(pick) - 1)), lw=1.2)
        a.set_aspect("equal")
        a.set_xlabel("r / r0")
        a.set_ylabel("z / r0")
        a.set_title(f"{lab} shapes, R = {R} r0 (h = 0 … {float(pick[-1]):.0f})", loc="left")
    tab = jl(os.path.join(R3, "ring_table", "ring_table.json"))
    if tab:
        a3 = fig.add_subplot(2, 3, 4)
        for k, R in enumerate(sorted({r["R"] for r in tab})):
            s = sorted([r for r in tab if r["R"] == R], key=lambda r: r["ell"])
            a3.semilogx([r["ell"] for r in s], [r["F_max_over_f0"] for r in s], "-o", color=C[k], ms=4,
                        label=f"R = {R:g}")
        a3.axhline(1, color=GRAY, lw=0.8)
        a3.set_xlabel(r"$\ell = \sqrt{\kappa/\sigma}$ / r0")
        a3.set_ylabel(r"$F_{max} / f_0$")
        a3.set_title("(d) force overshoot vs tension and ring size", loc="left")
        a3.legend(fontsize=7)
        summary["ring_table"] = tab
    fig.tight_layout()
    save(fig, "r3_tube.png")
    summary["tube"] = {f"R{R:g}_tan{t:g}": v[1] for (R, t), v in runs.items()}


# ---------------------------------------------------------------------------------------------------------------------
def fig_mobile():
    mo = jl(os.path.join(R3, "mobile", "mobile.json"))
    if mo is None:
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for k, r in enumerate(mo):
        p = r["params"]
        U = np.array([s["U"] for s in r["scan"]])[1:]
        lab = f"C0 = {p['C0']}, M = {p['M']}"
        ax[0].semilogx(U, [s["chi_c_along_flow"] for s in r["scan"]][1:], "-o", color=C[k], ms=4, label=lab)
        ax[0].axhline(r["chi_c0_analytic"], color=C[k], ls=":", lw=1)
        ax[1].semilogx(U, [abs(s["band_speed_along"]) / s["U"] for s in r["scan"][1:]], "-o", color=C[k], ms=4,
                       label=lab)
    ax[0].set_xlabel(r"flow speed U [$\kappa/(\zeta r_0^3)$]")
    ax[0].set_ylabel(r"$\chi_c$ for modulations along the flow")
    ax[0].set_title("(a) the flow suppresses modulations along it\n(dotted: χ_c without flow = threshold of stripes along the flow)",
                    loc="left", fontsize=8)
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("U")
    ax[1].set_ylabel("band speed / U")
    ax[1].set_title("(b) speed of travelling bands (if transverse stripes are excluded)", loc="left", fontsize=8)
    fig.tight_layout()
    save(fig, "r3_mobile.png")
    summary["mobile"] = mo


if __name__ == "__main__":
    for f in (fig_landau, fig_snap, fig_forcefree, fig_flutter_map, fig_lattice, fig_multi, fig_wrinkles, fig_tube,
              fig_mobile):
        try:
            f()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{f.__name__}: skipped ({type(e).__name__}: {e})")
    json.dump(summary, open(os.path.join(RES, "round3_summary.json"), "w"), indent=1, default=float)
    print("wrote results/round3_summary.json")
