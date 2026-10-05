'''
Figures of the fourth study (results/round4, collected by collect_round4.sh):
    r4_post_snap_forcefree.png   time stepping beyond the fold with a force-free PI (vs the clamped PI of round 3)
    r4_lattice_nonlinear.png     macroscopic vertical stress J(u) of a tilted lattice: effective tension and the cubic
                                 coefficient of the long-wave amplitude equation
    r4_proteins.png              mobile curvature-coupled proteins in IRENE with the flow around an anchored PI
    python3 plot_round4.py
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
R4 = os.path.join(HERE, "results", "round4")
R3 = os.path.join(HERE, "results", "round3")
FIG = os.path.join(HERE, "figures")
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY, INK = "#8a8985", "#2b2a27"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False})
summary = {}


def rows(p):
    if not os.path.exists(p):
        return []
    out = list(csv.DictReader(open(p)))
    for r in out:
        for k in r:
            try:
                r[k] = float(r[k])
            except ValueError:
                pass
    return out


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def fig_post_snap():
    runs = {f: rows(os.path.join(R4, f"snapff_v{f}", "time_series.csv")) for f in ("0.93", "0.97")}
    runs = {k: v for k, v in runs.items() if v}
    if not runs:
        return
    fig = plt.figure(figsize=(13, 8.2))
    fig.subplots_adjust(hspace=0.42, wspace=0.3)
    a0, a1, a2 = (fig.add_subplot(2, 3, k) for k in (1, 2, 3))
    for k, (f, rr) in enumerate(sorted(runs.items())):
        t = np.array([r["t"] for r in rr])
        t = (t - t[-1]) / 1e3
        a0.plot(t, [r["h"] for r in rr], "-", color=C[k], label=f"force-free PI, v0 = {f} v0_c")
        a1.plot(t, [r["z_min"] for r in rr], "-", color=C[k], label=f"pit depth, v0 = {f} v0_c")
        a2.semilogy(t, [max(r["max_slope"], 1e-3) for r in rr], "-", color=C[k])
    cl = rows(os.path.join(R3, "snap_t-0.3_v0.92", "time_series.csv"))
    if cl:
        t = np.array([r["t"] for r in cl])
        t = (t - t[-1]) / 1e3
        a1.plot(t, [r["z_min"] for r in cl], "--", color=GRAY, label="clamped PI (round 3), 0.92 v0_c")
        a2.semilogy(t, [max(r["max_slope"], 1e-3) for r in cl], "--", color=GRAY)
    for a, lab, ttl in ((a0, "PI height h / r0", "(a) the force-free PI is lifted"),
                        (a1, "z_min / r0", "(b) the pit deepens"),
                        (a2, "max |grad z|", "(c) steepest wall: Monge breakdown")):
        a.set_xlabel(r"time before the end of the run [$10^3\,\zeta r_0^4/\kappa$]")
        a.set_ylabel(lab)
        a.set_title(ttl, loc="left")
        a.set_xlim(-30, 1)
    a0.legend(fontsize=7)
    a1.legend(fontsize=7)
    d = None
    for f in ("0.97", "0.93"):
        p = os.path.join(R4, f"snapff_v{f}", "snapshots.npz")
        if os.path.exists(p):
            d, fsel = np.load(p), f
            break
    if d is not None:
        x, y = d["x"] - 50, d["y"] - 50
        tri = mtri.Triangulation(x, y, d["triangles"])
        xc, yc = x[tri.triangles].mean(1), y[tri.triangles].mean(1)
        tri.set_mask(np.hypot(xc, yc) < 1)
        lim = np.abs(d["z"]).max()
        a3 = fig.add_subplot(2, 3, 4)
        xs = np.linspace(-45, 45, 500)
        sel = [0, len(d["t"]) // 2, len(d["t"]) - 1]
        for k, i in enumerate(sel):
            zl = mtri.LinearTriInterpolator(tri, d["z"][i])(xs, 0 * xs)
            a3.plot(xs, zl, "-", color=C[k], lw=1.5, label=f"t = {d['t'][i]:.4g}")
        a3.set_xlabel("x / r0 (flow →), centreline")
        a3.set_ylabel("z / r0")
        a3.set_title(f"(d) centreline profiles, v0 = {fsel} v0_c", loc="left")
        a3.legend(fontsize=7)
        for k, i in enumerate(sel[1:]):
            a = fig.add_subplot(2, 3, 5 + k)
            pc = a.tripcolor(tri, d["z"][i], shading="gouraud", cmap="RdBu_r", vmin=-lim, vmax=lim)
            a.add_patch(plt.Circle((0, 0), 1, color=INK))
            a.set_xlim(-35, 35)
            a.set_ylim(-35, 35)
            a.set_aspect("equal")
            a.grid(False)
            a.set_title(f"({'ef'[k]}) z at t = {d['t'][i]:.4g}", loc="left")
        fig.colorbar(pc, ax=fig.axes[-2:], shrink=0.8, label="z / r0")
    save(fig, "r4_post_snap_forcefree.png")
    summary["post_snap"] = {f: rr[-1] for f, rr in runs.items()}


def fig_lattice_nonlinear():
    files = sorted(glob.glob(os.path.join(R4, "lattice_nonlinear", "nonlinear_cell_L*.json")),
                   key=lambda p: float(p.split("_L")[-1][:-5]))
    if not files:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.3))
    rows_g = []
    for k, p in enumerate(files):
        d = json.load(open(p))
        for r in d["runs"]:
            us = np.array([x["u"] for x in r["rows"]])
            Ju = np.array([x["J_over_u"] for x in r["rows"]])
            if r["drive_name"] == "lw":
                ax[0].plot(us, Ju, "-o", color=C[k % len(C)], ms=3, label=f"Lc = {d['Lc']:g} (φ = {d['phi']:.3f})")
            if r["drive_name"] == "zero":
                ax[1].plot(us, Ju / Ju[0], "-o", color=C[k % len(C)], ms=3, label=f"Lc = {d['Lc']:g}")
            rows_g.append(dict(Lc=d["Lc"], phi=d["phi"], drive=r["drive_name"], D=r["D"],
                               sigma_eff=r["sigma_eff_fit"], g=r["g_small_slope"], failed=r["failed_at_u"]))
    ax[0].axhline(0, color=INK, lw=0.8)
    ax[0].set_xlabel("slope u of the tilted lattice")
    ax[0].set_ylabel(r"$J(u)/u$ [$\kappa/r_0^2$]")
    ax[0].set_title("(a) at the long-wave threshold drive", loc="left")
    ax[0].legend(fontsize=7)
    ax[1].set_xlabel("u")
    ax[1].set_ylabel(r"$J(u)/(u\,\sigma_{eff})$ at rest")
    ax[1].set_title("(b) geometric softening at rest", loc="left")
    for k, drive in enumerate(("zero", "half", "lw")):
        rr = [r for r in rows_g if r["drive"] == drive]
        ax[2].semilogx([r["phi"] for r in rr], [r["g"] / max(abs(r["sigma_eff"]), 1e-12) if drive != "lw" else r["g"]
                                                 for r in rr], "-o", color=C[k], ms=4,
                       label={"zero": "g / σ_eff, at rest", "half": "g / σ_eff, half the threshold drive",
                              "lw": "g at the threshold (κ/r0²)"}[drive])
    ax[2].axhline(0, color=INK, lw=0.8)
    ax[2].set_xlabel("area fraction φ")
    ax[2].set_ylabel("cubic coefficient")
    ax[2].set_title("(c) J = σ_eff u + g u³: g < 0 subcritical, g > 0 saturating", loc="left", fontsize=8)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "r4_lattice_nonlinear.png")
    summary["lattice_nonlinear"] = rows_g


def fig_proteins():
    a = json.load(open(os.path.join(R4, "phi_chic2", "phi_stability.json"))) if os.path.exists(
        os.path.join(R4, "phi_chic2", "phi_stability.json")) else None
    b = json.load(open(os.path.join(R4, "phi_v0c", "phi_stability.json"))) if os.path.exists(
        os.path.join(R4, "phi_v0c", "phi_stability.json")) else None
    if a is None and b is None:
        return
    st = os.path.join(R4, "phi_v0c_stiff", "phi_stability.json")
    if b is not None and os.path.exists(st):
        b["v0_c"] = b["v0_c"] + json.load(open(st))["v0_c"]
    fig = plt.figure(figsize=(14, 7.5))
    a0 = fig.add_subplot(2, 3, 1)
    a1 = fig.add_subplot(2, 3, 2)
    if a:
        cc = [r for r in a["chi_c"] if r.get("chi_c") is not None]
        bulk = [r for r in cc if r["v0"] < 0.1]
        edge = [r for r in cc if r["v0"] >= 0.1]
        a0.plot([r["SL"] for r in bulk], [r["chi_c"] for r in bulk], "-o", color=C[0], label="IRENE + proteins (box)")
        if edge:
            a0.plot([bulk[-1]["SL"]] + [r["SL"] for r in edge], [bulk[-1]["chi_c"]] + [r["chi_c"] for r in edge], ":",
                    color=C[0])
            a0.plot([r["SL"] for r in edge], [r["chi_c"] for r in edge], "o", mfc="white", color=C[0],
                    label="mode confined to the outflow boundary layer")
        a0.axhline(a["chi_c0_infinite"], color=GRAY, ls=":", lw=1)
        a0.text(0.5, a["chi_c0_infinite"], " infinite flat membrane, no flow", color=GRAY, fontsize=7, va="top")
        a0.set_xlabel(r"$SL = \eta v_0 L/\kappa$")
        a0.set_ylabel(r"protein threshold $\chi_c$")
        a0.set_title("(a) protein demixing threshold vs flow", loc="left")
        a0.legend(fontsize=7)
        summary["chi_c"] = a["chi_c"]
    if b:
        vv = sorted([r for r in b["v0_c"] if r.get("v0_c") is not None], key=lambda r: r["chi"])
        a1.set_xscale("symlog", linthresh=0.1, linscale=0.6)
        a1.plot([r["chi"] for r in vv], [r["SL_c"] for r in vv], "-o", color=C[1], label="IRENE + mobile proteins")
        xe = np.array([0.03, 0.1, 0.3, 1, 3, 10, 30, 100])
        a1.plot(xe, 27.50 * xe / (xe + 4 * b["C"] ** 2), "--", color=C[2], lw=1.2,
                label=r"27.50 $\kappa_{eff}/\kappa$ (proteins in equilibrium)")
        a1.set_xticks([-0.04, 0, 1, 10, 100])
        a1.set_xticklabels(["−0.04", "0", "1", "10", "100"])
        a1.set_ylim(20, 28)
        a1.legend(fontsize=7, loc="lower right")
        a1.axhline(27.50, color=GRAY, ls=":", lw=1)
        a1.text(a1.get_xlim()[0], 27.5, " no proteins", color=GRAY, fontsize=7, va="bottom")
        a1.set_xlabel(r"$\chi$ (protein osmotic stiffness)")
        a1.set_ylabel(r"buckling threshold $SL_c$")
        a1.set_title("(b) mobile proteins lower the buckling threshold", loc="left")
        summary["v0_c"] = b["v0_c"]
    k = 3

    def num(p, key):
        return float(os.path.basename(p)[:-4].split(key)[-1])

    picks = []
    for pat, key, lab in (("phi_chic2/mode_chi_c_v*.npz", "_v", "protein mode at χ_c, v0 = {:g}"),
                          ("phi_v0c/mode_v0c_chi*.npz", "_chi", "buckling mode at v0_c, χ = {:g}")):
        ps = sorted(glob.glob(os.path.join(R4, pat)), key=lambda q: num(q, key))
        if pat.startswith("phi_v0c"):
            ps = ps[::-1]      # stiff proteins (large χ) first, then the softest computed
        if pat.startswith("phi_chic2"):
            # v0 >= 0.1: the leading protein mode sits in the outflow boundary layer; show the last bulk mode
            ps = [q for q in ps if num(q, key) < 0.1]
        for p in ([ps[0], ps[-1]] if len(ps) > 1 else ps):
            picks.append((p, lab.format(num(p, key))))
    for p, lab in picks:
            if k > 6:
                break
            d = np.load(p)
            ax = fig.add_subplot(2, 3, k)
            x, y = d["x"] - 50, d["y"] - 50
            tri = mtri.Triangulation(x, y, d["triangles"])
            ph = d["phi"] / max(np.abs(d["phi"]).max(), 1e-30)
            ax.tripcolor(tri, ph, shading="gouraud", cmap="PuOr_r", vmin=-1, vmax=1)
            zz = d["z"] / max(np.abs(d["z"]).max(), 1e-30)
            ax.tricontour(tri, zz, levels=[-0.6, -0.3, 0.3, 0.6], colors=[INK], linewidths=0.6)
            ax.add_patch(plt.Circle((0, 0), 1, color=INK))
            ax.set_aspect("equal")
            ax.grid(False)
            ax.set_title(f"({'abcdefgh'[k - 1]}) {lab}\nφ (colour, purple–orange), z (lines, dashed < 0)",
                         loc="left", fontsize=8)
            k += 1
    fig.tight_layout()
    save(fig, "r4_proteins.png")


if __name__ == "__main__":
    for f in (fig_post_snap, fig_lattice_nonlinear, fig_proteins):
        try:
            f()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{f.__name__}: skipped ({e})")
    json.dump(summary, open(os.path.join(HERE, "results", "round4_summary.json"), "w"), indent=1, default=float)
