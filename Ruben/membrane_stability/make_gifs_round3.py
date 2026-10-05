'''
Animations of the third study (figures/gifs/):
    gif5_post_snap_collapse.gif  time stepping beyond the fold (t = -0.3, v0 = 0.92 v0_c): the membrane collapses into
                                 a deep pit upstream of the PI (until the Monge description breaks down)
    gif6_lattice_wave.gif        lattice of PIs at the threshold: long-wave undulation travelling upstream
    gif7_tube_extrusion.gif      PI pulled out of a ring-bounded membrane (R = 300 nm): shape with overhangs and force
    python3 make_gifs_round3.py
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
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.join(HERE, "results", "round3")
OUT = os.path.join(HERE, "figures", "gifs")
os.makedirs(OUT, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, GRAY = "#2b2a27", "#8a8985"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "lines.linewidth": 2, "legend.frameon": False})
kT = 1.380649e-23 * 300
F_UNIT = 10 * kT / 10e-9 * 1e12


def save(anim, name, fps):
    path = os.path.join(OUT, name)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(anim._fig)
    print(f"wrote figures/gifs/{name} ({os.path.getsize(path) / 1e6:.1f} MB)")


def gif5():
    d = np.load(os.path.join(R3, "snap_t-0.3_v0.92", "snapshots.npz"))
    ts = list(csv.DictReader(open(os.path.join(R3, "snap_t-0.3_v0.92", "time_series.csv"))))
    x, y = d["x"] - 50, d["y"] - 50
    tri = mtri.Triangulation(x, y, d["triangles"])
    g = np.linspace(-42, 42, 90)
    X, Y = np.meshgrid(g, g)
    zlim = np.abs(d["z"]).max()
    fig = plt.figure(figsize=(11, 4.4))
    a3 = fig.add_subplot(1, 2, 1, projection="3d")
    a1 = fig.add_subplot(1, 2, 2)
    t_all = np.array([float(r["t"]) for r in ts])
    z_all = np.array([float(r["max_abs_z"]) for r in ts])
    a1.plot(t_all / 1e5, z_all, "-", color=C[0])
    dot, = a1.plot([], [], "o", color=C[1], ms=9, mec="white", mew=1.5)
    a1.set_xlabel(r"time [$10^5\,\zeta r_0^4/\kappa$]")
    a1.set_ylabel("depth of the pit, max |z| / r0")
    a1.set_title("(b) slow drift past the fold, then collapse", loc="left")
    interp = mtri.LinearTriInterpolator

    def frame(i):
        Z = interp(tri, d["z"][i])(X, Y).filled(np.nan)
        Z[X ** 2 + Y ** 2 < 1] = np.nan
        a3.clear()
        Zc = np.nan_to_num(Z, nan=0.0)
        cols = plt.get_cmap("RdBu_r")(0.5 + 0.5 * Zc / zlim)
        a3.plot_surface(X, Y, Zc, facecolors=cols, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
        a3.plot_wireframe(X, Y, Zc, rstride=10, cstride=10, color="#b9b8b2", linewidth=0.3)
        a3.set_zlim(-zlim, 0.4 * zlim)
        a3.set_zticks([])
        a3.view_init(elev=22, azim=-70)
        a3.set_xlabel("x / r0 (flow →)", labelpad=-6)
        a3.set_ylabel("y / r0", labelpad=-6)
        a3.tick_params(labelsize=6, pad=-3)
        a3.set_title(f"(a) t = {d['t'][i]:.3g}: pit depth {np.abs(d['z'][i]).max():.1f} r0", fontsize=9)
        dot.set_data([d["t"][i] / 1e5], [np.abs(d["z"][i]).max()])
        return []

    anim = FuncAnimation(fig, frame, frames=len(d["t"]), blit=False)
    fig.suptitle("Beyond the fold (contact slope t = -0.3, v0 = 0.92 v0_c): no nearby steady state, the membrane "
                 "collapses into a deep pit upstream of the PI", fontsize=10)
    fig.subplots_adjust(left=0.02, right=0.97, top=0.85, bottom=0.12, wspace=0.15)
    save(anim, "gif5_post_snap_collapse.gif", fps=3)


def gif6():
    mp = sorted(glob.glob(os.path.join(R3, "lat_force", "mode_L*.npz")), key=lambda p: float(p.split("_L")[-1][:-4]))
    if not mp:
        raise FileNotFoundError("lattice modes")
    p = [m for m in mp if m.endswith("_L10.npz")] or mp[:1]
    d = np.load(p[0])
    Lc = float(p[0].split("_L")[-1][:-4])
    q = np.array(d["q"])
    tri = mtri.Triangulation(d["x"], d["y"], d["triangles"])
    nc = int(min(16, max(6, np.ceil(2 * np.pi / max(np.hypot(*q), 1e-9) / Lc))))
    X, Y = np.meshgrid(np.linspace(0, nc * Lc, 520), np.linspace(0, 4 * Lc, 160))
    xm, ym = np.mod(X, Lc), np.mod(Y, Lc)
    pr = mtri.LinearTriInterpolator(tri, d["p_r"])(xm, ym).filled(np.nan)
    pi_ = mtri.LinearTriInterpolator(tri, d["p_i"])(xm, ym).filled(np.nan)
    ph0 = q[0] * X + q[1] * Y
    lim = np.nanmax(np.hypot(pr, pi_))
    fig, ax = plt.subplots(figsize=(11, 3.6))
    im = ax.pcolormesh(X, Y, np.cos(ph0) * pr - np.sin(ph0) * pi_, shading="auto", cmap="RdBu_r", vmin=-lim, vmax=lim)
    for i in range(nc):
        for j in range(4):
            ax.add_patch(plt.Circle((Lc * (i + 0.5), Lc * (j + 0.5)), 1.0, color=INK))
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x / r0 (flow →)")
    n = 40

    def frame(k):
        ph = ph0 + 2 * np.pi * k / n          # Re[exp(i (q.x + omega t))] p, omega > 0: the wave moves upstream
        im.set_array((np.cos(ph) * pr - np.sin(ph) * pi_).ravel())
        return []

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    ax.set_title(f"Lattice of anchored PIs (cell {Lc:g} r0, area fraction {np.pi / Lc ** 2:.3f}) driven by a uniform "
                 "tangential force: at the threshold the whole lattice\nbuckles in a long wave that travels upstream "
                 "(one period; colour = height)", fontsize=9, loc="left")
    fig.tight_layout()
    save(anim, "gif6_lattice_wave.gif", fps=12)


def gif7():
    base = os.path.join(R3, "tube", "tube_R30_tan0.5")
    pf = np.load(os.path.join(base, "profiles.npz"))
    br = list(csv.DictReader(open(os.path.join(base, "branch.csv"))))
    h = np.array([float(r["h"]) for r in br])
    F = -np.array([float(r["F"]) for r in br]) * F_UNIT
    keys = sorted([k for k in pf.files if float(k) >= 0], key=float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.2]))
    hmax = float(keys[-1])
    ax[1].plot(h[h >= 0] * 10, F[h >= 0], "-", color=C[0])
    f0 = 2 * np.pi * np.sqrt(2 * 0.0025) * F_UNIT
    ax[1].axhline(f0, color=INK, ls=":", lw=1)
    ax[1].text(0, f0, f" tether force {f0:.2f} pN", fontsize=8, va="bottom")
    dot, = ax[1].plot([], [], "o", color=C[1], ms=9, mec="white", mew=1.5)
    ax[1].set_xlabel("PI displacement h [nm]")
    ax[1].set_ylabel("pulling force [pN]")
    ax[1].set_title("(b) force: overshoot, then the tether plateau", loc="left")
    l1, = ax[0].plot([], [], "-", color=C[0])
    l2, = ax[0].plot([], [], "-", color=C[0])
    pi_, = ax[0].plot([], [], "-", color=INK, lw=5, solid_capstyle="butt")
    ax[0].plot([-300, 300], [0, 0], "^", color=C[1], ms=8)
    ax[0].set_xlim(-330, 330)
    ax[0].set_ylim(-60, hmax * 10 + 60)
    ax[0].set_aspect("equal")
    ax[0].set_xlabel("r [nm]")
    ax[0].set_ylabel("z [nm]")

    def frame(i):
        k = keys[i]
        psi, u, r, z, gam = pf[k]
        l1.set_data(10 * r, 10 * z)
        l2.set_data(-10 * r, 10 * z)
        pi_.set_data([-10, 10], [10 * float(k), 10 * float(k)])
        dot.set_data([10 * float(k)], [np.interp(float(k), h, F)])
        ax[0].set_title(f"(a) ring R = 300 nm, h = {10 * float(k):.0f} nm" +
                        ("  (overhang)" if np.abs(psi).max() > np.pi / 2 else ""), loc="left")
        return []

    anim = FuncAnimation(fig, frame, frames=len(keys), blit=False)
    fig.suptitle(r"Pulling a PI beyond the Monge range (axisymmetric shape equations, tan $\alpha$ = 0.5, "
                 r"$\kappa$ = 10 kT, $\sigma$ = 1e-6 N/m): a tube forms", fontsize=10)
    fig.subplots_adjust(top=0.86, bottom=0.12, wspace=0.25)
    save(anim, "gif7_tube_extrusion.gif", fps=4)


if __name__ == "__main__":
    import sys
    for name in sys.argv[1:] or ["gif5", "gif6", "gif7"]:
        try:
            globals()[name]()
        except Exception as e:
            print(f"{name}: skipped ({type(e).__name__}: {e})")
