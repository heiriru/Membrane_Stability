'''
Animated GIFs of the main results (written to figures/gifs/):

    1. gif1_flow_buckling_onset.gif   flow -> compressive stress upstream of the PI -> eigenvalue crosses zero -> mode
    2. gif2_subcritical_snap.gif      bifurcation diagram (t = 0 and t = -0.3) with the steady shapes along the branches
    3. gif3_divergence_vs_flutter.gif critical mode without friction (static buckling) vs with in-plane friction
                                      ell_b = 5 (flutter: travelling, oscillating wave)
    4. gif4_ring_pulling.gif          ring: membrane profile while the PI is pulled/pushed, force (pN) and the force-
                                      control stability window

Data: results/flow2/{crit_C_free, plate_C_rigid, gif_snap_free, gif_snap_ell5, gif_amp_t0, gif_amp_t-0.3_dense} and
results/ring_R10_tan0.5 (missing inputs: that GIF is skipped).
    python3 make_gifs.py [names...]
'''
import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
F2 = os.path.join(RES, "flow2")
OUT = os.path.join(HERE, "figures", "gifs")
os.makedirs(OUT, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRAY = "#8a8985"
INK = "#2b2a27"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3df", "grid.linewidth": 0.6, "lines.linewidth": 2, "legend.frameon": False,
                     "text.color": INK, "axes.labelcolor": INK})
L = 100.0


def rows(path):
    r = list(csv.DictReader(open(path)))
    return {k: np.array([float(x[k]) if x[k] not in ("", "None") else np.nan for x in r]) for k in r[0]
            if k != "kind"}


def masked_tri(x, y, triangles=None, r=1.0):
    '''triangulation centred on the PI, without the triangles inside the PI'''
    tri = mtri.Triangulation(x, y, triangles) if triangles is not None else mtri.Triangulation(x, y)
    xc, yc = x[tri.triangles].mean(axis=1), y[tri.triangles].mean(axis=1)
    tri.set_mask(xc ** 2 + yc ** 2 < r ** 2)
    return tri


def grid_field(tri, v, half, n=90):
    '''field on a regular grid (for 3D surfaces), NaN inside the PI'''
    g = np.linspace(-half, half, n)
    X, Y = np.meshgrid(g, g)
    Z = mtri.LinearTriInterpolator(tri, v)(X, Y).filled(np.nan)
    Z[X ** 2 + Y ** 2 < 1.0] = np.nan
    return X, Y, Z


def surface(ax, X, Y, Z, zlim, cmap="RdBu_r", title=None):
    ax.clear()
    Zc = np.clip(np.nan_to_num(Z, nan=0.0), -zlim, zlim)
    colors = plt.get_cmap(cmap)(0.5 + 0.5 * Zc / zlim)
    colors[np.isnan(Z)] = (0.17, 0.16, 0.15, 1.0)     # the PI
    ax.plot_surface(X, Y, Zc, facecolors=colors, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False)
    ax.plot_wireframe(X, Y, Zc, rstride=10, cstride=10, color="#b9b8b2", linewidth=0.4)
    ax.set_zlim(-zlim, zlim)
    ax.set_xlabel("x / r0", labelpad=-8)
    ax.set_ylabel("y / r0", labelpad=-8)
    ax.tick_params(pad=-3, labelsize=7)
    ax.set_zticks([])
    ax.view_init(elev=28, azim=-60)
    if title:
        ax.set_title(title, fontsize=9)


def flow_arrow(ax, x, y, length):
    ax.annotate("", xy=(x + length, y), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5))
    ax.text(x + length + 1, y, "flow", ha="left", va="center", fontsize=8)


def save(anim, name, fps):
    path = os.path.join(OUT, name)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(anim._fig)
    print(f"wrote figures/gifs/{name} ({os.path.getsize(path) / 1e6:.1f} MB)")


# ---------------------------------------------------------------------------------------------------------------------
def gif1():
    '''flow-induced stress, leading eigenvalue and critical mode (mesh C, force-free PI, Gamma = 25)'''
    T1 = np.load(os.path.join(F2, "plate_C_rigid", "plate_T1_rigid.npz"))
    crit = rows(os.path.join(F2, "crit_C_free", "critical_search.csv"))
    m = np.load(os.path.join(F2, "crit_C_free", "critical_mode.npz"))
    v0c, sigma0 = 0.27449, 0.0025
    x, y = T1["x"] - L / 2, T1["y"] - L / 2
    tri = masked_tri(x, y, T1["triangles"])

    def smin(v0):
        a, b, d = sigma0 + v0 * T1["T1_00"], v0 * T1["T1_01"], sigma0 + v0 * T1["T1_11"]
        return 0.5 * (a + d) - np.sqrt(0.25 * (a - d) ** 2 + b ** 2)

    zoom = 30.0
    sel = (np.abs(x) < zoom) & (np.abs(y) < zoom)
    lim = 0.04     # the stress concentration at the PI rim (down to -0.24) is clipped to show the broad compression
    mx, my = m["x"] - L / 2, m["y"] - L / 2
    keep = (np.abs(mx) < 45) & (np.abs(my) < 45)
    mtri_ = masked_tri(mx[keep], my[keep])
    mz = m["z"][keep] / np.abs(m["z"][keep]).max()
    X, Y, Z = grid_field(mtri_, mz, 42.0)
    order = np.argsort(crit["v0"])
    v_s, lam_s = crit["v0"][order], crit["lead_re"][order] * 1e5

    n_ramp, n_grow = 44, 26
    v_frames = np.concatenate([v0c * np.linspace(0, 1, n_ramp) ** 0.7, np.full(n_grow, v0c * 1.004)])
    fig = plt.figure(figsize=(12.5, 4.1))
    a0 = fig.add_subplot(1, 3, 1)
    a1 = fig.add_subplot(1, 3, 2)
    a2 = fig.add_subplot(1, 3, 3, projection="3d")
    pc = a0.tripcolor(tri, smin(0.0), shading="gouraud", cmap="RdBu", vmin=-lim, vmax=lim)
    fig.colorbar(pc, ax=a0, shrink=0.85, extend="min",
                 label=r"smallest principal stress [$\kappa/r_0^2$]  (< 0: compression)")
    a0.add_patch(plt.Circle((0, 0), 1.0, color=INK))
    a0.set_xlim(-zoom, zoom)
    a0.set_ylim(-zoom, zoom)
    a0.set_aspect("equal")
    a0.grid(False)
    a0.set_xlabel("x / r0")
    a0.set_ylabel("y / r0")
    flow_arrow(a0, -28, 26, 10)
    a1.axhline(0, color=INK, lw=0.8)
    a1.plot(100 * v_s, lam_s, "-", color=C[0], lw=2)
    a1.axvline(100 * v0c, color=GRAY, lw=1, ls="--")
    a1.text(100 * v0c, a1.get_ylim()[0], r" $SL_c$ = 27.45", color=GRAY, va="bottom", fontsize=8)
    a1.text(1, 0.3, "unstable", color=C[1], fontsize=8, va="bottom")
    a1.text(1, -0.3, "stable", color=C[0], fontsize=8, va="top")
    dot, = a1.plot([], [], "o", color=C[0], ms=9, mec="white", mew=1.5)
    a1.set_xlabel(r"flow strength $SL = \eta v_0 L / \kappa$")
    a1.set_ylabel(r"leading eigenvalue  Re $\lambda$  [$10^{-5}$]")
    a1.set_title("(b) stability of the flat membrane", loc="left")
    contours = []

    def frame(i):
        v = v_frames[i]
        pc.set_array(smin(v))
        for c in contours:
            for coll in c.collections:      # matplotlib 3.6
                coll.remove()
        contours.clear()
        if smin(v).min() < 0:
            contours.append(a0.tricontour(tri, smin(v), levels=[0.0], colors=[INK], linewidths=1.0))
        a0.set_title(f"(a) membrane stress,  SL = {100 * v:5.2f}", loc="left")
        lam = np.interp(v, v_s, lam_s)
        dot.set_data([100 * v], [lam])
        dot.set_color(C[1] if lam > 0 else C[0])
        k = i - n_ramp
        amp = 0.0 if k < 0 else 0.02 * np.exp(np.log(50.0) * k / (n_grow - 1))
        surface(a2, X, Y, amp * Z, zlim=1.0,
                title="(c) flat state: perturbations decay" if k < 0 else
                "(c) above threshold: the critical mode grows")
        return []

    anim = FuncAnimation(fig, frame, frames=len(v_frames), blit=False)
    fig.suptitle("Flow-driven buckling: the flow compresses the membrane upstream of the protein (Γ = 25, "
                 "force-free protein, linear theory)", fontsize=10)
    fig.subplots_adjust(left=0.05, right=0.98, wspace=0.35, top=0.86, bottom=0.13)
    save(anim, "gif1_flow_buckling_onset.gif", fps=10)


# ---------------------------------------------------------------------------------------------------------------------
def gif2():
    '''subcritical pitchfork (t = 0) and imperfect branch with a fold (t = -0.3), with the steady shapes'''
    br = {}
    for t, name in (("0", "gif_amp_t0"), ("-0.3", "gif_amp_t-0.3_dense")):
        p = os.path.join(F2, name, "shapes.npz")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        br[t] = np.load(p)
    d = br["-0.3"]
    x, y = d["x"] - L / 2, d["y"] - L / 2
    keep = (np.abs(x) < 45) & (np.abs(y) < 45)
    tri = masked_tri(x[keep], y[keep], None)

    def stable_segments(ax, b, color, label):
        s, A, lam = b["v0_over_v0c"], b["A"], b["lead_re"]
        st = lam < 0
        cut = [0] + [k + 1 for k in range(len(s) - 1) if st[k] != st[k + 1]] + [len(s)]
        for k0, k1 in zip(cut[:-1], cut[1:]):          # one line per stable / unstable run (dashes stay visible)
            k1 = min(k1 + 1, len(s))
            ax.plot(s[k0:k1], A[k0:k1], "-" if st[k0] else "--", color=color, lw=2)
        ax.plot([], [], "-", color=color, label=label)

    fig = plt.figure(figsize=(11.5, 4.4))
    a0 = fig.add_subplot(1, 2, 1)
    a1 = fig.add_subplot(1, 2, 2, projection="3d")
    b0 = br["0"]
    a0.plot([0, 1], [0, 0], "-", color=C[0], lw=2)
    a0.plot([1, 1.35], [0, 0], "--", color=C[0], lw=2)
    stable_segments(a0, b0, C[0], "perfect: flat PI (t = 0)")
    stable_segments(a0, d, C[1], "imperfect: contact slope t = -0.3")
    a0.plot([], [], "-", color=GRAY, label="stable")
    a0.plot([], [], "--", color=GRAY, label="unstable")
    a0.set_xlim(0, 1.35)
    a0.set_ylim(-0.5, max(d["A"].max(), b0["A"].max()) * 1.05)
    a0.set_xlabel(r"flow strength  $v_0 / v_{0,c}$")
    a0.set_ylabel("buckling amplitude A (projection on the critical mode)")
    a0.legend(loc="upper right", fontsize=8)
    a0.set_title("(a) steady branches", loc="left")
    dot, = a0.plot([], [], "o", ms=10, mec="white", mew=1.5)
    snap = a0.annotate("", xy=(0.95, 0), xytext=(0.95, 0), arrowprops=dict(arrowstyle="-|>", color=C[7], lw=2))
    snap_txt = a0.text(0, 0, "", color=C[7], fontsize=8)

    s, A, lam = d["v0_over_v0c"], d["A"], d["lead_re"]
    i_fold = int(np.argmax(s))
    zmax = np.abs(d["z"][:, keep]).max()
    # path: along the stable (lower) branch up to the fold, snap, then back along the unstable upper branch
    n_up, n_hold = i_fold + 1, 14
    idx = list(range(n_up)) + [i_fold] * n_hold + list(range(i_fold, len(s))) + [len(s) - 1] * 10

    def frame(j):
        k = idx[j]
        dot.set_data([s[k]], [A[k]])
        dot.set_color(C[1])
        if n_up <= j < n_up + n_hold:
            snap.xy = (s[i_fold] + 0.02, A[i_fold] + 6)
            snap.set_position((s[i_fold] + 0.02, A[i_fold] + 0.3))
            snap_txt.set_position((1.03, A[i_fold] + 3))
            snap_txt.set_text(f"fold at {s[i_fold]:.2f} v0c:\nno nearby steady state,\nthe membrane snaps\n"
                              "(end state not computed)")
        else:
            snap.xy = snap.get_position()
            snap_txt.set_text("")
        X, Y, Z = grid_field(tri, d["z"][k][keep], 42.0)
        stab = "stable" if lam[k] < 0 else "unstable (barrier state)"
        surface(a1, X, Y, Z, zlim=zmax,
                title=f"(b) steady shape, t = -0.3, v0/v0c = {s[k]:.3f}, A = {A[k]:.2f}: {stab}")
        return []

    anim = FuncAnimation(fig, frame, frames=len(idx), blit=False)
    fig.suptitle("Subcritical buckling: the buckled branch bends back, so a slightly tilted protein snaps before "
                 "the linear threshold", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, wspace=0.12, top=0.86, bottom=0.12)
    save(anim, "gif2_subcritical_snap.gif", fps=8)


# ---------------------------------------------------------------------------------------------------------------------
def gif3():
    '''static buckling (no in-plane friction) vs flutter (in-plane friction ell_b = 5 r0)'''
    snaps = {}
    for name in ("gif_snap_free", "gif_snap_ell5"):
        snaps[name] = np.load(os.path.join(F2, name, "snapshot.npz"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.9))
    half = 45.0
    panels = []
    for a, (name, title) in zip(ax, [("gif_snap_free", "no in-plane friction: static buckling (divergence)"),
                                     ("gif_snap_ell5", r"in-plane friction, $\ell_b = 5\,r_0$: flutter")]):
        d = snaps[name]
        x, y = d["x"] - L / 2, d["y"] - L / 2
        tri = masked_tri(x, y, d["triangles"])
        zr, zi = d["mode0_re_z"], d["mode0_im_z"]
        lam = d["eigenvalues"][0]
        # normalize so that max over a period is 1 and the real part carries the largest amplitude
        th = np.linspace(0, np.pi, 64)
        amp = max(np.abs(zr * np.cos(t) - zi * np.sin(t)).max() for t in th)
        zr, zi = zr / amp, zi / amp
        pc = a.tripcolor(tri, zr, shading="gouraud", cmap="RdBu_r", vmin=-1, vmax=1)
        rim = np.abs(np.hypot(x, y) - 1.0) < 1e-6         # the PI moves as a rigid body: colour it by its height
        disk = plt.Circle((0, 0), 2.0, ec=INK, lw=1)       # drawn twice the PI size to be visible
        a.add_patch(disk)
        a.set_xlim(-half, half)
        a.set_ylim(-half, half)
        a.set_aspect("equal")
        a.grid(False)
        a.set_xlabel("x / r0")
        a.set_ylabel("y / r0")
        a.set_title(title, loc="left", fontsize=9)
        flow_arrow(a, -42, 40, 10)
        kind = "real" if abs(lam[1]) < 1e-12 else f"complex, |Im/Re| = {abs(lam[1] / lam[0]):.1f}"
        a.text(-43, -43, f"SL = {100 * float(d['v0']):.1f} (just above threshold), eigenvalue {kind}",
               fontsize=8, va="bottom")
        panels.append((pc, zr, zi, lam, disk, rim))
    fig.colorbar(panels[0][0], ax=ax, shrink=0.8, label="height of the critical mode (normalized)")
    n = 48

    def frame(i):
        ph = 2 * np.pi * i / n
        for pc, zr, zi, lam, disk, rim in panels:
            if abs(lam[1]) < 1e-12:
                z = zr * (0.35 + 0.65 * (0.5 - 0.5 * np.cos(ph)))     # grows in place (pulsed)
            else:
                z = zr * np.cos(ph) - zi * np.sin(ph)                  # Re[(z_r + i z_i) e^{i w t}]
            pc.set_array(z)
            disk.set_facecolor(plt.get_cmap("RdBu_r")(0.5 + 0.5 * float(np.mean(z[rim]))))
        return []

    anim = FuncAnimation(fig, frame, frames=n, blit=False)
    fig.suptitle("Friction with the surrounding fluid turns the static buckling into an oscillating, "
                 "travelling instability (one period shown)", fontsize=10)
    save(anim, "gif3_divergence_vs_flutter.gif", fps=12)


# ---------------------------------------------------------------------------------------------------------------------
def gif4():
    '''ring (R = 10 r0 = 100 nm, tan alpha = 0.5): membrane profile while the PI is displaced'''
    run = os.path.join(RES, "ring_R10_tan0.5")
    fd = rows(os.path.join(run, "force_displacement.csv"))
    prof = np.load(os.path.join(run, "profiles.npz"))
    kT = 1.380649e-23 * 300
    F_unit = 10 * kT / 10e-9 * 1e12            # kappa / r0 in pN (kappa = 10 kT, r0 = 10 nm)
    h, F = fd["h"], fd["F_virtual"] * F_unit
    o = np.argsort(h)
    h, F = h[o], F[o]
    dF = np.gradient(F, h)
    up = [h[i] for i in range(len(h) - 1) if h[i] >= 0 and dF[i] < 0 <= dF[i + 1]]
    down = [h[i + 1] for i in range(len(h) - 1) if h[i + 1] <= 0 and dF[i] >= 0 > dF[i + 1]]
    hs = sorted(float(k) for k in prof.files if k != "r")
    r = prof["r"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw=dict(width_ratios=[1.25, 1]))
    a0, a1 = ax
    a1.plot(10 * h, F, "-", color=C[0])
    lo = down[0] if down else h.min()
    hi = up[0] if up else h.max()
    a1.axvspan(10 * lo, 10 * hi, color="#e4e3df", zorder=0)
    a1.text(10 * 0.5 * (lo + hi), F.max(), "stable under\nforce control", ha="center", va="top", fontsize=8,
            color=GRAY)
    for sgn in (1, -1):
        a1.axhline(sgn * 1.84, color=C[1], lw=1, ls=":")
    a1.text(10 * h.min(), 1.84, "|F| = 1.8 pN: tether pulling force", color=C[1], fontsize=7, va="bottom")
    dot, = a1.plot([], [], "o", color=C[0], ms=9, mec="white", mew=1.5)
    a1.set_xlabel("PI displacement h [nm]")
    a1.set_ylabel("force on the PI [pN]")
    a1.set_title("(b) force-displacement (displacement control: stable everywhere)", loc="left", fontsize=9)
    zl = max(abs(min(hs)), abs(max(hs))) + 1
    line_r, = a0.plot([], [], "-", color=C[0], lw=2)
    line_l, = a0.plot([], [], "-", color=C[0], lw=2)
    pi, = a0.plot([], [], "-", color=INK, lw=6, solid_capstyle="butt")
    a0.plot([-10 * r[-1], 10 * r[-1]], [0, 0], "^", color=C[1], ms=9)
    a0.text(10 * r[-1], -1.5 * 10, "ring", color=C[1], ha="center", fontsize=8)
    a0.set_xlim(-10 * r[-1] * 1.1, 10 * r[-1] * 1.1)
    a0.set_ylim(-10 * zl, 10 * zl)
    a0.set_aspect("equal")
    a0.set_xlabel("r [nm]")
    a0.set_ylabel("z [nm]")
    frames = hs[::-1][:0] + [v for v in sorted(hs)] + sorted(hs)[::-1]

    def frame(i):
        hv = frames[i]
        z = prof[f"{hv:.6f}"]
        line_r.set_data(10 * r, 10 * z)
        line_l.set_data(-10 * r, 10 * z)
        pi.set_data([-10 * r[0], 10 * r[0]], [10 * hv, 10 * hv])
        Fv = np.interp(hv, h, F)
        dot.set_data([10 * hv], [Fv])
        inside = lo <= hv <= hi
        dot.set_color(C[0] if inside else C[1])
        a0.set_title(f"(a) membrane between PI and ring (R = 100 nm), h = {10 * hv:+.0f} nm, F = {Fv:+.1f} pN"
                     + ("" if inside else "\n     beyond the force maximum: a PI held at fixed force would snap"),
                     loc="left", fontsize=9)
        return []

    anim = FuncAnimation(fig, frame, frames=len(frames), blit=False)
    fig.suptitle(r"Pulling a protein out of a ring-bounded membrane (tan $\alpha$ = 0.5, $\kappa$ = 10 kT, "
                 r"$\sigma$ = 1e-6 N/m; computed heights only)", fontsize=10)
    fig.subplots_adjust(left=0.07, right=0.98, wspace=0.25, top=0.82, bottom=0.13)
    save(anim, "gif4_ring_pulling.gif", fps=3)


if __name__ == "__main__":
    todo = sys.argv[1:] or ["gif1", "gif2", "gif3", "gif4"]
    for name in todo:
        try:
            globals()[name]()
        except Exception as e:
            print(f"{name}: skipped ({type(e).__name__}: {e})")
