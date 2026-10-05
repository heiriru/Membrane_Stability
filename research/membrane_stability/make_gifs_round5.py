'''
Animations of the fifth study (figures/gifs/gif8-*), rendered in 3D from the stored snapshots:
    gif8_proteins_rest_3d.gif      mobile curvature proteins demixing around an anchored PI (no flow)
    gif9_proteins_flow_3d.gif      the same in the flowing membrane (stripes along the flow)
    gif10_proteins_source_3d.gif   proteins injected at the PI and carried downstream (wake)
    gif11_flutter_3d.gif           saturated flutter: the PI emits a wave that travels upstream
    gif12_fold_parametric_3d.gif   post-snap fold followed beyond the Monge gauge (parametric surface)
    python3 make_gifs_round5.py [which ...]
'''
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
R5 = os.path.join(HERE, "results", "round5")
OUT = os.path.join(HERE, "figures", "gifs")
INK = "#2b2a27"
LIGHT = np.array([-0.4, -0.6, 0.7]) / np.linalg.norm([-0.4, -0.6, 0.7])


def shade(verts, rgba):
    '''Lambertian shading of triangle colours'''
    nrm = np.cross(verts[:, 1] - verts[:, 0], verts[:, 2] - verts[:, 0])
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-30
    nrm *= np.sign(nrm[:, 2:3] + 1e-12)
    f = 0.55 + 0.45 * np.clip(nrm @ LIGHT, 0, 1)
    out = rgba.copy()
    out[:, :3] *= f[:, None]
    return out


def pi_disc(cx, cy, r, h, n=40):
    th = np.linspace(0, 2 * np.pi, n)
    return [list(zip(cx + r * np.cos(th), cy + r * np.sin(th), np.full(n, h)))]


def surface_gif(path, frames, title, color_label=None, cmap="PuOr_r", clim=None, zlim=None, exag=1.0,
                pi=None, fps=10, elev=32, azim0=-60, rotate=20.0, zlabel="z / r0", sub=None, zbox=0.3):
    '''frames: list of dict(verts=(ntri, 3, 3), c=(ntri,) colour values or None, t=label, pi_h=height)'''
    fig = plt.figure(figsize=(9.0, 6.4))
    ax = fig.add_axes([0.0, 0.02, 0.86, 0.9], projection="3d")
    norm = colors.Normalize(*clim) if clim else None
    cmap_ = cm.get_cmap(cmap)
    if color_label:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_)
        cax = fig.add_axes([0.88, 0.25, 0.018, 0.45])
        cb = fig.colorbar(sm, cax=cax)
        cb.set_label(color_label)
    allv = np.concatenate([f["verts"].reshape(-1, 3) for f in frames])
    lo, hi = allv.min(0), allv.max(0)

    def draw(k):
        ax.cla()
        f = frames[k]
        V = f["verts"].copy()
        V[:, :, 2] *= exag
        if f.get("c") is not None:
            rgba = cmap_(norm(f["c"]))
        else:
            zc = f["verts"][:, :, 2].mean(1)
            rgba = cm.get_cmap("RdBu_r")(colors.Normalize(*(zlim or (lo[2], hi[2])))(zc))
        fc = shade(V, rgba)
        pc = Poly3DCollection(V, facecolors=fc, edgecolors=fc, linewidths=0.3)
        ax.add_collection3d(pc)
        if pi is not None:
            ph = f.get("pi_h", 0.0) * exag
            ax.add_collection3d(Poly3DCollection(pi_disc(pi[0], pi[1], pi[2], ph + 0.05), facecolors=INK))
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        zl = np.array(zlim if zlim else (lo[2], hi[2])) * exag
        ax.set_zlim(*zl)
        ax.set_box_aspect((hi[0] - lo[0], hi[1] - lo[1], zbox * (hi[0] - lo[0])))
        ax.view_init(elev=elev, azim=azim0 + rotate * k / max(len(frames) - 1, 1))
        ax.set_xlabel("x / r0 (flow →)")
        ax.set_ylabel("y / r0")
        ax.set_zlabel(zlabel + " (vertical scale exaggerated)" if zbox * (hi[0] - lo[0]) > 1.5 * (zl[1] - zl[0])
                      else zlabel, fontsize=8)
        ax.set_title(f"{title}\n{f['t']}", fontsize=10, loc="left")
        if sub:
            ax.text2D(0.0, -0.02, sub, transform=ax.transAxes, fontsize=7, color="#555")
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=fps), dpi=85)
    plt.close(fig)
    print("wrote", path)


def protein_frames(run, stride=1, nmax=120):
    d = np.load(os.path.join(run, "snapshots.npz"))
    x, y, tri = d["x"], d["y"], d["triangles"]
    idx = list(range(0, len(d["t"]), stride))
    if len(idx) > nmax:
        idx = list(np.linspace(0, len(d["t"]) - 1, nmax).astype(int))
    frames = []
    for i in idx:
        z, ph = d["z"][i], d["phi"][i]
        verts = np.stack([x[tri], y[tri], z[tri]], axis=-1)
        frames.append(dict(verts=verts, c=ph[tri].mean(1), t=f"t = {d['t'][i]:.4g} ζr0⁴/κ"))
    phmax = max(np.abs(d["phi"]).max(), 1e-6)
    return frames, phmax, (float(d["z"].min()), float(d["z"].max())), d


def gif_proteins(name, run, title, sub):
    frames, phmax, zr, d = protein_frames(run)
    c = d["c_r"] if "c_r" in d.files else (50.0, 50.0)
    zl = (min(zr[0], -0.5), max(zr[1], 0.5))
    surface_gif(os.path.join(OUT, name), frames, title, color_label="protein density φ (deviation)",
                clim=(-phmax, phmax), zlim=zl, exag=1.0, zbox=0.25, pi=(c[0], c[1], float(d["r"]) if "r" in d.files else 1.0),
                sub=sub)


if __name__ == "__main__":
    which = sys.argv[1:] or ["rest", "flow", "source", "flutter", "fold"]
    if "test" in which:
        gif_proteins("test_proteins.gif", "/tmp/claude-0/runs/phidyn_test", "test", "")
    if "rest" in which and os.path.exists(os.path.join(R5, "phidyn_rest", "snapshots.npz")):
        gif_proteins("gif8_proteins_rest_3d.gif", os.path.join(R5, "phidyn_rest"),
                     "Mobile curvature proteins around an anchored protein, no flow",
                     "IRENE + protein density φ (spontaneous curvature Cφ, χ = −0.08 < χ_c): demixing into a "
                     "curvature-modulated pattern; height exaggerated")
    for key, gname, ttl, sub in (
            ("flow", "gif9_proteins_flow_3d.gif", "Mobile curvature proteins in a flowing membrane (SL = 1)",
             "flow along +x past the anchored protein, χ = −0.10, free (absorbing) outflow: domains grow downstream of the "
             "clean inflow, elongated along the flow, then merge into a labyrinth"),
            ("source", "gif10_proteins_source_3d.gif",
             "Proteins recruited at the anchored protein and carried downstream (SL = 1)",
             "injection at the PI rim, χ = −0.05 (no demixing on its own): a protein plume carried downstream, "
             "with a shallow membrane trough under it")):
        if key not in which:
            continue
        for run in (os.path.join(R5, f"phidyn_{key}3"), os.path.join(R5, f"phidyn_{key}2"), os.path.join(R5, f"phidyn_{key}")):
            if os.path.exists(os.path.join(run, "snapshots.npz")):
                gif_proteins(gname, run, ttl, sub)
                break

    if "fold" in which:
        import csv
        run = os.path.join(R5, "param_eq")
        if os.path.exists(os.path.join(run, "snapshots.npz")):
            d = np.load(os.path.join(run, "snapshots.npz"))
            ts = {round(float(r["t"]), 3): float(r["h"]) for r in csv.DictReader(open(os.path.join(run, "time_series.csv")))}
            tri = d["triangles"]
            u = d["u"]
            # zoom on the fold and cut away the half y > 50 (reference coordinates) to show the folded profile
            keep = np.all(np.hypot(u[tri][:, :, 0] - 50, u[tri][:, :, 1] - 50) < 32, axis=1) & \
                np.all(u[tri][:, :, 1] <= 50.5, axis=1)
            frames = []
            for i in range(len(d["t"])):
                X = d["X"][i].astype(float)
                verts = X[tri[keep]]
                frames.append(dict(verts=verts, c=None, pi_h=ts.get(round(float(d["t"][i]), 3), 0.0),
                                   t=f"t − t0 = {(d['t'][i] - d['t'][0]) / 1e3:.2f}·10³ ζr0⁴/κ"))
            zall = np.concatenate([f["verts"][:, :, 2].ravel() for f in frames])
            surface_gif(os.path.join(OUT, "gif12_fold_parametric_3d.gif"), frames,
                        "After the snap, beyond the Monge gauge: the wall turns over (parametric surface)",
                        zlim=(zall.min(), zall.max()), pi=(50.0, 50.0, 1.0), zbox=0.6, fps=4, elev=8, azim0=-62,
                        rotate=-22.0, sub="cut-away along the centreline (half y < 50 shown); force-free PI (dark disc) on "
                                          "the crest; the pit wall folds under it; flow along +x; near-true vertical scale")
    if "flutter" in which:
        import glob as _g
        runs = _g.glob(os.path.join(R5, "flut_v1.10_k*"))
        if runs and os.path.exists(os.path.join(runs[0], "snapshots.npz")):
            d = np.load(os.path.join(runs[0], "snapshots.npz"))
            x, y, tri = d["x"], d["y"], d["triangles"]
            T = 7.8915e5
            idx = list(range(len(d["t"])))[-60:]
            import csv as _csv
            ts = [(float(r["t"]), float(r["h"])) for r in _csv.DictReader(open(os.path.join(runs[0], "time_series.csv")))]
            tt = np.array([a for a, _ in ts])
            hh = np.array([b for _, b in ts])
            frames = [dict(verts=np.stack([x[tri], y[tri], d["z"][i][tri]], axis=-1), c=None,
                           pi_h=float(np.interp(d["t"][i], tt, hh)),
                           t=f"t / T = {d['t'][i] / T:.2f} (T: flutter period)") for i in idx]
            zl = float(np.abs(d["z"][idx]).max())
            surface_gif(os.path.join(OUT, "gif11_flutter_3d.gif"), frames,
                        "Saturated flutter: the anchored protein emits a wave that travels upstream (1.10 v0_c)",
                        zlim=(-zl, zl), pi=(50.0, 50.0, 1.0), zbox=0.45, fps=8, rotate=0.0, elev=24,
                        sub="in-plane friction ℓ_b = 7 r0, σ = 0.0025 κ/r0²; flow along +x; vertical scale exaggerated")
