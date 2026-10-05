'''
Animations of the sixth study (figures/gifs/gif13-14), with the 3D renderer of make_gifs_round5.py:
    gif13_codim2_static_runaway_3d.gif   M = 10, chi = 0.10, SL = 16: buckling grows and snaps into a pit, the proteins
                                         condense in it (subcritical static branch)
    gif14_codim2_oscillatory_3d.gif      M = 10, chi = 0.05, SL = 20: a travelling protein-membrane oscillation grows
                                         and ends in the same runaway (subcritical Hopf branch)
    python3 make_gifs_round6.py
'''
import os

import numpy as np

import make_gifs_round5 as g5

HERE = os.path.dirname(os.path.abspath(__file__))
R6 = os.path.join(HERE, "results", "round6")


def gif(run, name, title, sub, n_last=60):
    frames, phmax, zr, d = g5.protein_frames(run, nmax=400)
    # the interesting part is the nonlinear phase: keep the frames from the time |z| exceeds 0.05
    zmax = np.array([np.abs(f["verts"][:, :, 2]).max() for f in frames])
    start = max(0, int(np.argmax(zmax > 0.05)) - 5)
    frames = frames[start:]
    if len(frames) > n_last:
        frames = [frames[i] for i in np.linspace(0, len(frames) - 1, n_last).astype(int)]
    phm = max(np.abs(f["c"]).max() for f in frames)
    zall = np.concatenate([f["verts"][:, :, 2].ravel() for f in frames])
    c = d["c_r"] if "c_r" in d.files else (50.0, 50.0)
    g5.surface_gif(os.path.join(g5.OUT, name), frames, title, color_label="protein density φ (deviation)",
                   clim=(-phm, phm), zlim=(min(zall.min(), -0.5), max(zall.max(), 0.5)), zbox=0.3,
                   pi=(c[0], c[1], float(d["r"]) if "r" in d.files else 1.0), sub=sub, fps=8)


if __name__ == "__main__":
    gif(os.path.join(R6, "c2dyn_static"), "gif13_codim2_static_runaway_3d.gif",
        "Mobile proteins at M = 10: the static buckling is subcritical (χ = 0.10, SL = 16)",
        "the buckle grows past the weakly nonlinear amplitude and snaps into a pit; proteins (colour) condense in it")
    gif(os.path.join(R6, "c2dyn_hopf"), "gif14_codim2_oscillatory_3d.gif",
        "Mobile proteins at M = 10: the travelling mode is subcritical too (χ = 0.05, SL = 20)",
        "a protein-membrane undulation travels and grows, then runs away into the same collapse")
