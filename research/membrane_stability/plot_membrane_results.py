'''
Figures of the force-displacement stability analysis of a protein inclusion in a pinned ring (IRENE + linear stability).
Pure numpy/matplotlib: reads the files written by irene_addon/stability/no_flow/force_displacement_stability.py and the
checks.

run with:
python3 plot_membrane_results.py [results directory] [figure directory]
'''
import csv
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

res = sys.argv[1] if len(sys.argv) > 1 else "results"
fig_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(fig_dir, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
STABLE_C, UNSTABLE_C = SERIES[0], SERIES[1]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 4, "legend.frameon": False,
    "savefig.dpi": 160, "savefig.bbox": "tight"})


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


runs = {}
for path in sorted(glob.glob(f"{res}/ring_R*_tan*/force_displacement.csv")):
    run = os.path.basename(os.path.dirname(path))
    info = json.load(open(os.path.join(os.path.dirname(path), "info.json")))
    runs[run] = (info, read_csv(path))
runs = dict(sorted(runs.items(), key=lambda kv: kv[1][0]["R"]))


def force_control_unstable(d):
    '''dF/dh > 0: the effective stiffness d^2E/dh^2 = -dF/dh is negative -> unstable under force control'''
    return np.gradient(d["F_virtual"], d["h"]) > 0


# ---------------------------------------------------------------- 1. force-displacement curves
if runs:
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 3.8), squeeze=False)
    for ax, (run, (info, d)) in zip(axes[0], runs.items()):
        unstable_fc = force_control_unstable(d)
        disp_unstable = d["lambda_max"] > 0
        ax.plot(d["h"], d["F_line"] if "F_line" in d else d["F_z"], "--", color=INK_2, lw=1.2,
                label="line-force integral (IRENE, PRE Eqs. 17-18)")
        F = np.ma.masked_where(unstable_fc, d["F_virtual"])
        ax.plot(d["h"], F, "-", color=STABLE_C, label=r"$F = -dE/dh$, stable (force & displacement control)")
        F_u = np.ma.masked_where(~unstable_fc, d["F_virtual"])
        ax.plot(d["h"], F_u, "-", color=UNSTABLE_C, lw=3, label="unstable under force control ($dF/dh > 0$)")
        if disp_unstable.any():
            ax.plot(d["h"][disp_unstable], d["F_virtual"][disp_unstable], "x", color=INK, ms=7,
                    label="unstable under displacement control")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel(r"PI displacement $h$ [$r_0$]")
        ax.set_title(f"R = {info['R']:g} $r_0$, tan$\\alpha$ = {info['tan_alpha']:g}, $\\ell$ = {info['ell']:g} $r_0$",
                     loc="left", fontsize=10)
    axes[0][0].set_ylabel(r"vertical force on the PI [$\kappa / r_0$]")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    fig.savefig(f"{fig_dir}/force_displacement.png")
    plt.close(fig)

# ---------------------------------------------------------------- 2. eigenvalues along the curve, per azimuthal number m
if runs:
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 3.6), squeeze=False)
    for ax, (run, (info, d)) in zip(axes[0], runs.items()):
        for m, c in zip(range(3), SERIES):
            key = f"lambda_m{m}"
            if key in d:
                ax.plot(d["h"], d[key], "-", color=c, label=f"m = {m}")
        ax.axhline(0, color=INK, lw=0.8)
        ax.set_xlabel(r"PI displacement $h$ [$r_0$]")
        ax.set_title(f"R = {info['R']:g} $r_0$: least stable eigenvalue per m", loc="left", fontsize=10)
    axes[0][0].set_ylabel(r"eigenvalue $\lambda$ [$\kappa/(\zeta r_0^4)$]")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/eigenvalues_vs_h.png")
    plt.close(fig)

# ---------------------------------------------------------------- 3. membrane profiles
for run, (info, d) in runs.items():
    path = f"{res}/{run}/profiles.npz"
    if not os.path.exists(path):
        continue
    prof = np.load(path)
    keys = sorted([k for k in prof.files if k != "r"], key=float)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    cmap = plt.get_cmap("RdBu_r")
    hs = np.array([float(k) for k in keys])
    for k, h in zip(keys, hs):
        ax.plot(prof["r"], prof[k], color=cmap(0.5 + 0.5 * h / max(abs(hs).max(), 1e-9)), lw=1.3)
    ax.set_xlabel(r"$r$ [$r_0$]")
    ax.set_ylabel(r"$z$ [$r_0$]")
    ax.set_title(f"Membrane profiles, R = {info['R']:g} $r_0$, tan$\\alpha$ = {info['tan_alpha']:g}"
                 f" (colour: h from {hs.min():g} to {hs.max():g})", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/profiles_{run}.png")
    plt.close(fig)

# ---------------------------------------------------------------- 4. force check: line force vs virtual work, two meshes
checks = {name: read_csv(f"{res}/checks/check_force_energy_{name}.csv")
          for name in ["ring_R5", "ring_R5_fine"] if os.path.exists(f"{res}/checks/check_force_energy_{name}.csv")}
if checks:
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, d), marker in zip(checks.items(), ["o", "s"]):
        order = np.argsort(d["h"])
        label = "coarse" if name == "ring_R5" else "fine"
        ax.plot(d["h"][order], d["F_virtual"][order], marker + "-", color=STABLE_C, mfc="white" if label == "fine" else STABLE_C,
                label=f"$-dE/dh$ (virtual work), {label} mesh")
        ax.plot(d["h"][order], d["F_line_z"][order], marker + "--", color=INK_2, mfc="white" if label == "fine" else INK_2,
                label=f"line-force integral, {label} mesh")
        ax.plot(d["h"][order], (d["F_line_z"] - d["moment_term"])[order], marker + ":", color=UNSTABLE_C,
                mfc="white" if label == "fine" else UNSTABLE_C, label=f"line force + moment term, {label} mesh")
    ax.set_xlabel(r"PI displacement $h$ [$r_0$]")
    ax.set_ylabel(r"vertical force on the PI [$\kappa / r_0$]")
    ax.set_title("Force on the PI: virtual work is mesh-converged, the line-force integral is not", loc="left", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/check_force_methods.png")
    plt.close(fig)

# ---------------------------------------------------------------- 5. stability summary
if runs:
    fig, ax = plt.subplots(figsize=(9, 0.9 + 0.8 * len(runs)))
    for i, (run, (info, d)) in enumerate(runs.items()):
        h = d["h"]
        edges = np.concatenate([[h[0]], 0.5 * (h[1:] + h[:-1]), [h[-1]]])
        unstable_fc = force_control_unstable(d)
        for j in range(len(h)):
            ax.barh(i + 0.18, edges[j + 1] - edges[j], left=edges[j], height=0.3,
                    color=UNSTABLE_C if unstable_fc[j] else STABLE_C)
            ax.barh(i - 0.18, edges[j + 1] - edges[j], left=edges[j], height=0.3,
                    color=UNSTABLE_C if d["lambda_max"][j] > 0 else STABLE_C, alpha=0.6)
    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([f"R = {info['R']:g} $r_0$\n(top: force control,\nbottom: displacement control)"
                        for info, _ in runs.values()], fontsize=7)
    ax.grid(False)
    ax.set_xlabel(r"PI displacement $h$ [$r_0$]")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=STABLE_C, label="stable"), Patch(color=UNSTABLE_C, label="unstable")],
              loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/stability_summary.png")
    plt.close(fig)

# ---------------------------------------------------------------- 6. contact-angle dependence at fixed R
by_R = {}
for run, (info, d) in runs.items():
    if "finemesh" not in run:
        by_R.setdefault(info["R"], []).append((info["tan_alpha"], d))
for R_value, items in by_R.items():
    if len(items) < 2:
        continue
    items.sort(key=lambda t: t[0])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for (tan_alpha, d), c in zip(items, SERIES):
        unstable_fc = force_control_unstable(d)
        axes[0].plot(d["h"], d["F_virtual"], "-", color=c, label=f"tan$\\alpha$ = {tan_alpha:g}")
        axes[0].plot(d["h"][unstable_fc], d["F_virtual"][unstable_fc], "o", color=c, ms=3)
        axes[1].plot(d["h"], d["lambda_max"], "-", color=c, label=f"tan$\\alpha$ = {tan_alpha:g}")
    axes[0].axhline(0, color=INK, lw=0.8)
    axes[0].set_xlabel(r"PI displacement $h$ [$r_0$]")
    axes[0].set_ylabel(r"$F = -dE/dh$ [$\kappa / r_0$]")
    axes[0].set_title(f"R = {R_value:g} $r_0$: force (dots: unstable under force control)", loc="left", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0, color=INK, lw=0.8)
    axes[1].set_xlabel(r"PI displacement $h$ [$r_0$]")
    axes[1].set_ylabel(r"least stable eigenvalue [$\kappa/(\zeta r_0^4)$]")
    axes[1].set_title("displacement control: all negative = stable", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/contact_angle_R{R_value:g}.png")
    plt.close(fig)

# ---------------------------------------------------------------- summary table
summary = {}
for run, (info, d) in runs.items():
    unstable_fc = force_control_unstable(d)
    i_min = int(np.argmin(d["F_virtual"]))
    summary[run] = dict(R=info["R"], tan_alpha=info["tan_alpha"], h_range=[float(d["h"].min()), float(d["h"].max())],
                        h_zero_force=float(np.interp(0.0, d["F_virtual"][::-1], d["h"][::-1]))
                        if (d["F_virtual"].min() < 0 < d["F_virtual"].max()) else None,
                        F_extremum=float(d["F_virtual"][i_min]), h_at_F_extremum=float(d["h"][i_min]),
                        force_control_unstable_h=[float(x) for x in d["h"][unstable_fc]],
                        displacement_control_max_lambda=float(d["lambda_max"].max()),
                        max_imag=float(d["max_imag"].max()))
json.dump(summary, open(f"{res}/summary.json", "w"), indent=1)
print(json.dumps(summary, indent=1))
