'''
Figures of the stability analysis and of the DNS cross-check (pure numpy/matplotlib, reads the files written by
stability_analysis.py and solve_dynamics.py).

run with:
python3 plot_results.py [solution directory] [figure directory]
'''
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import plot_utils as pu

sol = sys.argv[1] if len(sys.argv) > 1 else "solution"
fig_dir = sys.argv[2] if len(sys.argv) > 2 else "figures"
os.makedirs(fig_dir, exist_ok=True)

SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # categorical slots 1-3
STABLE_C, UNSTABLE_C = SERIES[0], SERIES[1]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5, "legend.frameon": False,
    "savefig.dpi": 160, "savefig.bbox": "tight"})

# literature: Re_c and omega_c of the cylinder wake (Giannetti & Luchini 2007: 46.7, 0.736;
# Sipp & Lebedev 2007: 46.8, 0.74; Jackson 1987: 46.2, 0.73)
RE_C_LIT, OMEGA_C_LIT = 46.7, 0.736
# steady flow: C_D and L_r / D (Dennis & Chang 1970; Fornberg 1980; Giannetti & Luchini 2007)
LIT_STEADY = {"Dennis & Chang 1970": ([20, 40], [2.045, 1.522], [0.94, 2.35]),
              "Fornberg 1980": ([20, 40], [2.000, 1.498], [0.91, 2.24]),
              "Giannetti & Luchini 2007": ([20, 40], [2.06, 1.54], [0.93, 2.24])}

MESHES = [m for m in ["coarse", "medium", "fine"] if os.path.exists(f"{sol}/cylinder_{m}/stability_vs_Re.csv")]


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


data = {m: read_csv(f"{sol}/cylinder_{m}/stability_vs_Re.csv") for m in MESHES}
crit = {}
for m in MESHES:
    path = f"{sol}/cylinder_{m}/critical_Re.json"
    if os.path.exists(path):
        crit[m] = json.load(open(path))

# ---------------------------------------------------------------- 1. growth rate and frequency vs Re
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for k, m in enumerate(MESHES):
    d = data[m]
    label = f"{m} mesh" + (f" (Re$_c$ = {crit[m]['Re_c']:.2f})" if m in crit else "")
    axes[0].plot(d["Re"], d["sigma"], "-o", color=SERIES[k], label=label)
    oscillatory = d["omega"] > 1e-6  # at low Re the leading eigenvalue is real (omega = 0)
    axes[1].plot(d["Re"][oscillatory], d["omega"][oscillatory], "-o", color=SERIES[k], label=f"{m} mesh")
    if m in crit:
        axes[0].plot(crit[m]["Re_c"], 0, "D", ms=7, color=SERIES[k], mec="white", mew=1.5, zorder=5)
        axes[1].plot(crit[m]["Re_c"], crit[m]["omega_c"], "D", ms=7, color=SERIES[k], mec="white", mew=1.5, zorder=5)
axes[0].axhline(0, color=INK, lw=1)
axes[0].axvline(RE_C_LIT, color=INK_2, lw=1, ls="--")
axes[0].text(RE_C_LIT + 0.8, axes[0].get_ylim()[0] * 0.85, f"literature\nRe$_c$ = {RE_C_LIT}", color=INK_2, fontsize=8)
axes[0].fill_between([axes[0].get_xlim()[0], axes[0].get_xlim()[1]], 0, 1, color="#eb6834", alpha=0.06, lw=0)
axes[0].set_ylim(top=max(max(d["sigma"]) for d in data.values()) * 1.15)
axes[0].text(0.03, 0.93, "unstable: Re $\\lambda$ > 0", transform=axes[0].transAxes, color=INK_2, fontsize=8)
axes[0].set_xlabel("Reynolds number Re")
axes[0].set_ylabel(r"growth rate $\sigma$ = max Re $\lambda$")
axes[0].set_title("Leading eigenvalue: growth rate", loc="left")
axes[0].legend(fontsize=8, loc="lower right")
axes[1].axhline(OMEGA_C_LIT, color=INK_2, lw=1, ls="--")
axes[1].axvline(RE_C_LIT, color=INK_2, lw=1, ls="--")
axes[1].text(axes[1].get_xlim()[0] + 1, OMEGA_C_LIT + 0.004, rf"literature $\omega_c$ = {OMEGA_C_LIT}", color=INK_2,
             fontsize=8)
axes[1].set_xlabel("Reynolds number Re")
axes[1].set_ylabel(r"frequency $\omega$ = Im $\lambda$")
axes[1].set_title("Leading eigenvalue: frequency", loc="left")
fig.tight_layout()
fig.savefig(f"{fig_dir}/growth_rate_vs_Re.png")
plt.close(fig)

# ---------------------------------------------------------------- 2. base-flow validation
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for k, m in enumerate(MESHES):
    d = data[m]
    axes[0].plot(d["Re"], d["C_D"], "-", color=SERIES[k], label=f"FE, {m} mesh")
    axes[1].plot(d["Re"], d["L_r"], "-", color=SERIES[k], label=f"FE, {m} mesh")
for (name, (Re, cd, lr)), marker in zip(LIT_STEADY.items(), ["s", "^", "v"]):
    axes[0].plot(Re, cd, marker, color=INK, mfc="white", ms=7, lw=0, label=name)
    axes[1].plot(Re, lr, marker, color=INK, mfc="white", ms=7, lw=0, label=name)
axes[0].set_xlabel("Re")
axes[0].set_ylabel("drag coefficient $C_D$")
axes[0].set_title("Steady base flow: drag", loc="left")
axes[1].set_xlabel("Re")
axes[1].set_ylabel("recirculation length $L_r / D$")
axes[1].set_title("Steady base flow: recirculation bubble", loc="left")
axes[0].legend(fontsize=7)
fig.tight_layout()
fig.savefig(f"{fig_dir}/base_flow_validation.png")
plt.close(fig)

# ---------------------------------------------------------------- 3. spectra at several Re
best = MESHES[-1]
spectra = np.load(f"{sol}/cylinder_{best}/spectra.npz")
keys = [k for k in ["40", "48", "52", "60"] if k in spectra.files] or spectra.files[-3:]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
colors = ["#9fc0ea", "#5b9be0", "#2a78d6", "#1d4f8f"]
for c, key in zip(colors, keys):
    lam = spectra[key]
    lam = np.concatenate([lam, lam.conj()])
    ax.plot(lam.real, lam.imag, "o", color=c, ms=5, mec="white", mew=0.6, label=f"Re = {key}")
ax.axvline(0, color=INK, lw=1)
ax.set_xlim(-0.2, 0.06)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel(r"Re $\lambda$ (growth rate)")
ax.set_ylabel(r"Im $\lambda$ (frequency)")
ax.set_title(f"Eigenvalues found near the scan shifts ({best} mesh)", loc="left")
ax.annotate("leading pair crosses\ninto Re $\\lambda$ > 0", xy=(0.0, 0.75), xytext=(-0.085, 1.3), fontsize=8, color=INK_2,
            arrowprops=dict(arrowstyle="->", color=INK_2, lw=0.8))
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
fig.savefig(f"{fig_dir}/spectra.png")
plt.close(fig)

# ---------------------------------------------------------------- 4. mesh
mesh_data = None
for m in MESHES:
    files = glob.glob(f"{sol}/cylinder_{m}/fields_Re*.npz")
    if files:
        mesh_data = np.load(files[0])
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.5, 1]))
        tri = pu.triangulation(mesh_data)
        for ax, (xl, yl) in zip(axes, [((-20, 40), (-20, 20)), ((-1.5, 4), (-1.6, 1.6))]):
            ax.triplot(tri, lw=0.25, color=INK_2)
            ax.set_xlim(*xl)
            ax.set_ylim(*yl)
            ax.set_aspect("equal")
            ax.grid(False)
        axes[0].set_title(f"{m} mesh: {len(mesh_data['cells'])} triangles (whole domain)", loc="left")
        axes[1].set_title("zoom on the cylinder", loc="left")
        fig.tight_layout()
        fig.savefig(f"{fig_dir}/mesh_{m}.png")
        plt.close(fig)

# ---------------------------------------------------------------- 5. base flows and eigenmodes
fields = {}
for tag in ["Re40", "Re50", "Re60", "Re_c"]:
    path = f"{sol}/cylinder_{best}/fields_{tag}.npz"
    if os.path.exists(path):
        fields[tag] = np.load(path)
if fields:
    tags = [t for t in ["Re40", "Re_c", "Re60"] if t in fields]
    fig, axes = plt.subplots(len(tags), 2, figsize=(12, 2.3 * len(tags)))
    axes = np.atleast_2d(axes)
    for row, tag in enumerate(tags):
        f = fields[tag]
        tri = pu.triangulation(f)
        title_re = f"Re$_c$ = {crit[best]['Re_c']:.2f}" if tag == "Re_c" and best in crit else f"Re = {tag[2:]}"
        cs = pu.plot_field(axes[row, 0], tri, f["vorticity"], (-2, 12), (-2.5, 2.5), vmax=3.0)
        axes[row, 0].tricontour(tri, f["u"][:, 0], levels=[0.0], colors=INK, linewidths=1.0)
        lam = complex(f["eigenvalue"])
        verdict = ("UNSTABLE" if lam.real > 1e-4 else "STABLE" if lam.real < -1e-4 else "NEUTRAL (threshold)")
        axes[row, 0].set_title(f"{title_re}, steady solution being tested (black: $u_x$ = 0)", loc="left", fontsize=9)
        axes[row, 0].text(0.99, 0.04, verdict, transform=axes[row, 0].transAxes, ha="right", fontsize=10,
                          fontweight="bold", color=UNSTABLE_C if lam.real > 1e-4 else STABLE_C if lam.real < -1e-4 else INK)
        pu.plot_field(axes[row, 1], tri, f["mode_vorticity_real"], (-2, 25), (-5, 5))
        growth = (f"grows like $e^{{{lam.real:.3f}t}}$" if lam.real > 1e-4 else
                  f"decays like $e^{{{lam.real:.3f}t}}$" if lam.real < -1e-4 else "neither grows nor decays")
        axes[row, 1].set_title(rf"leading eigenmode: $\lambda$ = {lam.real:+.4f} {lam.imag:+.4f}i, {growth}",
                               loc="left", fontsize=9)
        for ax in axes[row]:
            ax.grid(False)
    fig.suptitle("Left: the steady solution $w^*$ (it exists at every Re, like a pencil balanced on its tip). "
                 "Right: the perturbation shape that decides whether $w^*$ survives (amplitude arbitrary).\n"
                 "The verdict is the sign of Re $\\lambda$; what is actually observed above Re$_c$ "
                 "(vortex shedding) is in dns_vorticity.png.", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/base_flow_and_eigenmode.png")
    plt.close(fig)

# ---------------------------------------------------------------- 6. DNS cross-check
dns_runs = sorted(glob.glob(f"{sol}/dns_Re*/history.npz"), key=lambda p: float(p.split("dns_Re")[1].split("/")[0]))
coarse_eigen = {}
if os.path.exists(f"{sol}/cylinder_coarse/stability_vs_Re.csv"):
    d = data["coarse"]
    coarse_eigen.update({float(r): (s, o) for r, s, o in zip(d["Re"], d["sigma"], d["omega"])})
for path in glob.glob(f"{sol}/cylinder_coarse_extra*/stability_vs_Re.csv"):
    d = read_csv(path)
    coarse_eigen.update({float(r): (s, o) for r, s, o in zip(d["Re"], d["sigma"], d["omega"])})


def growing_run(h):
    return h["E"][-1] > h["E"].max() / 2


def fit_dns(h):
    '''
    growth rate and frequency of the perturbation in the linear regime.
    For a growing perturbation the window is 15 < t < (first time at which E = 1/2 |u - u*|^2 exceeds 1e-2, i.e. the
    perturbation is still ~100 times smaller than the saturated vortex shedding); for a decaying perturbation it
    starts 10 time units after the maximum of E (end of the transient growth caused by the kick). In the window, the
    probe signal is fitted with c + A exp(sigma t) cos(omega t + phi) (c absorbs the small non-zero value of the
    base flow at the probe).
    '''
    t, s, E = h["t"], h["v_probe0"], h["E"]
    if growing_run(h):
        t0, t1 = 15.0, t[np.where(E > 1e-2)[0][0]] if E.max() > 1e-2 else t[-1]
    else:
        t0, t1 = t[np.argmax(E)] + 10.0, t[-1]
    window = (t > t0) & (t <= t1)
    tw, sw = t[window], s[window]
    sigma_E = np.polyfit(tw, np.log(E[window]), 1)[0] / 2
    spectrum = np.abs(np.fft.rfft((sw - sw.mean()) * np.hanning(len(sw))))
    omega_fft = 2 * np.pi * np.fft.rfftfreq(len(sw), tw[1] - tw[0])[np.argmax(spectrum[1:]) + 1]
    model = lambda t, c, a, sig, om, ph: c + a * np.exp(sig * (t - tw[0])) * np.cos(om * (t - tw[0]) + ph)
    p0 = [sw.mean(), 0.5 * (sw.max() - sw.min()), sigma_E, omega_fft, 0.0]
    best = None
    for ph in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        p0[4] = ph
        try:
            popt, _ = curve_fit(model, tw, sw, p0=p0, maxfev=20000)
        except RuntimeError:
            continue
        err = np.sum((model(tw, *popt) - sw) ** 2)
        if best is None or err < best[1]:
            best = (popt, err)
    popt = best[0]
    return popt[2], abs(popt[3]), sigma_E, (t0, t1)


comparison = []
if dns_runs:
    fig, axes = plt.subplots(1, len(dns_runs), figsize=(4.2 * len(dns_runs), 3.6), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, path in zip(axes, dns_runs):
        h = np.load(path)
        Re = float(h["Re"])
        sigma, omega, sigma_E, (t0, t1) = fit_dns(h)
        amplitude = np.sqrt(2 * h["E"])
        ax.semilogy(h["t"], amplitude, color=SERIES[0], lw=1.5, label=r"DNS: $\|u - u^*\|$")
        entry = dict(Re=Re, sigma_dns=sigma, omega_dns=omega, sigma_dns_energy=sigma_E, window=[t0, t1])
        title = f"Re = {Re:g}\nDNS: $\\sigma$ = {sigma:+.4f}, $\\omega$ = {omega:.4f}"
        if Re in coarse_eigen:
            s_eig, o_eig = coarse_eigen[Re]
            tm = 0.5 * (t0 + t1)
            a_mid = np.exp(np.interp(tm, h["t"], np.log(amplitude)))
            tt = np.linspace(max(t0 - 25, 5), min(t1 + 25, h["t"][-1]), 50)
            ax.semilogy(tt, a_mid * np.exp(s_eig * (tt - tm)) * 2.5, "--", color=SERIES[1], lw=1.5,
                        label=r"slope of the eigenvalue")
            entry.update(sigma_eig=s_eig, omega_eig=o_eig)
            title += f"\neigenvalue: $\\sigma$ = {s_eig:+.4f}, $\\omega$ = {o_eig:.4f}"
        ax.axvspan(t0, t1, color=GRID, alpha=0.5, lw=0)
        ax.set_xlabel("time t")
        ax.set_title(title, loc="left", fontsize=9)
        ax.legend(fontsize=7, loc="lower right")
        if growing_run(h):
            late = h["t"] > h["t"][-1] - 100
            cl = h["C_L"][late] - h["C_L"][late].mean()
            tl = h["t"][late]
            up = np.where((cl[:-1] < 0) & (cl[1:] >= 0))[0]
            crossings = tl[up] - cl[up] * (tl[up + 1] - tl[up]) / (cl[up + 1] - cl[up])
            entry.update(St_saturated=1.0 / np.mean(np.diff(crossings)),
                         C_D_mean_saturated=h["C_D"][late].mean(), C_L_amplitude_saturated=np.abs(cl).max())
        comparison.append(entry)
    axes[0].set_ylabel("perturbation amplitude")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/dns_vs_eigenvalues.png")
    plt.close(fig)
    json.dump(comparison, open(f"{sol}/dns_vs_eigenvalues.json", "w"), indent=2)

    # lift coefficient time series
    fig, axes = plt.subplots(len(dns_runs), 1, figsize=(9, 1.9 * len(dns_runs)), sharex=True)
    axes = np.atleast_1d(axes)
    for k, (ax, path) in enumerate(zip(axes, dns_runs)):
        h = np.load(path)
        ax.plot(h["t"], h["C_L"], color=SERIES[k % 3], lw=1.2)
        ax.set_ylabel("$C_L$")
        ax.set_title(f"Re = {float(h['Re']):g}: lift coefficient", loc="left", fontsize=9)
    axes[-1].set_xlabel("time t")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/dns_lift.png")
    plt.close(fig)

    # vorticity snapshots
    fig, axes = plt.subplots(len(dns_runs), 1, figsize=(10, 2.3 * len(dns_runs)))
    axes = np.atleast_1d(axes)
    for ax, path in zip(axes, dns_runs):
        snap = np.load(path.replace("history.npz", "snapshots.npz"))
        tri = pu.triangulation(snap)
        pu.plot_field(ax, tri, snap["vorticity"][-1], (-2, 20), (-3.5, 3.5), vmax=2.5)
        Re = float(np.load(path)["Re"])
        ax.set_title(f"DNS, Re = {Re:g}: vorticity at t = {snap['t'][-1]:.0f}", loc="left", fontsize=9)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/dns_vorticity.png")
    plt.close(fig)

print(json.dumps(dict(critical=crit, dns=comparison), indent=1, default=float))

# ---------------------------------------------------------------- 7. animations of the DNS
# top: full vorticity; bottom: perturbation vorticity (minus the initial steady state), rescaled in every frame,
# with its amplitude, so that the decay (Re < Re_c) or the growth (Re > Re_c) is visible
if dns_runs:
    from matplotlib.animation import FuncAnimation, PillowWriter
    for history_path in dns_runs:
        h = np.load(history_path)
        snap = np.load(history_path.replace("history.npz", "snapshots.npz"))
        Re = float(h["Re"])
        tri = pu.triangulation(snap)
        base = snap["vorticity"][0]
        eig = coarse_eigen.get(Re)
        frames = range(0, len(snap["t"]), max(1, len(snap["t"]) // 90))
        fig, axes = plt.subplots(2, 1, figsize=(8, 5.2))

        def draw(k):
            for ax in axes:
                ax.clear()
                ax.grid(False)
            t = snap["t"][k]
            pu.plot_field(axes[0], tri, snap["vorticity"][k], (-2, 20), (-3.5, 3.5), vmax=2.5)
            axes[0].set_title(f"Re = {Re:g}, t = {t:.0f}: vorticity", loc="left", fontsize=9)
            perturbation = snap["vorticity"][k] - base
            amplitude = np.abs(perturbation).max()
            pu.plot_field(axes[1], tri, perturbation, (-2, 20), (-3.5, 3.5),
                          vmax=amplitude if amplitude > 0 else 1.0)
            verdict = ""
            if eig is not None:
                verdict = (f"   eigenvalue predicts: {'UNSTABLE' if eig[0] > 0 else 'STABLE'}, "
                           f"$\\sigma$ = {eig[0]:+.4f}")
            axes[1].set_title(f"perturbation of the vorticity (rescaled), max = {amplitude:.1e}{verdict}",
                              loc="left", fontsize=9)

        FuncAnimation(fig, draw, frames=frames).save(f"{fig_dir}/dns_Re{Re:g}_vorticity.gif",
                                                      writer=PillowWriter(fps=10), dpi=75)
        plt.close(fig)

# ================================================================ additional figures
dns_by_Re = {e["Re"]: e for e in comparison}
sweep_mesh = max(MESHES, key=lambda m: len(data[m]["Re"]))  # the mesh with the most Reynolds numbers
spectra_sweep = np.load(f"{sol}/cylinder_{sweep_mesh}/spectra.npz")
Re_keys = sorted(spectra_sweep.files, key=float)
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
cmap = LinearSegmentedColormap.from_list("Re", plt.get_cmap("Blues")(np.linspace(0.3, 1.0, 256)))
Re_min, Re_max = float(Re_keys[0]), float(Re_keys[-1])
shade = lambda Re: cmap((Re - Re_min) / max(Re_max - Re_min, 1e-9))

# ---------------------------------------------------------------- 8. eigenvalue trajectories in the complex plane
fig, ax = plt.subplots(figsize=(7, 4.6))
lead_path = []
for key in Re_keys:
    lam = spectra_sweep[key]
    lam = lam[np.argsort(-lam.real)][:8]
    ax.plot(lam.real, lam.imag, "o", color=shade(float(key)), ms=6, mec="white", mew=0.6, zorder=3)
    band = spectra_sweep[key][(spectra_sweep[key].imag > 0.6) & (spectra_sweep[key].imag < 0.9)]
    lead_path.append(band[np.argmax(band.real)] if len(band) else np.nan)
lead_path = np.array(lead_path)
ax.plot(lead_path.real, lead_path.imag, "-", color=INK_2, lw=1, zorder=2)
for k in [i for i in [0, 1, len(Re_keys) // 2, len(Re_keys) - 1] if np.isfinite(lead_path[i])][-3:]:
    ax.annotate(f"Re = {Re_keys[k]}", (lead_path[k].real, lead_path[k].imag), xytext=(6, 8),
                textcoords="offset points", fontsize=8, color=INK_2)
ax.axvline(0, color=INK, lw=1)
ax.axvspan(0, 0.2, color=UNSTABLE_C, alpha=0.06, lw=0)
ax.text(0.004, 0.08, "unstable\nhalf-plane", fontsize=8, color=INK_2)
ax.set_xlim(-0.16, 0.12)
ax.set_ylim(-0.05, 1.6)
ax.set_xlabel(r"Re $\lambda$ (growth rate $\sigma$)")
ax.set_ylabel(r"Im $\lambda$ (frequency $\omega$)")
ax.set_title(f"The 8 least stable eigenvalues for Re = {Re_keys[0]} ... {Re_keys[-1]} ({sweep_mesh} mesh);\n"
             "line: path of the leading oscillatory eigenvalue", loc="left", fontsize=9)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(Re_min, Re_max))
fig.colorbar(sm, ax=ax, label="Reynolds number")
fig.tight_layout()
fig.savefig(f"{fig_dir}/eigenvalue_trajectories.png")
plt.close(fig)

# ---------------------------------------------------------------- 9. two branches: oscillatory global mode vs real mode
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
Re_s = np.array([float(k) for k in Re_keys])
osc = np.array([max((l for l in spectra_sweep[k] if 0.6 < l.imag < 0.9), key=lambda l: l.real, default=np.nan)
                for k in Re_keys])
real_mode = np.array([max((l for l in spectra_sweep[k] if abs(l.imag) < 1e-6), key=lambda l: l.real,
                          default=np.nan) for k in Re_keys])
axes[0].plot(Re_s, osc.real, "-o", color=SERIES[0], label=r"oscillatory global mode ($\omega \approx 0.75$)")
axes[0].plot(Re_s, np.real(real_mode), "-s", color=SERIES[2], label=r"leading non-oscillatory mode ($\omega = 0$)")
axes[0].axhline(0, color=INK, lw=1)
axes[0].set_xlabel("Reynolds number Re")
axes[0].set_ylabel(r"growth rate $\sigma$")
axes[0].set_title("Which mode leads: growth rates of two branches", loc="left", fontsize=10)
axes[0].legend(fontsize=8)
axes[1].plot(Re_s, osc.imag / (2 * np.pi), "-o", color=SERIES[0])
axes[1].axhline(OMEGA_C_LIT / (2 * np.pi), color=INK_2, lw=1, ls="--")
axes[1].text(Re_s[0], OMEGA_C_LIT / (2 * np.pi) + 0.0008, "literature St at Re$_c$", fontsize=8, color=INK_2)
axes[1].set_xlabel("Reynolds number Re")
axes[1].set_ylabel(r"Strouhal number St = $\omega / 2\pi$")
axes[1].set_title("Frequency of the global mode", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{fig_dir}/mode_branches_vs_Re.png")
plt.close(fig)

# ---------------------------------------------------------------- 10. growth rate: eigenvalues vs DNS
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
for k, m in enumerate(MESHES):
    d = data[m]
    Re_all, sig_all, om_all = list(d["Re"]), list(d["sigma"]), list(d["omega"])
    if m == "coarse":
        for Re_x, (s_x, o_x) in coarse_eigen.items():
            if Re_x not in Re_all:
                Re_all.append(Re_x), sig_all.append(s_x), om_all.append(o_x)
    order = np.argsort(Re_all)
    Re_all, sig_all, om_all = np.array(Re_all)[order], np.array(sig_all)[order], np.array(om_all)[order]
    axes[0].plot(Re_all, sig_all, "-", color=SERIES[k], label=f"eigenvalue, {m} mesh")
    mask = om_all > 1e-6
    axes[1].plot(Re_all[mask], om_all[mask], "-", color=SERIES[k], label=f"eigenvalue, {m} mesh")
for e in comparison:
    axes[0].plot(e["Re"], e["sigma_dns"], "o", ms=9, color=INK, mfc="white", mew=1.5, zorder=5)
    axes[1].plot(e["Re"], e["omega_dns"], "o", ms=9, color=INK, mfc="white", mew=1.5, zorder=5)
axes[0].plot([], [], "o", ms=9, color=INK, mfc="white", mew=1.5, label="DNS (time integration, coarse mesh)")
axes[0].axhline(0, color=INK, lw=1)
axes[0].set_xlabel("Reynolds number Re")
axes[0].set_ylabel(r"growth rate $\sigma$")
axes[0].set_title("Growth rate: eigenvalue prediction vs measured in DNS", loc="left", fontsize=10)
axes[0].legend(fontsize=8)
axes[1].set_xlabel("Reynolds number Re")
axes[1].set_ylabel(r"frequency $\omega$")
axes[1].set_title("Frequency: eigenvalue prediction vs measured in DNS", loc="left", fontsize=10)
fig.tight_layout()
fig.savefig(f"{fig_dir}/eigenvalues_vs_dns.png")
plt.close(fig)

# ---------------------------------------------------------------- 11. stability map: prediction vs what happened
fig, ax = plt.subplots(figsize=(10, 0.8 + 0.55 * (len(MESHES) + 2)))
rows = [m for m in MESHES if m in crit]
Re_lo, Re_hi = 20, 105
for i, m in enumerate(rows):
    Rc = crit[m]["Re_c"]
    ax.barh(i, Rc - Re_lo, left=Re_lo, height=0.55, color=STABLE_C, alpha=0.8)
    ax.barh(i, Re_hi - Rc, left=Rc, height=0.55, color=UNSTABLE_C, alpha=0.8)
    ax.text(Rc + 0.8, i, f"Re$_c$ = {Rc:.2f}", va="center", fontsize=8, color="white", fontweight="bold")
i_lit = len(rows)
ax.barh(i_lit, RE_C_LIT - Re_lo, left=Re_lo, height=0.55, color=STABLE_C, alpha=0.35)
ax.barh(i_lit, Re_hi - RE_C_LIT, left=RE_C_LIT, height=0.55, color=UNSTABLE_C, alpha=0.35)
ax.text(RE_C_LIT + 0.8, i_lit, f"Re$_c$ = {RE_C_LIT}", va="center", fontsize=8, color=INK)
i_dns = len(rows) + 1
for e in comparison:
    unstable = e["sigma_dns"] > 0
    ax.plot(e["Re"], i_dns, "o", ms=11, color=UNSTABLE_C if unstable else STABLE_C, mec=INK, mew=0.8)
    ax.text(e["Re"], i_dns + 0.42, "grows: vortex shedding" if unstable else "decays: steady", ha="center",
            va="center", fontsize=8, color=INK)
ax.set_yticks(range(len(rows) + 2))
ax.set_yticklabels([f"FE eigenvalues, {m} mesh" for m in rows] + ["literature", "DNS outcome"])
ax.set_xlim(Re_lo, Re_hi)
ax.set_ylim(-0.6, len(rows) + 1.8)
ax.invert_yaxis()
ax.grid(False)
ax.set_xlabel("Reynolds number Re")
ax.legend(handles=[Patch(color=STABLE_C, label="stable: all Re λ < 0"),
                   Patch(color=UNSTABLE_C, label="unstable: an eigenvalue with Re λ > 0")],
          fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.tight_layout()
fig.savefig(f"{fig_dir}/stability_map.png")
plt.close(fig)

# ---------------------------------------------------------------- 12. mesh convergence and eigen-residuals
if crit:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    ms_ = [m for m in MESHES if m in crit]
    dofs = [crit[m]["dofs"] for m in ms_]
    axes[0].plot(dofs, [crit[m]["Re_c"] for m in ms_], "-o", color=SERIES[0])
    axes[0].axhspan(46.2, 47.0, color=GRID, alpha=0.8, lw=0)
    axes[0].text(dofs[0], 46.95, "literature range 46.2 - 47.0", fontsize=8, color=INK_2, va="top")
    axes[0].set_ylabel("critical Reynolds number Re$_c$")
    axes[1].plot(dofs, [crit[m]["omega_c"] for m in ms_], "-o", color=SERIES[0])
    axes[1].axhspan(0.73, 0.745, color=GRID, alpha=0.8, lw=0)
    axes[1].text(dofs[0], 0.7445, "literature range 0.73 - 0.745", fontsize=8, color=INK_2, va="top")
    axes[1].set_ylabel(r"critical frequency $\omega_c$")
    for ax, title in zip(axes[:2], ["Re$_c$ vs mesh size", r"$\omega_c$ vs mesh size"]):
        ax.set_xscale("log")
        ax.set_xlabel("degrees of freedom")
        ax.set_title(title, loc="left", fontsize=10)
        for m, x in zip(ms_, dofs):
            ax.annotate(m, (x, ax.lines[0].get_ydata()[ms_.index(m)]), xytext=(4, -12), textcoords="offset points",
                        fontsize=8, color=INK_2)
    for k, m in enumerate(MESHES):
        axes[2].semilogy(data[m]["Re"], data[m]["residual"], "o", color=SERIES[k], label=f"{m} mesh")
    axes[2].set_xlabel("Reynolds number Re")
    axes[2].set_ylabel(r"$\|A x - \lambda B x\| / \|\lambda B x\|$")
    axes[2].set_title("Residual of the leading eigenpair", loc="left", fontsize=10)
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/convergence_and_residuals.png")
    plt.close(fig)

# ---------------------------------------------------------------- 13. one-page summary
if dns_runs:
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25])
    ax = fig.add_subplot(gs[:, 0])
    m = "coarse"
    Re_c_all = np.array(sorted(coarse_eigen))
    ax.plot(Re_c_all, [coarse_eigen[r][0] for r in Re_c_all], "-o", color=SERIES[0],
            label="FE eigenvalue (coarse mesh, same as DNS)")
    for e in comparison:
        ax.plot(e["Re"], e["sigma_dns"], "o", ms=9, color=INK, mfc="white", mew=1.5, zorder=5)
    ax.plot([], [], "o", ms=9, color=INK, mfc="white", mew=1.5, label="measured in DNS")
    if m in crit:
        ax.plot(crit[m]["Re_c"], 0, "D", ms=8, color=UNSTABLE_C, mec="white", zorder=6,
                label=f"Re$_c$ = {crit[m]['Re_c']:.2f} (literature {RE_C_LIT})")
    ax.axhline(0, color=INK, lw=1)
    ax.axvspan(RE_C_LIT, 105, color=UNSTABLE_C, alpha=0.06, lw=0)
    ax.set_xlim(right=105)
    ax.set_xlabel("Reynolds number Re")
    ax.set_ylabel(r"growth rate $\sigma$ = max Re $\lambda$")
    ax.set_title("Linear stability from the FE Jacobian", loc="left")
    ax.legend(fontsize=8, loc="upper left")
    stable_run = next(p for p in dns_runs if not growing_run(np.load(p)))
    unstable_run = dns_runs[-1]
    for row, path in enumerate([stable_run, unstable_run]):
        a = fig.add_subplot(gs[row, 1])
        snap = np.load(path.replace("history.npz", "snapshots.npz"))
        Re = float(np.load(path)["Re"])
        pu.plot_field(a, pu.triangulation(snap), snap["vorticity"][-1], (-2, 20), (-3.5, 3.5), vmax=2.5)
        a.grid(False)
        verdict = "predicted unstable, DNS sheds vortices" if Re > RE_C_LIT else "predicted stable, DNS stays steady"
        a.set_title(f"DNS at Re = {Re:g} ({verdict}), t = {snap['t'][-1]:.0f}", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/summary.png")
    plt.close(fig)
