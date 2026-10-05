'''
Critical inflow velocity of the flow-driven divergence (static buckling) of a membrane with a protein inclusion.

As the inflow velocity v0 grows, the lowest bending waves lambda = +/- i omega_1 slow down (omega_1 -> 0), collide at the
origin and split into a real pair lambda = +/- s: the membrane buckles (divergence, as for a pipe conveying fluid).
At the threshold the Jacobian A is singular, so its location does not depend on the mass form (on rho as a mass, or on
a normal friction zeta).

The threshold is located by bisection on the sign of
    f(v0) = Re(lambda_*^2),   lambda_* = eigenvalue closest to the origin among those on / near the imaginary axis or
                              in the right half plane (f < 0: oscillatory, f > 0: a real eigenvalue exists, diverged),
and refined with the square-root law lambda_*^2 ~ (v0 - v0_c) (linear interpolation of f between the last bracket).

run with:
python3 flow_critical.py square_a [mesh directory] [output directory] --bracket 8,9 --scan 0,2,4,6,8,9
'''
import irene_paths  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
from fenics import assign, interpolate, Constant

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--scan", default="0,2,4,6,8", help="v0 values of the preliminary scan (continuation)")
parser.add_argument("--bracket", default=None, help="v0_low,v0_high: the threshold lies in between")
parser.add_argument("--tol", type=float, default=1e-3, help="relative tolerance of the bisection on v0")
parser.add_argument("--nev", type=int, default=12)
parser.add_argument("--target", type=float, default=0.0)
parser.add_argument("--criterion", choices=["divergence", "collision", "leading"], default="divergence",
                    help="divergence (default): f = largest real part among real (or nearly real) eigenvalues with "
                         "positive real part, -1 if there is none; collision: f = Re(lambda_*^2) (inertial model, "
                         "eigenvalues collide on the imaginary axis); leading: f = largest real part (overdamped model)")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)

prm = fp.rpam.parameters
L = fp.rmsh.parameters["L"]

fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))

states = {}   # v0 -> converged steady state (continuation starts from the closest one)


def steady_state(v0, n_substeps=4, max_halvings=5):
    '''steady state at inflow v0 by continuation from the closest converged state; failed steps are halved'''
    if states:
        v_start = min(states, key=lambda v: abs(v - v0))
        fsp.psi.vector()[:] = states[v_start]
    else:
        v_start = 0.0
    targets = list(np.linspace(v_start, v0, n_substeps + 1)[1:]) if v0 != v_start else [v0]
    min_step = abs(v0 - v_start) / n_substeps / 2 ** max_halvings
    current, backup = v_start, fsp.psi.vector().copy()
    while targets:
        target = targets[0]
        fp.set_inflow(target)
        try:
            fp.newton()
            backup = fsp.psi.vector().copy()
            current = target
            targets.pop(0)
        except RuntimeError:
            fsp.psi.vector()[:] = backup
            if abs(target - current) <= min_step * 1.0001:
                fp.set_inflow(current)
                raise
            targets.insert(0, 0.5 * (current + target))
    states[v0] = fsp.psi.vector().get_local().copy()


def near_axis(lam):
    '''eigenvalues on/near the imaginary axis or unstable (excludes the strongly damped viscous modes)'''
    return lam.real > -0.05 * abs(lam) - 1e-8


records = []


modes = {}


def evaluate(v0):
    t0 = time.time()
    steady_state(v0)
    t1 = time.time()
    pairs = fp.eigenpairs(target=options.target, nev=options.nev)
    lams = np.array([p.eigenvalue for p in pairs])
    cand = [l for l in lams if near_axis(l)]
    lam_star = min(cand, key=abs) if cand else np.nan
    lead = max(lams, key=lambda l: l.real)
    lead_pair = max(pairs, key=lambda p: p.eigenvalue.real)
    modes[f"{v0:g}"] = ls.eigenvector_to_functions(lead_pair, fsp.Q)[0].sub(3, deepcopy=True).vector().get_local()
    modes[f"base_{v0:g}"] = fsp.psi.sub(3, deepcopy=True).vector().get_local()
    if options.criterion == "leading":
        lam_star = lead
        f = lead.real
    elif options.criterion == "divergence":
        # real (or nearly real) unstable eigenvalues; the weak oscillatory growth of the bending waves (|Im| >> Re,
        # not mesh converged, see README) is not counted
        real_unstable = [l for l in lams if l.real > 0 and abs(l.imag) <= 0.5 * l.real]
        lam_star = max(real_unstable, key=lambda l: l.real) if real_unstable else lead
        f = lam_star.real if real_unstable else -1.0
    else:
        f = (lam_star ** 2).real
    rec = dict(v0=v0, f=float(f), lam_star_re=float(lam_star.real), lam_star_im=float(abs(lam_star.imag)),
               lead_re=float(lead.real), lead_im=float(abs(lead.imag)),
               max_abs_z=float(np.abs(fsp.psi.sub(3, deepcopy=True).vector().get_local()).max()),
               residual=float(lead_pair.residual),
               t_newton=t1 - t0, time=time.time() - t0)
    records.append(rec)
    print(f"v0 = {v0:.6f}: lambda_* = {lam_star.real:+.4e} {abs(lam_star.imag):+.4e} i, f = {f:+.4e}, "
          f"leading = {lead.real:+.4e} {abs(lead.imag):+.4e} i, max|z*| = {rec['max_abs_z']:.3e}  "
          f"[{rec['t_newton']:.0f} + {rec['time'] - rec['t_newton']:.0f} s]", flush=True)
    return f, lams


spectra = {}
newton_failed_at = None
for v0 in [float(x) for x in options.scan.split(",")]:
    try:
        _, spectra[f"{v0:g}"] = evaluate(v0)
    except RuntimeError:
        # no steady state reached by continuation (e.g. past a fold of the steady branch): stop the scan here
        newton_failed_at = v0
        print(f"v0 = {v0:.6f}: continuation failed (Newton), scan stopped", flush=True)
        break
    if options.criterion == "divergence" and records[-1]["f"] > 0:
        break   # first unstable point found: bracket with the previous one
    if options.criterion == "leading" and records[-1]["f"] > 0:
        break

bracket = None
if options.bracket:
    lo, hi = [float(x) for x in options.bracket.split(",")]
    f_lo, _ = evaluate(lo)
    f_hi, _ = evaluate(hi)
    bracket = (lo, hi, f_lo, f_hi)
else:
    # first sign change in the scan
    for a, b in zip(records[:-1], records[1:]):
        if a["f"] < 0 < b["f"]:
            bracket = (a["v0"], b["v0"], a["f"], b["f"])
            break
if bracket is None:
    print("no threshold in the scanned range", flush=True)
else:
    lo, hi, f_lo, f_hi = bracket
    if not (f_lo < 0 < f_hi):
        raise RuntimeError(f"no sign change in the bracket: f({lo}) = {f_lo}, f({hi}) = {f_hi}")
    while hi - lo > options.tol * hi:
        mid = 0.5 * (lo + hi)
        f_mid, _ = evaluate(mid)
        if f_mid < 0:
            lo, f_lo = mid, f_mid
        else:
            hi, f_hi = mid, f_mid
    # interpolation of the root (f is linear in v0 near the threshold for 'collision' and 'leading'; for
    # 'divergence' f jumps from -1 to a small positive value, and the midpoint is used)
    v0_c = lo - f_lo * (hi - lo) / (f_hi - f_lo) if options.criterion != "divergence" else 0.5 * (lo + hi)
    result = dict(v0_c=v0_c, bracket=[lo, hi], SL_c=prm["eta"] * v0_c * L / prm["kappa"],
                  Re_c=prm["rho"] * v0_c * L / prm["eta"] if prm["eta"] else None,
                  rho=prm["rho"], eta=prm["eta"], kappa=prm["kappa"], sigma_r=prm["sigma_r_const"], zeta=prm["zeta"],
                  L=L, W=fp.rmsh.parameters["h"], r=fp.rmsh.parameters["r"], dofs=fsp.Q.dim(),
                  b_friction=fp.B_FRICTION, ell_b=float(np.sqrt(prm["eta"] / fp.B_FRICTION)) if fp.B_FRICTION else None,
                  z_outer=fp.Z_OUTER, pi_height=fp.PI_HEIGHT, omega_circle=prm["omega_circle_const"],
                  Gamma=prm["sigma_r_const"] * L ** 2 / prm["kappa"], mesh=rarg.args.input_directory,
                  h_min=fp.rmsh.r_mesh, criterion=options.criterion, newton_failed_at=newton_failed_at)
    print("CRITICAL:", json.dumps(result), flush=True)
    json.dump(result, open(os.path.join(out_dir, "critical.json"), "w"), indent=1)
    # eigenvector of the unstable real mode just above the threshold, for plotting
    steady_state(hi)
    pairs = fp.eigenpairs(target=options.target, nev=options.nev)
    unstable = max(pairs, key=lambda p: p.eigenvalue.real)
    mode = ls.eigenvector_to_functions(unstable, fsp.Q)
    z_mode = mode[0].sub(3, deepcopy=True)
    coords = fsp.Q_z.tabulate_dof_coordinates()
    np.savez(os.path.join(out_dir, "critical_mode.npz"), x=coords[:, 0], y=coords[:, 1],
             z=z_mode.vector().get_local(), eigenvalue=unstable.eigenvalue, v0=hi,
             base=states[hi], x_real=unstable.x_real)

import csv
with open(os.path.join(out_dir, "critical_search.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    writer.writeheader()
    writer.writerows(records)
np.savez(os.path.join(out_dir, "scan_spectra.npz"), **spectra)
coords = fsp.Q_z.tabulate_dof_coordinates()
np.savez(os.path.join(out_dir, "leading_modes.npz"), x=coords[:, 0], y=coords[:, 1], **modes)
print("... done.", flush=True)
