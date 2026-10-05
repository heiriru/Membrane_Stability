'''
Force-displacement curve of a protein inclusion (PI) in a pinned ring (Ferraro & Castellana, PRE 2026, Fig. 5) and
linear stability of every steady state along the curve.

Geometry and BCs (IRENE's ring problem, lengths in units of the PI radius r0, energies in units of kappa):
    z = h, dz/dr = tan(alpha)  on the PI circle r = r0;     z = 0, dz/dr = 0  on the outer circle r = R.
For each h (continuation from h = 0 in both directions):
    1. steady state with IRENE's Newton solver setup (residual vp.F of steady_state/no_flow/variational_problem_bc_ring),
    2. Helfrich energy E(h) and vertical force exerted by the membrane on the PI by virtual work, F = -dE/dh
       (mesh-converged); for comparison also IRENE's line-force integral (dFdl_sigma_kappa_3d, PRE Eqs. (17)-(18)) and
       the same corrected by the moment term (see check_force_energy.py),
    3. eigenvalues of the linearized dynamics closest to 0 (SLEPc, shift-and-invert), with the azimuthal number m of
       every eigenmode.
Displacement control (h imposed): the steady state is stable iff all eigenvalues are negative.
Force control (force imposed, h free): stable iff, in addition, the effective stiffness d^2E/dh^2 = -dF/dh is
positive (Schur complement of the Hessian); dF/dh is obtained from the computed curve.

run with:
python3 force_displacement_stability.py ring [mesh directory] [output directory] --tan_alpha 0.5 --h_max 8 --dh 0.25
'''
import irene_paths  # noqa: F401

import argparse
import csv
import json
import os
import time

import numpy as np

import runtime_arguments as rarg
import ring_problem as rp
import function_spaces as fsp
from stability import linear_stability as ls
from stability import modes

parser = argparse.ArgumentParser()
parser.add_argument("--tan_alpha", type=float, default=0.5)
parser.add_argument("--h_max", type=float, default=8.0)
parser.add_argument("--h_min", type=float, default=None, help="most negative h (default -h_max)")
parser.add_argument("--dh", type=float, default=0.25)
parser.add_argument("--nev", type=int, default=10)
parser.add_argument("--eps", type=float, default=0.01, help="step of the finite difference for F = -dE/dh")
parser.add_argument("--profile_every", type=int, default=4, help="store the membrane profile every n steps")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
h_min = -options.h_max if options.h_min is None else options.h_min

sigma = float(fsp.sigma.vector().get_local()[0])
info = dict(r0=rp.r0, R=rp.R, kappa=rp.KAPPA, sigma=sigma, ell=float(np.sqrt(rp.KAPPA / sigma)),
            tan_alpha=options.tan_alpha, dofs=fsp.Q.dim(), mesh=rarg.args.input_directory)
print(json.dumps(info), flush=True)
json.dump(info, open(os.path.join(out_dir, "info.json"), "w"), indent=1)

radii = rp.r0 + (rp.R - rp.r0) * np.array([0.05, 0.15, 0.3, 0.5])
x_profile = np.linspace(rp.r0, rp.R, 400)


def analyse(h):
    A, B = ls.assemble_eigenproblem(rp.F_dynamics, fsp.psi, rp.bcs_hom, rp.mass)
    pairs = ls.solve_eigenproblem(A, B, target=0.0, nev=options.nev)
    result = []
    for pair in pairs[:options.nev]:
        z_mode = ls.eigenvector_to_functions(pair, fsp.Q)[0].split(deepcopy=True)[0]
        z_mode.set_allow_extrapolation(True)
        m, _ = modes.dominant_m(z_mode, (0.0, 0.0), radii)
        result.append((pair.eigenvalue.real, pair.eigenvalue.imag, m, pair.residual))
    return result


rows, spectra, profiles = [], {}, {}


def virtual_work_force(h):
    '''F = -dE/dh (vertical force of the membrane on the PI, by virtual work), central finite difference'''
    state = fsp.psi.vector().copy()
    energies = []
    for dh in (-options.eps, options.eps):
        rp.solve_steady(h + dh, h_previous=h)
        energies.append(rp.energy())
        fsp.psi.vector()[:] = state
        rp.set_parameters(h=h)
    return -(energies[1] - energies[0]) / (2 * options.eps)


def record(h, step):
    t0 = time.time()
    F_virtual = virtual_work_force(h)
    force = rp.force_on_protein()
    M = rp.moment_term()
    E = rp.energy()
    eig = analyse(h)
    z = rp.fields()[0]
    z.set_allow_extrapolation(True)
    lead = eig[0]
    lead_by_m = {}
    for lam, _, m, _ in eig:
        lead_by_m.setdefault(m, lam)
    rows.append(dict(h=h, F_virtual=F_virtual, F_x=force[0], F_y=force[1], F_z=force[2], F_z_corrected=force[2] - M, energy=E, lambda_max=lead[0], m_max=lead[2],
                     max_imag=max(abs(e[1]) for e in eig), residual=max(e[3] for e in eig),
                     **{f"lambda_m{m}": lead_by_m.get(m, np.nan) for m in range(4)}))
    spectra[f"{h:.6f}"] = np.array([(e[0], e[2]) for e in eig])
    if step % options.profile_every == 0:
        profiles[f"{h:.6f}"] = np.array([z(x, 0.0) for x in x_profile])
    print(f"h = {h:+7.3f}: F = -dE/dh = {F_virtual:+.5e}, F_line = {force[2]:+.5e}, F_z - M = {force[2] - M:+.5e}, E = {E:.5e}, lambda_max = {lead[0]:+.4e} (m = {lead[2]}), "
          f"lambda(m=0,1,2) = {lead_by_m.get(0, np.nan):+.3e} {lead_by_m.get(1, np.nan):+.3e} "
          f"{lead_by_m.get(2, np.nan):+.3e}  [{time.time() - t0:.0f} s]", flush=True)


def write_outputs():
    ordered = sorted(rows, key=lambda r: r["h"])
    with open(os.path.join(out_dir, "force_displacement.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ordered[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)
    np.savez(os.path.join(out_dir, "spectra.npz"), **spectra)
    np.savez(os.path.join(out_dir, "profiles.npz"), r=x_profile, **profiles)


rp.set_parameters(h=0.0, slope_r=options.tan_alpha, slope_R=0.0)
rp.flat_initial_guess()
rp.solve_steady(0.0, h_previous=0.0)
record(0.0, 0)
state_at_zero = fsp.psi.vector().copy()

for direction, h_end in ((+1, options.h_max), (-1, h_min)):
    fsp.psi.vector()[:] = state_at_zero
    h_prev = 0.0
    step = 0
    for h in np.arange(direction * options.dh, h_end + direction * 1e-9, direction * options.dh):
        step += 1
        try:
            rp.solve_steady(h, h_previous=h_prev, n_substeps=2)
        except RuntimeError:
            print(f"Newton failed beyond h = {h_prev:+.3f}: end of the branch in this direction", flush=True)
            break
        try:
            record(h, step)
        except RuntimeError:
            # the finite-difference solve at h +- eps (virtual-work force) failed: end of the branch
            print(f"Newton failed next to h = {h:+.3f} (force by virtual work): end of the branch in this direction",
                  flush=True)
            break
        h_prev = h
        write_outputs()

write_outputs()
print("... done.", flush=True)
