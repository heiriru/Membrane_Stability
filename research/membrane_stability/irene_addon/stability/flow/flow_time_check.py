'''
Time-dependent cross-check of the flow stability analysis (the analogue of the DNS check of the cylinder wake).

The full nonlinear equations  m(d_t psi) + F(psi) = 0  (IRENE's residual F with the free-slip / traction-free boundary
terms and the optional normal friction, mass m on v, w, z with the current area element) are integrated in time with
BDF2 (first step backward Euler), starting from the steady state psi* plus a small multiple of the leading eigenvector:
    psi(0) = psi* + eps * x_lead / max|z of x_lead|.
The growth (or decay) rate of |z - z*| is fitted and compared with the real part of the leading eigenvalue lambda.

run with:
STABILITY_PARAMS=zeta=1 python3 flow_time_check.py square_b [mesh] [out] --v0 0.2 --n_steps 200 --dt_factor 0.02
'''
import irene_paths  # noqa: F401

import argparse
import csv
import json
import os
import time

import numpy as np
from fenics import (Constant, Function, NonlinearVariationalProblem, NonlinearVariationalSolver, assign, derivative,
                    interpolate, split)

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--v0", type=float, required=True)
parser.add_argument("--v0_path", default=None, help="continuation path to v0, e.g. 0,0.1,0.15 (default: 8 steps)")
parser.add_argument("--eps", type=float, default=1e-3, help="initial amplitude of the perturbation (max |z'|)")
parser.add_argument("--dt", type=float, default=None, help="time step (default dt_factor / |Re lambda|)")
parser.add_argument("--dt_factor", type=float, default=0.02)
parser.add_argument("--n_steps", type=int, default=200)
parser.add_argument("--nev", type=int, default=12)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters

# steady state
fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
path = [float(x) for x in options.v0_path.split(",")] if options.v0_path else list(np.linspace(0, options.v0, 9))
for v in path + [options.v0]:
    fp.set_inflow(v)
    fp.newton()
psi_star = fsp.psi.vector().get_local().copy()
z_star = fsp.psi.sub(3, deepcopy=True).vector().get_local()

# leading eigenpair
pairs = fp.eigenpairs(target=0.0, nev=options.nev)
lead = max(pairs, key=lambda p: p.eigenvalue.real)
lam = lead.eigenvalue
print(f"leading eigenvalue at v0 = {options.v0}: {lam.real:+.6e} {lam.imag:+.6e} i (residual {lead.residual:.1e})",
      flush=True)
if abs(lam.imag) > 1e-8 * abs(lam):
    print("warning: the leading eigenvalue is complex; the fit below uses the envelope", flush=True)
mode = ls.eigenvector_to_functions(lead, fsp.Q)[0]
z_mode = mode.sub(3, deepcopy=True).vector().get_local()
scale = options.eps / np.abs(z_mode).max()
dt = options.dt if options.dt else options.dt_factor / abs(lam.real)

# time stepping: BDF2, m((3 psi - 4 psi_n + psi_nm1) / (2 dt)) + F(psi) = 0
psi_n = Function(fsp.Q)
psi_nm1 = Function(fsp.Q)
c0, c1, c2 = Constant(1.0 / dt), Constant(-1.0 / dt), Constant(0.0)   # backward Euler for the first step


def mass_rate():
    v, w, _, z, _, _ = split(fsp.psi)
    v1, w1, _, z1, _, _ = split(psi_n)
    v2, w2, _, z2, _, _ = split(psi_nm1)
    nu_v, nu_w, nu_z = fsp.nu_v, fsp.nu_w, fsp.nu_z
    dv = [c0 * v[k] + c1 * v1[k] + c2 * v2[k] for k in range(2)]
    dw = c0 * w + c1 * w1 + c2 * w2
    dz = c0 * z + c1 * z1 + c2 * z2
    return (fp.RHO * (dv[0] * nu_v[0] + dv[1] * nu_v[1]) + fp.RHO * dw * nu_w + dz * nu_z) * fp.sqrt_g * fp.rmsh.dx


F_time = mass_rate() + fp.F_used
problem = NonlinearVariationalProblem(F_time, fsp.psi, fp.bcs, derivative(F_time, fsp.psi, fsp.J_psi))
solver = NonlinearVariationalSolver(problem)
sprm = solver.parameters["newton_solver"]
sprm["linear_solver"] = "mumps"
sprm["absolute_tolerance"] = 1e-11
sprm["relative_tolerance"] = 1e-10
sprm["maximum_iterations"] = 20
sprm["report"] = False

x0 = psi_star + scale * lead.x_real
fsp.psi.vector()[:] = x0
psi_n.vector()[:] = x0
psi_nm1.vector()[:] = x0


def amplitude():
    return float(np.sqrt(np.mean((fsp.psi.sub(3, deepcopy=True).vector().get_local() - z_star) ** 2)))


rows = [dict(t=0.0, amplitude=amplitude())]
t = 0.0
t0 = time.time()
for n in range(options.n_steps):
    if n == 1:
        c0.assign(1.5 / dt)
        c1.assign(-2.0 / dt)
        c2.assign(0.5 / dt)
    solver.solve()
    t += dt
    psi_nm1.vector()[:] = psi_n.vector()
    psi_n.vector()[:] = fsp.psi.vector()
    rows.append(dict(t=t, amplitude=amplitude()))
    if n % 20 == 0:
        print(f"step {n}: t = {t:.4e}, amplitude = {rows[-1]['amplitude']:.4e}  [{time.time() - t0:.0f} s]", flush=True)

ts = np.array([r["t"] for r in rows])
amps = np.array([r["amplitude"] for r in rows])
# fit over the second half (the initial condition is the eigenvector, so the transient is short)
sel = slice(len(ts) // 2, None)
rate = np.polyfit(ts[sel], np.log(amps[sel]), 1)[0]
# (BDF2 time-discretization error of the rate: O((lambda dt)^2) = O(dt_factor^2))
result = dict(v0=options.v0, eigenvalue_re=lam.real, eigenvalue_im=lam.imag, fitted_rate=float(rate),
              rel_error=float(abs(rate - lam.real) / abs(lam.real)), dt=dt, n_steps=options.n_steps, eps=options.eps,
              final_amplitude=float(amps[-1]), rho=prm["rho"], eta=prm["eta"], zeta=prm["zeta"],
              SL=prm["eta"] * options.v0 * fp.rmsh.parameters["L"] / prm["kappa"], mesh=rarg.args.input_directory)
print("TIME CHECK:", json.dumps(result), flush=True)
json.dump(result, open(os.path.join(out_dir, "time_check.json"), "w"), indent=1)
with open(os.path.join(out_dir, "time_series.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["t", "amplitude"])
    writer.writeheader()
    writer.writerows(rows)
np.savez(os.path.join(out_dir, "final_state.npz"), psi=fsp.psi.vector().get_local(), psi_star=psi_star)
