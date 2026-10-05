'''
Linear stability of a membrane with a protein inclusion (PI) in an in-plane flow, as a function of the inflow velocity
(IRENE's steady_state/flow problem 'square_a': square domain, PI = circular hole, inflow v = (v0, 0) on the left,
tension sigma_r on the right, z and dz/dn imposed on the square and on the PI).

For each v0 (continuation from v0 = 0):
    1. steady state (v, w, sigma, z, omega, mu) with IRENE's residual,
    2. eigenvalues of the linearized dynamics (mass: rho on v, w; 1 on z) with the largest real part, scanned with
       complex shifts s = s_re + i omega_k, omega_k = 0 ... omega_max (bending waves at rest: lambda = +/- i sqrt(Lambda/rho)).
The membrane at rest is marginally stable (undamped bending waves, check_flow_at_rest.py): the flow decides whether they
are damped or amplified. Dimensionless numbers: Scriven-Love SL = eta v0 L / kappa, Reynolds Re = rho v0 L / eta.

run with:
python3 flow_sweep.py square_a [mesh directory] [output directory] --v0 0,1,2,4,8 --rho 1 --eta 1
'''
import irene_paths  # noqa: F401

import argparse
import csv
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
parser.add_argument("--v0", default="0,1,2,4,8,16")
parser.add_argument("--omega_max", type=float, default=300.0, help="largest frequency scanned")
parser.add_argument("--n_shifts", type=int, default=7)
parser.add_argument("--shift_re", type=float, default=5.0)
parser.add_argument("--nev", type=int, default=6)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)

prm = fp.rpam.parameters
L = fp.rmsh.parameters["L"]
info = dict(kappa=prm["kappa"], rho=prm["rho"], eta=prm["eta"], sigma_r=prm["sigma_r_const"],
            omega_circle=prm["omega_circle_const"], L=L, r=fp.rmsh.parameters["r"], dofs=fsp.Q.dim(),
            mesh=rarg.args.input_directory)
json.dump(info, open(os.path.join(out_dir, "info.json"), "w"), indent=1)
print(json.dumps(info), flush=True)

shifts = [complex(options.shift_re, w) for w in np.linspace(0.0, options.omega_max, options.n_shifts)]

fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))

rows, spectra = [], {}
v_prev = 0.0
for v0 in [float(x) for x in options.v0.split(",")]:
    t0 = time.time()
    # continuation in v0
    for v in np.linspace(v_prev, v0, 5)[1:] if v0 != v_prev else [v0]:
        fp.set_inflow(v)
        fp.newton()
    v_prev = v0
    z_star = fsp.psi.sub(3, deepcopy=True).vector().get_local()
    A, B = fp.eigenproblem()
    pairs = ls.leading_eigenvalues(A, B, shifts=shifts, nev=options.nev)
    lead = pairs[0]
    spectra[f"{v0:g}"] = np.array([p.eigenvalue for p in pairs])
    rows.append(dict(v0=v0, SL=prm["eta"] * v0 * L / prm["kappa"], Re=prm["rho"] * v0 * L / prm["eta"],
                     sigma_max=lead.eigenvalue.real, omega_at_max=abs(lead.eigenvalue.imag), residual=lead.residual,
                     max_abs_z=float(np.abs(z_star).max())))
    print(f"v0 = {v0:8.3f} (SL = {rows[-1]['SL']:.3g}, Re = {rows[-1]['Re']:.3g}): leading eigenvalue "
          f"{lead.eigenvalue.real:+.5e} {lead.eigenvalue.imag:+.5e} i  (residual {lead.residual:.1e}), "
          f"max|z*| = {rows[-1]['max_abs_z']:.2e}  [{time.time() - t0:.0f} s]", flush=True)

with open(os.path.join(out_dir, "flow_sweep.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
np.savez(os.path.join(out_dir, "spectra.npz"), **spectra)
print("... done.", flush=True)
