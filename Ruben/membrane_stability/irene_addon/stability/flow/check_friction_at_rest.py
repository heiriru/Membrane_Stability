'''
Verification of the overdamped (rho -> 0) version of the flow stability analysis: membrane at rest, flat.

At rest the normal modes of the inertial model (rho > 0, zeta = 0) are undamped bending waves lambda = +/- i sqrt(Lambda/rho)
(check_flow_at_rest.py). With rho = 0 and a normal friction zeta (rho D_t w = f_n - zeta w), the same modes relax with
lambda = -Lambda / zeta. This script computes Lambda_k = rho omega_k^2 from the inertial model and compares with
-zeta lambda_k of the overdamped model, for two values of zeta. Each model is a separate run (the parameters are compiled
into the forms): call it twice,
    STABILITY_PARAMS=rho=1,zeta=0   python3 check_friction_at_rest.py square_a [mesh] [out] --mode inertial
    STABILITY_PARAMS=rho=0,zeta=Z   python3 check_friction_at_rest.py square_a [mesh] [out] --mode overdamped
and the second run compares with the result of the first one.
'''
import irene_paths  # noqa: F401

import argparse
import os

import numpy as np
from fenics import assign, interpolate, Constant

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["inertial", "overdamped"], required=True)
parser.add_argument("--n_modes", type=int, default=6)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters

fp.set_inflow(0.0)
fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
fp.newton()
A, B = fp.eigenproblem()

if options.mode == "inertial":
    pairs = ls.solve_eigenproblem(A, B, target=0.0, nev=4 * options.n_modes + 20)
    omegas = sorted({round(abs(p.eigenvalue.imag), 8) for p in pairs
                     if abs(p.eigenvalue.real) < 1e-6 * abs(p.eigenvalue) and abs(p.eigenvalue.imag) > 0})
    Lam = prm["rho"] * np.array(omegas[:options.n_modes]) ** 2
    np.savetxt(os.path.join(out_dir, "Lambda_inertial.csv"), Lam, header="Lambda = rho omega^2", comments="")
    print("Lambda from the bending waves (rho omega^2):", Lam)
else:
    Lam_ref = np.loadtxt(os.path.join(out_dir, "Lambda_inertial.csv"), skiprows=1)
    pairs = ls.solve_eigenproblem(A, B, target=0.0, nev=options.n_modes + 4)
    lam = np.sort(np.array([p.eigenvalue for p in pairs]).real)[::-1][:options.n_modes]
    imag = max(abs(p.eigenvalue.imag) for p in pairs)
    Lam = -prm["zeta"] * lam
    err = np.abs(Lam - Lam_ref[:len(Lam)]) / Lam_ref[:len(Lam)]
    rows = np.column_stack([Lam_ref[:len(Lam)], lam, Lam, err])
    np.savetxt(os.path.join(out_dir, f"check_friction_zeta{prm['zeta']:g}.csv"), rows, delimiter=",",
               header="Lambda_inertial,lambda_overdamped,-zeta*lambda,rel_error", comments="")
    for row in rows:
        print("   Lambda = %.8e   lambda = %+.8e   -zeta*lambda = %.8e   rel. error %.1e" % tuple(row))
    print(f"max relative error = {err.max():.2e}, max |Im lambda| = {imag:.1e}")
    print("CHECK FRICTION AT REST:", "PASSED" if err.max() < 1e-6 and imag < 1e-6 * abs(lam).max() else "FAILED")
