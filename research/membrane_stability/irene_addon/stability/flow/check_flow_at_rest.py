'''
Verification of the stability analysis on IRENE's problem with flows: membrane at rest.

For v = 0, w = 0, flat membrane (z = 0) and uniform tension sigma, the linearized equations decouple into
    * in-plane viscous modes (real, negative eigenvalues),
    * normal modes: rho d_t w = -(kappa Laplace^2 - sigma Laplace) z, d_t z = w  ->  undamped bending waves,
      lambda = +/- i sqrt(Lambda / rho), where Lambda > 0 are the eigenvalues of kappa Laplace^2 - sigma Laplace
      with z = 0 and dz/dn = 0 on the boundary (the membrane viscosity does not damp the normal motion of a flat
      membrane at linear order).
Lambda is computed independently from IRENE's *no-flow* residual on the same mesh (no_flow/static_spectrum.py,
Lambda = -lambda_static), so this checks the flow residual, the mass form and the BCs of the flow analysis.

run with:
python3 check_flow_at_rest.py square_a [mesh directory] [output directory] --static [static_spectrum.csv]
'''
import irene_paths  # noqa: F401

import argparse
import os

import numpy as np

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--static", required=True, help="static_spectrum.csv from no_flow/static_spectrum.py")
parser.add_argument("--n_modes", type=int, default=4)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)

Lambda = -np.loadtxt(options.static, skiprows=1)
Lambda = np.sort(Lambda)[:options.n_modes]

fp.set_inflow(0.0)
fsp.psi.vector().zero()
# uniform tension sigma = sigma_r (IRENE's BC at the outflow) in the initial guess
sigma_r = fp.rpam.parameters["sigma_r_const"]
from fenics import assign, interpolate, Constant
assign(fsp.psi.sub(2), interpolate(Constant(sigma_r), fsp.Q_sigma))
fp.newton()
print(f"steady state at rest: max|v| = {np.abs(fsp.psi.sub(0, deepcopy=True).vector().get_local()).max():.1e}, "
      f"max|z| = {np.abs(fsp.psi.sub(3, deepcopy=True).vector().get_local()).max():.1e}")

A, B = fp.eigenproblem()
rows = []
print("   Lambda (no-flow)   expected i*sqrt(Lambda/rho)   computed eigenvalue (flow)           rel. error |Im|   Re")
for L in Lambda:
    expected = np.sqrt(L / fp.RHO)
    pairs = ls.solve_eigenproblem_complex_shift(A, B, 0.0 + 1.0j * expected * 1.001, nev=4)
    lam = min((p.eigenvalue for p in pairs), key=lambda z: abs(z - 1j * expected))
    err = abs(abs(lam.imag) - expected) / expected
    rows.append((L, expected, lam.real, lam.imag, err))
    print(f"   {L:.6e}       {expected:.6e}                 {lam.real:+.3e} {lam.imag:+.6e}i        {err:.1e}      {lam.real:+.1e}",
          flush=True)
np.savetxt(os.path.join(out_dir, "check_flow_at_rest.csv"), np.array(rows), delimiter=",",
           header="Lambda_no_flow,expected_imag,computed_real,computed_imag,rel_error", comments="")
max_err = max(r[4] for r in rows)
max_re = max(abs(r[2]) / r[1] for r in rows)
print(f"max relative error of the frequency = {max_err:.2e}, max |Re|/|Im| = {max_re:.1e}")
print("CHECK FLOW AT REST:", "PASSED" if max_err < 5e-2 and max_re < 1e-3 else "FAILED")
