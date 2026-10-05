'''
Verification of the stability analysis built on IRENE's ring problem: flat membrane.

For h = 0 and zero slopes the steady state is z = 0, and the linearized overdamped dynamics
    zeta dz/dt = -kappa Laplace^2 z + sigma Laplace z,    z = dz/dr = 0 at r = r0 and r = R
has the exact eigenvalues lambda = -Lambda/zeta, where for each azimuthal number m, Lambda solves
    det[ J_m(k r), Y_m(k r), I_m(q r), K_m(q r) ; derivatives ]_{r = r0, R} = 0,
    kappa k^4 + sigma k^2 = Lambda,   q^2 = k^2 + sigma / kappa.
The eigenvalues computed from the Jacobian of IRENE's residual (z, omega, mu mixed formulation with penalty terms)
must converge to these values.

run with:
python3 check_flat_membrane.py ring [mesh directory] [output directory]
'''
import irene_paths  # noqa: F401  (sets up the paths to IRENE)

import argparse
import os

import numpy as np
from scipy import special
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import runtime_arguments as rarg
import ring_problem as rp
import function_spaces as fsp
from stability import linear_stability as ls
from stability import modes

parser = argparse.ArgumentParser()
parser.add_argument("--nev", type=int, default=14)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)

kappa = rp.KAPPA
sigma = float(fsp.sigma.vector().get_local()[0])
a, b = rp.r0, rp.R
zeta = rp.ZETA


def determinant(Lambda, m):
    k = np.sqrt((-sigma + np.sqrt(sigma ** 2 + 4 * kappa * Lambda)) / (2 * kappa))
    q = np.sqrt(k ** 2 + sigma / kappa)
    rows = []
    for r in (a, b):
        rows.append([special.jv(m, k * r), special.yv(m, k * r), special.iv(m, q * r), special.kv(m, q * r)])
        rows.append([k * special.jvp(m, k * r), k * special.yvp(m, k * r), q * special.ivp(m, q * r),
                     q * special.kvp(m, q * r)])
    M = np.array(rows)
    M /= np.abs(M).max(axis=0)
    return np.linalg.det(M)


def exact_eigenvalues(m, n_roots, Lambda_max):
    grid = np.logspace(-6, np.log10(Lambda_max), 20000)
    values = np.array([determinant(L, m) for L in grid])
    roots = []
    for i in np.where(np.sign(values[:-1]) != np.sign(values[1:]))[0]:
        roots.append(brentq(determinant, grid[i], grid[i + 1], args=(m,)))
        if len(roots) == n_roots:
            break
    return -np.array(roots) / zeta


# steady state: flat membrane
rp.set_parameters(h=0.0, slope_r=0.0, slope_R=0.0)
rp.flat_initial_guess()
rp.solve_steady(0.0)
z_star = rp.fields()[0]
print(f"steady state: max|z| = {np.abs(z_star.vector().get_local()).max():.2e}, residual = {rp.residual_norm():.2e}")

A, B = ls.assemble_eigenproblem(rp.F_dynamics, fsp.psi, rp.bcs_hom, rp.mass)
pairs = ls.solve_eigenproblem(A, B, target=0.0, nev=options.nev)
print(f"{len(pairs)} eigenpairs, max |Im lambda| = {max(abs(p.eigenvalue.imag) for p in pairs):.1e}")

radii = np.linspace(a, b, 7)[1:-1]
computed = []
for pair in pairs:
    z_mode = ls.eigenvector_to_functions(pair, fsp.Q)[0].split(deepcopy=True)[0]
    z_mode.set_allow_extrapolation(True)
    m, _ = modes.dominant_m(z_mode, (0.0, 0.0), radii)
    computed.append((pair.eigenvalue.real, m, pair.residual))

Lambda_max = 1.5 * max(-lam for lam, _, _ in computed) * zeta
exact = {m: exact_eigenvalues(m, 4, Lambda_max) for m in range(0, 7)}

print(f"\nflat membrane, r0 = {a}, R = {b}, kappa = {kappa}, sigma = {sigma}")
print("   lambda (FE)      m   lambda (exact)   rel. error")
errors = []
rows = []
for lam, m, res in computed:
    candidates = exact.get(m, np.array([]))
    if len(candidates) == 0:
        print(f"   {lam:+.6e}  {m:2d}   (no exact root found)")
        continue
    ex = candidates[np.argmin(np.abs(candidates - lam))]
    err = abs(lam - ex) / abs(ex)
    errors.append(err)
    rows.append((lam, m, ex, err))
    print(f"   {lam:+.6e}  {m:2d}   {ex:+.6e}   {err:.1e}")
max_error = max(errors)
print(f"max relative error = {max_error:.2e}")
print("CHECK FLAT MEMBRANE:", "PASSED" if max_error < 1e-2 and all(l < 0 for l, _, _ in computed) else "FAILED")

np.savetxt(os.path.join(out_dir, "check_flat_membrane.csv"), np.array(rows), delimiter=",",
           header="lambda_FE,m,lambda_exact,relative_error", comments="")

fig, ax = plt.subplots(figsize=(6.5, 4))
for m in range(0, 7):
    ax.plot(exact[m], [m] * len(exact[m]), "o", mfc="none", ms=11, color="#52514e")
ax.plot([r[0] for r in rows], [r[1] for r in rows], "x", ms=8, color="#2a78d6", label="FE (Jacobian of IRENE's residual)")
ax.plot([], [], "o", mfc="none", ms=11, color="#52514e", label="exact (Bessel functions)")
ax.set_xlabel(r"eigenvalue $\lambda$  [$\kappa / (\zeta r_0^4)$]")
ax.set_ylabel("azimuthal number m")
ax.set_xlim(1.1 * min(r[0] for r in rows), 0)
ax.set_title(f"Flat membrane on a ring, R = {b:g} r0, $\\ell$ = {np.sqrt(kappa / sigma):g} r0", loc="left", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "check_flat_membrane.png"), dpi=150)
