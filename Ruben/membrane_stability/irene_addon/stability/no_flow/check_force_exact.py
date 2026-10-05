'''
Force on the protein inclusion (PI) against exact solutions, for the two ways of computing it:
    F_virtual = -dE/dh (virtual work)    and    F_line = IRENE's line-force integral (PRE Eqs. 17-18).

(a) Small deformations (linear theory). For |grad z| << 1 the shape equation is kappa Laplace^2 z - sigma Laplace z = 0,
    whose axisymmetric solutions are z = C + D ln r + a I0(q r) + b K0(q r), q = sqrt(sigma/kappa). With
    z(r0) = h, z'(r0) = s, z(R) = 0, z'(R) = 0 the energy E = int [kappa/2 (Laplace z)^2 + sigma/2 z'^2] 2 pi r dr (+ const)
    is quadratic, and F = -dE/dh is exact.
(b) Large deformation with an exact solution: the catenoid z(r) = c [ln(r + sqrt(r^2 - c^2)) - ln(R + sqrt(R^2 - c^2))]
    has zero mean curvature, hence it solves the full nonlinear shape equation for any sigma and kappa. The bending
    stresses vanish and the exact vertical force on the PI is the tension force  F = 2 pi r0 sigma sin(theta),
    sin(theta) = z'(r0) / sqrt(1 + z'(r0)^2) = c / r0. Since H = 0 on the boundary, the moment term vanishes too, so
    F_line and F_virtual must both be exact: this isolates the discretization error of the line-force integral.

run with:
python3 check_force_exact.py ring [mesh directory] [output directory]
'''
import irene_paths  # noqa: F401

import argparse
import json
import os

import numpy as np
from scipy import special
from scipy.integrate import quad

import runtime_arguments as rarg
import ring_problem as rp
import function_spaces as fsp

parser = argparse.ArgumentParser()
parser.add_argument("--eps", type=float, default=1e-3)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
r0, R, kappa = rp.r0, rp.R, rp.KAPPA
sigma0 = float(fsp.sigma.vector().get_local()[0])


def fe_forces(h, slope_r, slope_R, sigma, eps):
    rp.set_parameters(h=h, slope_r=slope_r, slope_R=slope_R, sigma=sigma)
    rp.flat_initial_guess()
    rp.solve_steady(h, h_previous=0.0, n_substeps=4)
    state = fsp.psi.vector().copy()
    F_line = rp.force_on_protein()[2]
    M = rp.moment_term()
    energies = []
    for dh in (-eps, eps):
        rp.solve_steady(h + dh, h_previous=h)
        energies.append(rp.energy())
        fsp.psi.vector()[:] = state
        rp.set_parameters(h=h)
    return -(energies[1] - energies[0]) / (2 * eps), F_line, M


# ---------------------------------------------------------------- (a) linear theory
def linear_force(h, s, sigma):
    q = np.sqrt(sigma / kappa)
    basis = [lambda r: 1.0 + 0 * r, np.log, lambda r: special.i0(q * r), lambda r: special.k0(q * r)]
    d_basis = [lambda r: 0 * r, lambda r: 1.0 / r, lambda r: q * special.i1(q * r), lambda r: -q * special.k1(q * r)]
    lap_basis = [lambda r: 0 * r, lambda r: 0 * r, lambda r: q ** 2 * special.i0(q * r),
                 lambda r: q ** 2 * special.k0(q * r)]

    def energy(hh):
        M = np.array([[f(r0) for f in basis], [f(r0) for f in d_basis], [f(R) for f in basis], [f(R) for f in d_basis]])
        coef = np.linalg.solve(M, [hh, s, 0.0, 0.0])
        lap = lambda r: sum(c * f(r) for c, f in zip(coef, lap_basis))
        dz = lambda r: sum(c * f(r) for c, f in zip(coef, d_basis))
        integrand = lambda r: (0.5 * kappa * lap(r) ** 2 + 0.5 * sigma * dz(r) ** 2) * 2 * np.pi * r
        return quad(integrand, r0, R, limit=200, epsabs=1e-13, epsrel=1e-12)[0]

    e = 1e-4
    return -(energy(h + e) - energy(h - e)) / (2 * e)


results = {"linear": [], "catenoid": []}
print("(a) small deformations: exact linear theory vs FE")
print("      h      tan(alpha)   sigma     F_exact        F_virtual      F_line        err_virtual  err_line")
for h, s, sigma in [(0.02, 0.0, sigma0), (0.0, 0.02, sigma0), (0.02, 0.0, 1.0), (0.0, 0.02, 1.0)]:
    F_exact = linear_force(h, s, sigma)
    F_v, F_l, M = fe_forces(h, s, 0.0, sigma, options.eps * 0.1)
    ev, el = abs(F_v - F_exact) / abs(F_exact), abs(F_l - F_exact) / abs(F_exact)
    print(f"  {h:6.3f}   {s:6.3f}    {sigma:7.4f}   {F_exact:+.5e}   {F_v:+.5e}   {F_l:+.5e}   {ev:.1e}      {el:.1e}",
          flush=True)
    results["linear"].append(dict(h=h, tan_alpha=s, sigma=sigma, F_exact=F_exact, F_virtual=F_v, F_line=F_l,
                                  moment_term=M, err_virtual=ev, err_line=el))

print("\n(b) catenoid (zero mean curvature, exact for any sigma): F_exact = 2 pi r0 sigma c / r0")
print("      c      h          sigma    F_exact        F_virtual      F_line        err_virtual  err_line   moment term")
for c, sigma in [(0.25, 1.0), (0.5, 1.0), (0.7, 1.0), (0.5, sigma0)]:
    z = lambda r: c * (np.log(r + np.sqrt(r ** 2 - c ** 2)) - np.log(R + np.sqrt(R ** 2 - c ** 2)))
    dz = lambda r: c / np.sqrt(r ** 2 - c ** 2)
    h = z(r0)
    F_exact = 2 * np.pi * r0 * sigma * (dz(r0) / np.sqrt(1 + dz(r0) ** 2))
    F_v, F_l, M = fe_forces(h, dz(r0), dz(R), sigma, options.eps)
    ev, el = abs(F_v - F_exact) / abs(F_exact), abs(F_l - F_exact) / abs(F_exact)
    print(f"  {c:5.2f}  {h:+.4f}   {sigma:7.4f}   {F_exact:+.5e}   {F_v:+.5e}   {F_l:+.5e}   {ev:.1e}      {el:.1e}"
          f"    {M:+.1e}", flush=True)
    results["catenoid"].append(dict(c=c, h=h, sigma=sigma, F_exact=F_exact, F_virtual=F_v, F_line=F_l,
                                    moment_term=M, err_virtual=ev, err_line=el))

json.dump(results, open(os.path.join(out_dir, "check_force_exact.json"), "w"), indent=1)
