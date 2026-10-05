'''
Mobile curvature-coupled proteins in IRENE's flowing membrane around an anchored PI (square_b, flow_problem.py of this
folder): linear stability of the flat, uniform state (z = 0, phi = 0) with the full IRENE equations plus the protein
fields.

Studies:
  1. no flow: protein instability threshold chi_c (bisection on the leading eigenvalue), compared with the infinite
     flat membrane: chi_c0 = eps sigma - 2 C0 sqrt(eps sigma), C0 = 2 C (plate/analysis_mobile.py);
  2. chi_c as a function of the inflow v0 (the flow around the PI compresses the membrane upstream: lower tension
     lowers the protein threshold there; advection carries the proteins);
  3. buckling threshold v0_c as a function of chi (proteins that soften the membrane lower it);
  critical modes (z and phi) are stored for figures.

run with:
STABILITY_PARAMS=zeta=1,omega_circle_const=0,pi_height=2,C_phi=0.25,eps_phi=1,M_phi=1 python3 phi_stability.py square_b \
    [mesh] [out] --v0_list 0,0.1,0.2 --chi_list 1,0,-0.02
'''
import irene_paths  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
from fenics import Constant, FunctionSpace, assign, interpolate, project, vertex_to_dof_map

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--v0_list", default="0,0.1,0.2,0.25")
parser.add_argument("--chi_list", default="1.0,0.0,-0.02,-0.03")
parser.add_argument("--nev", type=int, default=10)
parser.add_argument("--target", type=float, default=0.0)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
t0 = time.time()
C0 = 2 * fp.C_PHI
sigma0, eps = prm["sigma_r_const"], fp.eps_phi
chi_c0 = eps * sigma0 - 2 * C0 * np.sqrt(eps * sigma0)
k_star = np.sqrt(max(C0 * np.sqrt(sigma0 / eps) - sigma0, 0.0))
states = {}


def steady(v0):
    if v0 in states:
        fsp.psi.vector()[:] = states[v0]
        fp.set_inflow(v0)
        return
    fsp.psi.vector().zero()
    assign(fsp.psi.sub(2), interpolate(Constant(sigma0), fsp.Q_sigma))
    for v in np.linspace(0, v0, 6)[1:] if v0 > 0 else [0.0]:
        fp.set_inflow(v)
        fp.newton()
    states[v0] = fsp.psi.vector().get_local().copy()


def leading(v0, chi):
    fp.chi_phi.assign(chi)
    steady(v0)
    pairs = fp.eigenpairs(target=options.target, nev=options.nev)
    return max(pairs, key=lambda p: p.eigenvalue.real)


def bisect(fun, lo, hi, n=14):
    flo = fun(lo)
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        fm = fun(mid)
        if (fm > 0) == (flo > 0):
            lo, flo = mid, fm
        else:
            hi = mid
    return 0.5 * (lo + hi)


P1 = FunctionSpace(fsp.Q.mesh(), "P", 1)
xy = P1.tabulate_dof_coordinates()


def save_mode(pair, name):
    f = ls.eigenvector_to_functions(pair, fsp.Q)[0]
    z = project(f.sub(3), P1).vector().get_local()
    phi = project(f.sub(6), P1).vector().get_local()
    np.savez(os.path.join(out_dir, f"mode_{name}.npz"), x=xy[:, 0], y=xy[:, 1],
             triangles=vertex_to_dof_map(P1)[fsp.Q.mesh().cells()], z=z, phi=phi,
             lam=np.array([pair.eigenvalue.real, pair.eigenvalue.imag]))


res = dict(C=fp.C_PHI, C0=C0, eps=eps, M=fp.M_phi, sigma0=sigma0, chi_c0_infinite=chi_c0, k_star=k_star,
           wavelength=2 * np.pi / k_star if k_star > 0 else None, phi_inflow_dirichlet=fp.PHI_INFLOW,
           pi_height=fp.PI_HEIGHT, phi_outflow=fp.PHI_OUTFLOW, phi_form=fp.PHI_FORM, chi_c=[], v0_c=[])
# 1-2. chi_c(v0)
for v0 in [float(x) for x in options.v0_list.split(",") if x]:
    g = lambda chi: leading(v0, chi).eigenvalue.real  # noqa: E731
    lo, hi = chi_c0 - 0.1, max(0.05, -chi_c0)
    if g(hi) > 0:
        res["chi_c"].append(dict(v0=v0, chi_c=None, note="unstable even at chi = %g (flow buckling)" % hi))
        print(f"v0 = {v0}: unstable at chi = {hi} (flow-driven buckling first)", flush=True)
        continue
    # the flow can stabilize the protein instability strongly (proteins are flushed out of the patch): lower the
    # bracket until the state is unstable
    while g(lo) < 0 and lo > -3.0:
        hi, lo = lo, lo - 2 * abs(lo - chi_c0) - 0.05
    if g(lo) < 0:
        res["chi_c"].append(dict(v0=v0, chi_c=None, note=f"stable down to chi = {lo:g}"))
        print(f"v0 = {v0}: stable down to chi = {lo:g}", flush=True)
        continue
    c = bisect(g, lo, hi)
    pair = leading(v0, c - 1e-4)
    save_mode(pair, f"chi_c_v{v0:g}")
    res["chi_c"].append(dict(v0=v0, SL=v0 * fp.rmsh.parameters["L"], chi_c=c, lam_imag=pair.eigenvalue.imag))
    print(f"v0 = {v0}: chi_c = {c:+.5f} (infinite flat membrane, no flow: {chi_c0:+.5f}); "
          f"Im lambda = {pair.eigenvalue.imag:+.2e}  [{time.time() - t0:.0f} s]", flush=True)
    json.dump(res, open(os.path.join(out_dir, "phi_stability.json"), "w"), indent=1)
# 3. buckling threshold v0_c(chi)
for chi in [float(x) for x in options.chi_list.split(",") if x]:
    g = lambda v0: leading(v0, chi).eigenvalue.real  # noqa: E731
    lo, hi = 0.0, 0.05
    while g(hi) < 0 and hi < 0.6:
        lo, hi = hi, hi * 1.25
    if hi >= 0.6:
        res["v0_c"].append(dict(chi=chi, v0_c=None))
        continue
    vc = bisect(g, lo, hi, n=10)
    pair = leading(round(vc * 1.002, 6), chi)
    save_mode(pair, f"v0c_chi{chi:g}")
    res["v0_c"].append(dict(chi=chi, v0_c=vc, SL_c=vc * fp.rmsh.parameters["L"], lam_imag=pair.eigenvalue.imag))
    print(f"chi = {chi}: v0_c = {vc:.5f} (SL_c = {vc * fp.rmsh.parameters['L']:.2f}); Im lambda = "
          f"{pair.eigenvalue.imag:+.2e}  [{time.time() - t0:.0f} s]", flush=True)
    json.dump(res, open(os.path.join(out_dir, "phi_stability.json"), "w"), indent=1)
print("PHI:", json.dumps(res), flush=True)
