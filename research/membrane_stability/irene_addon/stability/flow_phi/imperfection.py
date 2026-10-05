'''
Does a protein wake smooth the snap? Proteins recruited at the anchored PI (a source of total rate Q on the PI rim,
as in phi_dynamics.py --Q_source) and carried downstream break the up-down symmetry of the flat flowing membrane
(the proteins have a spontaneous curvature: the plume bends the membrane), so they act as an imperfection of the
static buckling, like the contact slope t of the PI (section 8).

1. Threshold v0_c of the static buckling without source (bisection on the leading eigenvalue).
2. Weakly nonlinear reduction there (as flow_landau.py, with the protein fields): with the critical mode q0 (max|z| = 1),
   the left eigenvector y0 (y0^T B q0 = 1), the dynamics B psi_t = A d + B2(d, d)/2 + C3(d, d, d)/6 + Q f_Q gives
       dx/dt = lambda_v (v0 - v0_c) x + a x^3 + H_Q Q,
       a = y0^T [B2(q0, h2)/2 + C3(q0, q0, q0)/6],  h2 = -A^-1 B2(q0, q0),   H_Q = y0^T f_Q,
   f_Q = (1 / circumference) nu_phi on the rim. a > 0: subcritical (snap); then the source cannot remove the snap at
   small Q: the branch connected to the flat state has a fold at
       x_f^3 = H_Q Q / (2 a),   1 - v_fold / v0_c = 3 a x_f^2 / (lambda_v v0_c)   (Koiter: proportional to Q^(2/3)).
3. Check: steady states with the source continued in v0 (Newton), fold located by bisection on the leading real
   eigenvalue (Newton failure counts as beyond the fold).

run with:
STABILITY_PARAMS=zeta=1,omega_circle_const=0,pi_height=1,C_phi=0.25,eps_phi=1,M_phi=1,phi_sponge=2,phi_form=1,beta_phi=1,chi_phi=-0.05 \
    python3 imperfection.py square_b [mesh] [out] --Q_list 0.01,0.03,0.1
'''
import irene_paths  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
from fenics import (Constant, Function, NonlinearVariationalProblem, NonlinearVariationalSolver, as_backend_type,
                    assemble, assign, derivative, interpolate)
from petsc4py import PETSc

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--Q_list", default="0.01,0.03,0.1,0.3")
parser.add_argument("--v0_c", type=float, default=None)
parser.add_argument("--nev", type=int, default=8)
parser.add_argument("--wnl_only", action="store_true")
parser.add_argument("--v_lo", type=float, default=0.15, help="bisection bracket for the static threshold")
parser.add_argument("--v_hi", type=float, default=0.2)
parser.add_argument("--v_start", type=float, default=0.8, help="continuation start, in units of v0_c")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
t0 = time.time()
assert fp.PI_HEIGHT == 1
L = fp.rmsh.parameters["L"]
Q = fsp.Q
odd = np.concatenate([np.array(Q.sub(k).dofmap().dofs()) for k in (1, 3, 4, 5, 6, 7)])
bc_dofs = np.array(sorted(set().union(*[set(bc.get_boundary_values().keys()) for bc in fp.bcs_hom])), dtype=int)
circ = assemble(Constant(1.0) * fp.rmsh.ds_circle)
Qc = Constant(0.0)
F_src = fp.F_used - Qc / circ * fsp.nu_phi * fp.rmsh.ds_circle
res = dict(M=fp.M_phi, chi=float(fp.chi_phi.values()[0]), beta=float(fp.beta_phi.values()[0]), C=fp.C_PHI,
           phi_outflow=fp.PHI_OUTFLOW, phi_form=fp.PHI_FORM)


def newton(F):
    problem = NonlinearVariationalProblem(F, fsp.psi, fp.bcs, derivative(F, fsp.psi, fsp.J_psi))
    solver = NonlinearVariationalSolver(problem)
    sp = solver.parameters["newton_solver"]
    sp.update({"linear_solver": "mumps", "absolute_tolerance": 1e-10, "relative_tolerance": 1e-11,
               "maximum_iterations": 25, "report": False})
    return solver.solve()


def flat(v0, Qv=0.0, n=6):
    fsp.psi.vector().zero()
    assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
    Qc.assign(0.0)
    for v in np.linspace(0, v0, n + 1)[1:]:
        fp.set_inflow(v)
        newton(F_src)
    for q in np.linspace(0, Qv, 9)[1:] if Qv else []:
        Qc.assign(q)
        newton(F_src)


def eig(F, nev=None):
    A, B = ls.assemble_eigenproblem(F, fsp.psi, fp.bcs_hom, fp.mass)
    pairs = ls.solve_eigenproblem(A, B, target=0.0, nev=nev or options.nev)
    return max(pairs, key=lambda p: p.eigenvalue.real), A, B


# 1. threshold without source
cache = {}


def lead0(v0):
    if v0 not in cache:
        flat(v0)
        cache[v0] = eig(F_src)[0].eigenvalue
    return cache[v0]


if options.v0_c:
    v0_c = options.v0_c
else:
    lo, hi = options.v_lo, options.v_hi
    while lead0(hi).real < 0:
        lo, hi = hi, hi * 1.15
    for _ in range(16):
        mid = 0.5 * (lo + hi)
        if lead0(mid).real > 0:
            hi = mid
        else:
            lo = mid
    v0_c = 0.5 * (lo + hi)
print(f"v0_c = {v0_c:.6f} (SL_c = {v0_c * L:.3f}), leading eigenvalue {lead0(v0_c):.3e}  [{time.time() - t0:.0f} s]",
      flush=True)
dv = 1e-3 * v0_c
lambda_v = (lead0(v0_c + dv).real - lead0(v0_c - dv).real) / (2 * dv)

# 2. weakly nonlinear coefficients
flat(v0_c)
psi0 = fsp.psi.vector().get_local().copy()
lead, A_mat, B_mat = eig(F_src)
Am, Bm = as_backend_type(A_mat).mat(), as_backend_type(B_mat).mat()
A_s, B_s = ls.petsc_to_scipy(Am), ls.petsc_to_scipy(Bm)
q0 = lead.x_real.copy()
f = Function(Q)
f.vector()[:] = q0
zq = f.sub(3, deepcopy=True).vector().get_local()
q0 /= zq[np.argmax(np.abs(zq))]
At, Bt = Am.copy(), Bm.copy()
At.transpose()
Bt.transpose()
adj = min(ls.solve_eigenproblem(At, Bt, target=lead.eigenvalue.real, nev=4), key=lambda p: abs(p.eigenvalue - lead.eigenvalue))
y0 = adj.x_real.copy()
y0[bc_dofs] = 0.0
y0 /= y0 @ (B_s @ q0)
U1, U2, U3 = Function(Q), Function(Q), Function(Q)
d2_form = derivative(derivative(fp.F_used, fsp.psi, U1), fsp.psi, U2)
d3_form = derivative(derivative(derivative(fp.F_used, fsp.psi, U1), fsp.psi, U2), fsp.psi, U3)


def _vec(form):
    b = assemble(form).get_local()
    b[bc_dofs] = 0.0
    return -b


fsp.psi.vector()[:] = psi0
U1.vector()[:] = q0
U2.vector()[:] = q0
U3.vector()[:] = q0
b00 = _vec(d2_form)
c000 = _vec(d3_form)
ksp = PETSc.KSP().create()
ksp.setOperators(Am)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
r_, s_ = Am.createVecs()
r_.setArray(-b00)
ksp.solve(r_, s_)
h2 = s_.getArray().copy()
h2[odd] = 0.0
U2.vector()[:] = h2
b0h = _vec(d2_form)
a_quad, a_cub = float(y0 @ (0.5 * b0h)), float(y0 @ c000 / 6.0)
a = a_quad + a_cub
fQ = assemble(1.0 / circ * fsp.nu_phi * fp.rmsh.ds_circle).get_local()
fQ[bc_dofs] = 0.0
H_Q = float(y0 @ fQ)
print(f"lambda_v = {lambda_v:+.4e}, a = {a:+.4e} (second order {a_quad:+.3e}, cubic {a_cub:+.3e}), H_Q = {H_Q:+.4e}; "
      f"symmetry: odd part of B2(q0, q0) {np.linalg.norm(b00[odd]) / np.linalg.norm(b00):.1e}  [{time.time() - t0:.0f} s]",
      flush=True)


def fold_prediction(Qv):
    if a <= 0:
        return None, None
    xf = np.cbrt(H_Q * Qv / (2 * a))
    return float(xf), float(3 * a * xf ** 2 / (lambda_v * v0_c))


res.update(v0_c=v0_c, SL_c=v0_c * L, lambda_v=lambda_v, a=a, a_parts=dict(second_order=a_quad, cubic=a_cub), H_Q=H_Q,
           criticality="subcritical (snap)" if a > 0 else "supercritical (smooth)",
           perfect_branch_coef=-a / (lambda_v * v0_c), runs=[])
json.dump(res, open(os.path.join(out_dir, "imperfection.json"), "w"), indent=1)
if options.wnl_only:
    raise SystemExit


# 3. continuation with the source
def zmax():
    zz = fsp.psi.sub(3, deepcopy=True).vector().get_local()
    return float(zz[np.argmax(np.abs(zz))])


def phimax():
    return float(np.abs(fsp.psi.sub(6, deepcopy=True).vector().get_local()).max())


for Qv in [float(q) for q in options.Q_list.split(",") if q]:
    flat(options.v_start * v0_c, Qv)
    good = fsp.psi.vector().get_local().copy()
    rows = []
    v, dvv = options.v_start * v0_c, 0.02 * v0_c
    while dvv > 2e-5 * v0_c:
        fsp.psi.vector()[:] = good
        fp.set_inflow(v + dvv)
        ok = True
        try:
            newton(F_src)
            lam = eig(F_src)[0].eigenvalue
            z_new = zmax()
            # a jump to a far branch also counts as beyond the fold
            if lam.real > 0 or (rows and abs(z_new - rows[-1]["z_max"]) > 0.5 + 2 * abs(rows[-1]["z_max"])):
                ok = False
        except RuntimeError:
            ok = False
        if ok:
            v += dvv
            good = fsp.psi.vector().get_local().copy()
            rows.append(dict(v0=v, z_max=z_new, phi_max=phimax(), lam=lam.real))
            print(f"  Q = {Qv}: v0/v0_c = {v / v0_c:.5f}, z_max = {z_new:+.4f}, phi_max = {rows[-1]['phi_max']:.4f}, "
                  f"lambda = {lam.real:+.3e}", flush=True)
        else:
            dvv *= 0.5
    xf, shift = fold_prediction(Qv)
    run = dict(Q=Qv, v_fold=v, one_minus_vfold=1 - v / v0_c, z_at_fold=rows[-1]["z_max"] if rows else None,
               predicted_one_minus_vfold=shift, predicted_x_fold=xf, rows=rows)
    res["runs"].append(run)
    print(f"Q = {Qv}: fold at v0/v0_c = {v / v0_c:.5f} (1 - v_f/v0_c = {1 - v / v0_c:.4f}; WNL {shift}), z at fold "
          f"{run['z_at_fold']}  [{time.time() - t0:.0f} s]", flush=True)
    json.dump(res, open(os.path.join(out_dir, "imperfection.json"), "w"), indent=1)
print("IMPERFECTION:", json.dumps({k: v for k, v in res.items() if k != "runs"}), flush=True)
