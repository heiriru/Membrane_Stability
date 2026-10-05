'''
Weakly nonlinear (centre-manifold) reduction of the flow-driven buckling at the threshold v0_c: Landau coefficient and
imperfection law, from the direct and adjoint critical modes of IRENE's equations.

Dynamics B dpsi/dt = -F(psi; v0, t) (t: contact slope of the PI, imperfection). Around the flat state psi0 at v0_c
    psi = psi0 + A x1 + A^2 x2 + ...,   F(psi0 + d) = J d + 1/2 D2(d, d) + 1/6 D3(d, d, d) + ...,
with J x1 = 0 (critical mode, normalized as in flow_branch.py: max |z| = 1, so A = <z, z_c>/<z_c, z_c>), the adjoint
mode y (J^T y = 0, y^T B x1 = 1), and the second-order field  J x2 = -1/2 D2(x1, x1)  (x2 has no x1 component; for the
up-down symmetric flat state y^T D2(x1, x1) = 0). Projection on y gives the amplitude equation
    dA/dt = lambda_v (v0 - v0_c) A + N3 A^3 + H t,
    lambda_v = d lambda / d v0 (finite differences of the leading eigenvalue),
    N3 = -y^T [ D2(x1, x2) + 1/6 D3(x1, x1, x1) ]       (N3 > 0: subcritical),
    H  = -y^T dF/dt                                       (the slope enters through the boundary condition on the PI:
                                                           dF/dt = F(psi0 + t lifting; t) / t on the free rows).
Predictions:
    perfect branch      v0 / v0_c = 1 - (N3 / (lambda_v v0_c)) A^2,
    fold (imperfect)    A_f^3 = H t / (2 N3),   1 - v_fold / v0_c = 3 (N3 / (lambda_v v0_c)) |H t / (2 N3)|^(2/3)
                        (Koiter's 2/3 law with its prefactor).

run with:
STABILITY_PARAMS=zeta=1,pi_height=1 python3 flow_landau.py square_b [mesh] [out] --critical_dir [dir]
'''
import irene_paths  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
from fenics import (Constant, Function, PETScMatrix, PETScVector, TestFunction, TrialFunction, as_backend_type,
                    assemble, assign, derivative, interpolate)
from petsc4py import PETSc

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--critical_dir", required=True)
parser.add_argument("--dv0_rel", type=float, default=1e-3)
parser.add_argument("--dt_slope", type=float, default=1e-4)
parser.add_argument("--nev", type=int, default=8)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
t_wall = time.time()

crit = json.load(open(os.path.join(options.critical_dir, "critical.json")))
v0_c = crit["v0_c"]
cm = np.load(os.path.join(options.critical_dir, "critical_mode.npz"))
ref = Function(fsp.Q)
ref.vector()[:] = cm["x_real"]
z_ref = ref.sub(3, deepcopy=True).vector().get_local()

omega_unit = fp.vp.omega_circle.vector().get_local() / prm["omega_circle_const"]


def set_slope(t):
    fp.vp.omega_circle.vector()[:] = omega_unit * t


set_slope(0.0)


def steady(v0, n=8):
    fsp.psi.vector().zero()
    assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
    for v in np.linspace(0, v0, n + 1)[1:]:
        fp.set_inflow(v)
        fp.newton()


def leading():
    return max(fp.eigenpairs(target=0.0, nev=options.nev), key=lambda p: p.eigenvalue.real)


# lambda_v by central differences
dv = options.dv0_rel * v0_c
lam_pm = []
for v in (v0_c - dv, v0_c + dv):
    steady(v)
    lam_pm.append(leading().eigenvalue.real)
lambda_v = (lam_pm[1] - lam_pm[0]) / (2 * dv)

steady(v0_c)
psi0 = fsp.psi.vector().get_local().copy()
lead = leading()
lam_c = lead.eigenvalue.real
print(f"v0_c = {v0_c:.6f}: lambda = {lam_c:+.3e}, lambda_v = {lambda_v:+.6e}  [{time.time() - t_wall:.0f} s]",
      flush=True)

# direct mode x1 (normalized as in flow_branch.py), adjoint mode y (y^T B x1 = 1)
X1 = Function(fsp.Q)
X1.vector()[:] = lead.x_real
zx = X1.sub(3, deepcopy=True).vector().get_local()
X1.vector()[:] *= np.sign(np.dot(zx, z_ref)) / np.abs(zx).max()
x1 = X1.vector().get_local().copy()
A_mat, B_mat = fp.eigenproblem()
Am, Bm = as_backend_type(A_mat).mat(), as_backend_type(B_mat).mat()
At, Bt = Am.copy(), Bm.copy()
At.transpose()
Bt.transpose()
adj = min(ls.solve_eigenproblem(At, Bt, target=lam_c, nev=4), key=lambda p: abs(p.eigenvalue - lam_c))
bc_dofs = np.array(sorted(set().union(*[set(bc.get_boundary_values().keys()) for bc in fp.bcs_hom])), dtype=int)
y = adj.x_real.copy()
y[bc_dofs] = 0.0
xv, Bx = Am.createVecs()
xv.setArray(x1)
Bm.mult(xv, Bx)
Bx1 = Bx.getArray().copy()
y /= np.dot(y, Bx1)
print(f"adjoint eigenvalue {adj.eigenvalue.real:+.3e}", flush=True)

X2 = Function(fsp.Q)
test = TestFunction(fsp.Q)


def vec(form):
    b = assemble(form).get_local()
    b[bc_dofs] = 0.0
    return b


def D2(a, b):
    return vec(derivative(derivative(fp.F_used, fsp.psi, a), fsp.psi, b))


fsp.psi.vector()[:] = psi0
d2_11 = D2(X1, X1)
proj_d2 = float(np.dot(y, d2_11))
print(f"y^T D2(x1, x1) = {proj_d2:+.3e} (0 by symmetry), |D2(x1, x1)| = {np.linalg.norm(d2_11):.3e}  "
      f"[{time.time() - t_wall:.0f} s]", flush=True)

# second-order field: J x2 = -1/2 D2(x1, x1) + (projection), then remove the x1 component
J = PETScMatrix()
assemble(derivative(fp.F_used, fsp.psi, TrialFunction(fsp.Q)), tensor=J)
for bc in fp.bcs_hom:
    bc.apply(J)
rhs_np = -0.5 * d2_11 + 0.5 * proj_d2 * Bx1
ksp = PETSc.KSP().create()
ksp.setOperators(as_backend_type(J).mat())
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
rhs, sol = as_backend_type(J).mat().createVecs()
rhs.setArray(rhs_np)
ksp.solve(rhs, sol)
x2 = sol.getArray().copy()
xv.setArray(x2)
Bm.mult(xv, Bx)
x2 -= np.dot(y, Bx.getArray()) * x1
X2.vector()[:] = x2
z2 = X2.sub(3, deepcopy=True).vector().get_local()
print(f"x2: max|z2| = {np.abs(z2).max():.3e} (0 by symmetry), max|x2| = {np.abs(x2).max():.3e}", flush=True)

fsp.psi.vector()[:] = psi0
d2_12 = D2(X1, X2)
d3_111 = vec(derivative(derivative(derivative(fp.F_used, fsp.psi, X1), fsp.psi, X1), fsp.psi, X1))
N3 = -float(np.dot(y, d2_12 + d3_111 / 6.0))
parts = dict(from_x2=-float(np.dot(y, d2_12)), from_cubic=-float(np.dot(y, d3_111)) / 6.0)
coef = N3 / (lambda_v * v0_c)
print(f"N3 = {N3:+.6e} ({parts}); N3 / (lambda_v v0_c) = {coef:+.6e}  [{time.time() - t_wall:.0f} s]", flush=True)

# imperfection: H = -y^T dF/dt, with the slope entering through the BCs on the PI (lifting) and the forms
r = []
for t in (options.dt_slope, -options.dt_slope):
    set_slope(t)
    fsp.psi.vector()[:] = psi0
    for bc in fp.bcs:
        bc.apply(fsp.psi.vector())
    r.append(vec(fp.F_used))
set_slope(0.0)
fsp.psi.vector()[:] = psi0
r0 = vec(fp.F_used)
dFdt = (r[0] - r[1]) / (2 * options.dt_slope)
H = -float(np.dot(y, dFdt))
print(f"H = {H:+.6e} (residual of psi0 at t = 0: {np.linalg.norm(r0):.1e})", flush=True)


def fold(t):
    Af = np.cbrt(H * t / (2 * N3))
    return float(Af), float(3 * coef * Af ** 2)


ts = [-0.01, -0.03, -0.1, -0.3]
pred = {f"{t:g}": dict(zip(("A_fold", "one_minus_vfold_over_v0c"), fold(t))) for t in ts}
for t in ts:
    print(f"t = {t:+.2f}: predicted fold A_f = {pred[f'{t:g}']['A_fold']:+.3f}, 1 - v_fold/v0_c = "
          f"{pred[f'{t:g}']['one_minus_vfold_over_v0c']:.4f}", flush=True)
res = dict(v0_c=v0_c, lambda_c=lam_c, lambda_v=lambda_v, N3=N3, N3_parts=parts, coef_perfect_branch=coef,
           y_D2_x1x1=proj_d2, max_abs_z2=float(np.abs(z2).max()), H=H,
           koiter_prefactor=float(3 * coef * abs(H / (2 * N3)) ** (2.0 / 3.0)), fold_predictions=pred,
           pi_height=fp.PI_HEIGHT, mesh=rarg.args.input_directory, wall_time=time.time() - t_wall)
json.dump(res, open(os.path.join(out_dir, "landau.json"), "w"), indent=1)
print("LANDAU:", json.dumps(res), flush=True)
