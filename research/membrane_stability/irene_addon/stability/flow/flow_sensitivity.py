'''
Adjoint eigenmode and sensitivity of the flow-driven buckling to steady forcing (Giannetti & Luchini 2007, Marquet,
Sipp & Jacquin 2008), for the membrane with a PI.

At the steady state psi* (inflow v0) with the leading eigenpair A x = lambda B x:
    adjoint mode      A^T y = lambda B^T y,   normalized with y^T B x = 1,
    wavemaker         |x| |y| (pointwise, z components): where a local feedback changes lambda most,
    base-state gradient  d lambda / d psi* = y^T (dA/dpsi* - lambda dB/dpsi*) x  (second derivative of the residual),
    sensitivity to a steady forcing f added to the equations (F(psi) = <f, test>):
        d lambda = <S, f>,   S = J^{-T} (d lambda / d psi*),   J = dF/dpsi (with the homogeneous BCs);
    S has one component per equation: S_v (in-plane force density), S_w (normal force density), S_sigma (source of
    area), ... As a FE function, S_v(x0) . e is the change of lambda per unit point force e applied at x0.
    The threshold moves by  d v0_c = -d lambda / (d lambda / d v0)  (the derivative is computed by finite differences).
The prediction is checked against a direct computation with a small localized in-plane force (--check).

run with:
STABILITY_PARAMS=zeta=1,omega_circle_const=0,pi_height=1 python3 flow_sensitivity.py square_b [mesh] [out] --v0 0.27
'''
import irene_paths  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
import ufl
from fenics import (Constant, Expression, Function, PETScMatrix, PETScVector, TestFunction, as_backend_type, assemble,
                    assign, derivative, exp, interpolate, split)
from petsc4py import PETSc
from slepc4py import SLEPc

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--v0", type=float, required=True)
parser.add_argument("--dv0", type=float, default=None, help="step for d lambda / d v0 (default 1e-3 v0)")
parser.add_argument("--nev", type=int, default=8)
parser.add_argument("--check", action="store_true", help="verify the forcing sensitivity with a direct computation")
parser.add_argument("--check_eps", type=float, default=1e-5)
parser.add_argument("--check_x0", default="-5,0", help="position of the check force relative to the PI centre")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
if fp.tie_groups:
    raise ValueError("use pi_height = 0 or 1 (the tied force-free PI is not implemented here)")
c_r = fp.rmsh.parameters["c_r"]
t0 = time.time()


def steady(v0, n=6):
    fsp.psi.vector().zero()
    assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
    for v in np.linspace(0, v0, n + 1)[1:]:
        fp.set_inflow(v)
        fp.newton()


def leading_pair():
    pairs = fp.eigenpairs(target=0.0, nev=options.nev)
    return max(pairs, key=lambda p: p.eigenvalue.real)


# derivative of the leading eigenvalue with respect to v0 (finite differences)
dv0 = options.dv0 or 1e-3 * options.v0
lam_pm = []
for v in (options.v0 - dv0, options.v0 + dv0):
    steady(v)
    lam_pm.append(leading_pair().eigenvalue.real)
dlam_dv0 = (lam_pm[1] - lam_pm[0]) / (2 * dv0)

steady(options.v0)
psi_star = fsp.psi.vector().get_local().copy()
A, B = fp.eigenproblem()
direct = leading_pair()
lam = direct.eigenvalue
if abs(lam.imag) > 1e-10 * abs(lam):
    raise ValueError("the leading eigenvalue is complex: this script handles real modes only")
lam = lam.real
print(f"v0 = {options.v0}: lambda = {lam:+.6e}, d lambda / d v0 = {dlam_dv0:+.6e}  [{time.time() - t0:.0f} s]",
      flush=True)

# adjoint eigenproblem with the transposed matrices
Am, Bm = as_backend_type(A).mat(), as_backend_type(B).mat()
At, Bt = Am.copy(), Bm.copy()
At.transpose()
Bt.transpose()
adj = min(ls.solve_eigenproblem(At, Bt, target=lam, nev=4), key=lambda p: abs(p.eigenvalue - lam))
print(f"adjoint eigenvalue {adj.eigenvalue.real:+.6e} (direct {lam:+.6e}), residual {adj.residual:.1e}", flush=True)
x = direct.x_real.copy()
y = adj.x_real.copy()
bc_dofs = np.array(sorted(set().union(*[set(bc.get_boundary_values().keys()) for bc in fp.bcs_hom])), dtype=int)
y[bc_dofs] = 0.0
xv, yv, Bx = Am.createVecs()[0], Am.createVecs()[0], Am.createVecs()[0]
xv.setArray(x)
Bm.mult(xv, Bx)
yBx = float(np.dot(y, Bx.getArray()))
y /= yBx
X, Y = Function(fsp.Q), Function(fsp.Q)
X.vector()[:] = x
Y.vector()[:] = y
xz, yz = X.sub(3, deepcopy=True), Y.sub(3, deepcopy=True)
# direct and adjoint z-components: at a flat base state the normal problem is self-adjoint, so y_z ~ x_z
cos = float(np.dot(xz.vector().get_local(), yz.vector().get_local()) /
            (np.linalg.norm(xz.vector().get_local()) * np.linalg.norm(yz.vector().get_local())))
print(f"cosine between direct and adjoint z-components: {cos:+.6f}", flush=True)

# gradient of lambda with respect to the base state
test = TestFunction(fsp.Q)
J_x = derivative(fp.F_used, fsp.psi, X)                  # dF/dpsi [x], a 1-form in the test function
yJx = ufl.action(J_x, Y)                                  # y^T (dF/dpsi) x, a functional of psi*
yBx_form = fp.mass(X, Y)                                  # y^T B x, depends on psi* through sqrt(g)
grad_lam = -assemble(derivative(yJx, fsp.psi, test)).get_local() - lam * assemble(
    derivative(yBx_form, fsp.psi, test)).get_local()

# sensitivity to steady forcing: S = J^{-T} grad_lam, J = dF/dpsi with the homogeneous BCs (identity rows)
J = PETScMatrix()
assemble(derivative(fp.F_used, fsp.psi, ufl.TrialFunction(fsp.Q)), tensor=J)
for bc in fp.bcs_hom:
    bc.apply(J)
Jt = as_backend_type(J).mat().copy()
Jt.transpose()
rhs, S_vec = Jt.createVecs()
g = grad_lam.copy()
g[bc_dofs] = 0.0
rhs.setArray(g)
ksp = PETSc.KSP().create()
ksp.setOperators(Jt)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
ksp.solve(rhs, S_vec)
S = Function(fsp.Q)
S.vector()[:] = S_vec.getArray()
S_v, S_w, S_sigma, S_z, _, _ = S.split(deepcopy=True)
print(f"sensitivity computed [{time.time() - t0:.0f} s]", flush=True)

result = dict(v0=options.v0, eigenvalue=lam, dlam_dv0=dlam_dv0, adjoint_eigenvalue=adj.eigenvalue.real,
              adjoint_residual=adj.residual, cos_direct_adjoint_z=cos, pi_height=fp.PI_HEIGHT,
              omega_circle=prm["omega_circle_const"], mesh=rarg.args.input_directory)

if options.check:
    # direct computation: a small Gaussian in-plane force density eps * e_x * G(x - x0) (width 1) added to the
    # equations (F -> F - <f, nu_v>) and the change of lambda, compared with <S_v, f>
    dx0 = [float(s) for s in options.check_x0.split(",")]
    x0 = (c_r[0] + dx0[0], c_r[1] + dx0[1])
    G = Expression("exp(-((x[0]-a)*(x[0]-a) + (x[1]-b)*(x[1]-b)) / (2*w*w)) / (2*pi*w*w)", a=x0[0], b=x0[1], w=1.0,
                   degree=4)
    eps = Constant(0.0)
    forcing = eps * G * fsp.nu_v[0] * fp.rmsh.dx
    F_saved = fp.F_used
    fp.F_used = F_saved - forcing
    predicted = options.check_eps * assemble(G * S_v[0] * fp.rmsh.dx)
    lams = []
    for e in (-options.check_eps, options.check_eps):
        eps.assign(e)
        fsp.psi.vector()[:] = psi_star
        fp.newton()
        lams.append(leading_pair().eigenvalue.real)
    measured = 0.5 * (lams[1] - lams[0])
    fp.F_used = F_saved
    result.update(check_x0=list(x0), check_eps=options.check_eps, check_predicted=predicted, check_measured=measured,
                  check_rel_error=abs(predicted - measured) / abs(measured))
    print(f"CHECK: d lambda predicted {predicted:+.6e}, measured {measured:+.6e}, relative error "
          f"{result['check_rel_error']:.2e}", flush=True)

# fields for the figures (P1 at the vertices)
from fenics import FunctionSpace, project, vertex_to_dof_map
P1 = FunctionSpace(fsp.Q.mesh(), "P", 1)
fields = dict(x_z=project(xz, P1), y_z=project(yz, P1), wavemaker=project(abs(xz) * abs(yz), P1),
              S_vx=project(S_v[0], P1), S_vy=project(S_v[1], P1), S_w=project(S_w, P1), S_sigma=project(S_sigma, P1))
c1 = P1.tabulate_dof_coordinates()
np.savez(os.path.join(out_dir, "sensitivity.npz"), x=c1[:, 0], y=c1[:, 1],
         triangles=vertex_to_dof_map(P1)[fsp.Q.mesh().cells()], v0=options.v0, eigenvalue=lam, dlam_dv0=dlam_dv0,
         **{k: f.vector().get_local() for k, f in fields.items()})
# threshold shift per unit point force: d v0_c = -S_v . e / (d lambda / d v0)
Sv = np.stack([fields["S_vx"].vector().get_local(), fields["S_vy"].vector().get_local()])
result["max_abs_dv0c_per_unit_point_force"] = float(np.abs(Sv).max() / abs(dlam_dv0))
json.dump(result, open(os.path.join(out_dir, "sensitivity.json"), "w"), indent=1)
print("SENSITIVITY:", json.dumps(result), flush=True)
