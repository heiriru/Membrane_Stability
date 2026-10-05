'''
IRENE's flowing membrane beyond the Monge gauge: a parametric surface X(u, t) over the reference domain (IRENE's
square_b mesh, coordinates u = (u^1, u^2)).

Gauge. IRENE's differential geometry (modules/differential_geometry/manifold/geometry.py) is written for a general
frame e_i (metric g = e e^T, normal, b_ij = n . d_j e_i, Christoffel symbols, covariant derivatives, rate of
deformation); only e, normal and K are gauge-specific. Here the frame is a mixed variable E[i, k] = d_i X_k (the analogue
of IRENE's omega = grad z), and
    e(E) = E,   normal(E) = E_1 x E_2 / |E_1 x E_2|,   K(E) = det(b g^{-1}).
IRENE's boundary geometry (conormals, line elements) is also written with geo.e and geo.g and is reused unchanged: the
boundary points stay on the reference boundary (X is imposed on the outer square and on the PI rim).

Unknowns: v (P2, tangential velocity in the coordinate basis), w (P1, normal velocity), sigma (P1), X (P1, 3
components), E (P1, 2 x 3), mu (P1, mean curvature). Equations (rho = 0, as in the Monge runs):
  - F_sigma, F_v, F_w, F_mu: IRENE's square_b forms, verbatim with omega -> E (and the add-on corrections of
    ../flow/flow_problem.py: dissipative outflow / free slip, normal friction zeta w, optional in-plane friction b);
  - F_E: E = grad X weakly (as F_omega for omega = grad z), and the Nitsche condition n . grad X_3 = 0 on the square;
  - kinematics: d_t X = w n + a^i e_i, where a (the tangential velocity of the parametrization) is a gauge choice.
    "Quasi-Monge" gauge: a minimizes |P_h(w n + a^i e_i)|^2 + delta |a|_g^2 (P_h: horizontal projection): with delta = 0
    the reference points move vertically and the surface stays the Monge graph z(x, y) of IRENE (validation); delta > 0
    regularizes the gauge where the surface turns vertical, so the parametrization can follow overhangs. Optional
    equidistribution: a += D_eq g^ij d_j log sqrt(g), which moves parametrization points towards stretched regions
    (steep walls) to keep them resolved.

Boundary conditions: v = v_l (inflow), v = 0 on the PI; n.v = 0 on the walls (Nitsche); sigma = sigma_r at the
outflow; X = (u, 0) and w = 0 on the square; on the PI rim X = (u, h), w = 0, E = [[1, 0, t r_x], [0, 1, t r_y]] (rigid
horizontal PI at height h with contact slope t).

Vertical force on the PI: the consistent reaction. For a vertical virtual velocity delta V = phi_c e_z (phi_c = 1 on the
rim, 0 outside r = 6), decomposed as nu_w = phi_c n_z, nu_v^i = phi_c g^ij e_j . e_z, the volume part of F_v + F_w
equals minus the stress flux through the rim (the forms are integrations by parts of the force balances).
'''
import importlib

import dolfin
import numpy as np
from fenics import (Constant, DirichletBC, Expression, FiniteElement, Function, FunctionSpace, MixedElement,
                    NonlinearVariationalProblem, NonlinearVariationalSolver, TensorElement, TestFunctions,
                    TrialFunction, VectorElement, as_tensor, as_vector, assemble, assign, cross, derivative,
                    interpolate, project, split, sqrt, triangle)
import ufl

dolfin.parameters["form_compiler"]["quadrature_degree"] = 4
# MUMPS workspace relaxation (percent): the default is too small for the larger remeshed problems (DIVERGED_PC_FAILED)
from petsc4py import PETSc as _PETSc
_PETSc.Options().setValue("mat_mumps_icntl_14", 300)

import parameters.read.solution as rpam
import command as cmd
import os

rpam.parameters.setdefault("zeta", 0.0)
for _item in filter(None, os.environ.get("STABILITY_PARAMS", "").split(",")):
    _key, _value = _item.split("=")
    rpam.parameters[_key.strip()] = float(_value)

rmsh = importlib.import_module("mesh.read.square")
cmd.set_gauge("monge")                       # boundary geometry functions (generic in geo.e, geo.g)
import differential_geometry.manifold.geometry as geo
import differential_geometry.boundary.geometry as bgeo


def _e(E, nu=None):
    return E


def _normal(E, nu=None):
    c = cross(E[0, :], E[1, :])
    return c / sqrt(ufl.dot(c, c))


_ii, _jj, _kk = ufl.indices(3)


def _K(E, nu=None):
    return ufl.det(as_tensor(geo.b(E)[_ii, _kk] * geo.g_c(E)[_kk, _jj], (_ii, _jj)))


geo.e, geo.normal, geo.K = _e, _normal, _K

prm = rpam.parameters
assert prm["rho"] == 0, "rho = 0 only"
KAPPA, ETA, ZETA = prm["kappa"], prm["eta"], prm["zeta"]
B_FRICTION = prm.get("b_friction", 0.0)
DELTA = prm.get("gauge_delta", 0.05)
D_EQ = prm.get("gauge_eq", 0.0)
mesh = importlib.import_module("mesh.load").mesh
c_r = rmsh.parameters["c_r"][:2]
r_pi = rmsh.parameters["r"]

el = MixedElement([VectorElement("P", triangle, 2), FiniteElement("P", triangle, 1), FiniteElement("P", triangle, 1),
                   VectorElement("P", triangle, 1, dim=3), TensorElement("P", triangle, 1, shape=(2, 3)),
                   FiniteElement("P", triangle, 1)])
Q = FunctionSpace(mesh, el)
psi = Function(Q)
J_psi = TrialFunction(Q)
v, w, sigma, X, E, mu = split(psi)
nu_v, nu_w, nu_sigma, nu_X, nu_E, nu_mu = TestFunctions(Q)
i, j, k, l = ufl.indices(4)
dx, ds = rmsh.dx, None
sg = geo.sqrt_detg(E)
n3 = geo.normal(E)
xc = ufl.SpatialCoordinate(mesh)

# ------------------------------------------------------------------------------------------------- force balances
v_far = Constant((0.0, 0.0))


def F_v_vol(nu):
    f = (sigma * geo.g_c(E)[i, j] * geo.Nabla_f(nu, E)[i, j]
         + 2.0 * ETA * geo.d_c(v, w, E)[j, i] * geo.Nabla_f(nu, E)[j, i]) * sg * dx
    if B_FRICTION:
        f += B_FRICTION * ((v[0] - v_far[0]) * nu[0] + (v[1] - v_far[1]) * nu[1]) * sg * dx
    return f


def F_w_vol(nu):
    return (2.0 * KAPPA * (-geo.g_c(E)[i, j] * mu.dx(i) * nu.dx(j) + 2.0 * mu * (mu ** 2 - geo.K(E)) * nu)
            - (2.0 * sigma * mu + 2.0 * ETA * (geo.g_c(E)[i, k] * geo.Nabla_v(v, E)[j, k] * geo.b(E)[i, j]
                                                - 2.0 * w * (2.0 * mu ** 2 - geo.K(E)))) * nu
            + ZETA * w * nu) * sg * dx


# boundary terms of F_v (tension; the tangential viscous tractions vanish on the walls, the PI and the outflow, as
# with ../flow/flow_problem.py's outflow correction; the left edge has Dirichlet v)
F_v_bdry = -((sigma * (bgeo.n_lr(E))[i] * nu_v[i]) * bgeo.sqrt_deth_lr(E) * rmsh.ds_lr
             + (sigma * (bgeo.n_tb(E))[i] * nu_v[i]) * bgeo.sqrt_deth_tb(E) * rmsh.ds_tb
             + (sigma * (bgeo.n_circle(E))[i] * nu_v[i]) * bgeo.sqrt_deth_circle(E, c_r) * (1.0 / r_pi) * rmsh.ds_circle) \
    - 2.0 * ETA * (geo.d_c(v, w, E)[i, j] * geo.g(E)[i, k] * (bgeo.n_lr(E))[k] * nu_v[j]) * bgeo.sqrt_deth_lr(E) * rmsh.ds_l
F_sigma = (geo.Nabla_v(v, E)[i, i] - 2.0 * mu * w) * nu_sigma * sg * dx
F_mu = (geo.H(E) - mu) * nu_mu * sg * dx
Nf = ufl.FacetNormal(mesh)
F_E = (E[i, k] * nu_E[i, k] + X[k] * nu_E[i, k].dx(i)) * dx - X[k] * nu_E[i, k] * Nf[i] * rmsh.ds
alpha_N = prm["alpha"] / rmsh.r_mesh
F_N = alpha_N * (
    ((bgeo.n_tb(E))[i] * geo.g(E)[i, j] * v[j]) * ((bgeo.n_tb(E))[k] * nu_v[k]) * bgeo.sqrt_deth_tb(E) * rmsh.ds_tb
    + ((bgeo.n_lr(E))[i] * E[i, 2]) * ((bgeo.n_lr(E))[k] * geo.g(E)[k, l] * nu_E[l, 2]) * bgeo.sqrt_deth_lr(E) * rmsh.ds_lr
    + ((bgeo.n_tb(E))[i] * E[i, 2]) * ((bgeo.n_tb(E))[k] * geo.g(E)[k, l] * nu_E[l, 2]) * bgeo.sqrt_deth_tb(E) * rmsh.ds_tb
    + (geo.H(E) - mu) * nu_mu * bgeo.sqrt_deth_lr(E) * rmsh.ds_lr
    + (geo.H(E) - mu) * nu_mu * bgeo.sqrt_deth_tb(E) * rmsh.ds_tb
    + (geo.H(E) - mu) * nu_mu * bgeo.sqrt_deth_circle(E, c_r) * (1.0 / r_pi) * rmsh.ds_circle)

# ------------------------------------------------------------------------------------------------------ kinematics
_Ph = [as_vector([E[0, 0], E[0, 1], 0.0]), as_vector([E[1, 0], E[1, 1], 0.0])]
_M = as_tensor([[ufl.dot(_Ph[a], _Ph[b]) for b in range(2)] for a in range(2)]) + DELTA * geo.g(E)
_B = as_vector([ufl.dot(_Ph[a], as_vector([n3[0], n3[1], 0.0])) for a in range(2)])
a_gauge = -w * ufl.dot(ufl.inv(_M), _B)
if D_EQ:
    a_gauge = a_gauge + D_EQ * as_vector([geo.g_c(E)[a, j] * ufl.ln(sg).dx(j) for a in range(2)])
X_velocity = w * n3 + a_gauge[0] * E[0, :] + a_gauge[1] * E[1, :]

F_static = F_v_vol(nu_v) + F_v_bdry + F_w_vol(nu_w) + F_sigma + F_mu + F_E + F_N

# ------------------------------------------------------------------------------------------------ boundary conditions
v_l = Function(Q.sub(0).collapse())
bc_v_l = DirichletBC(Q.sub(0), v_l, rmsh.boundary_l)
bc_v_c = DirichletBC(Q.sub(0), Constant((0.0, 0.0)), rmsh.boundary_circle)
bc_w_sq = DirichletBC(Q.sub(1), Constant(0.0), rmsh.boundary_square)
bc_w_c = DirichletBC(Q.sub(1), Constant(0.0), rmsh.boundary_circle)
bc_s_r = DirichletBC(Q.sub(2), Constant(prm["sigma_r_const"]), rmsh.boundary_r)
bc_X_sq = DirichletBC(Q.sub(3), Expression(("x[0]", "x[1]", "0.0"), degree=1), rmsh.boundary_square)
h_pi = Constant(0.0)
X_rim = Expression(("x[0]", "x[1]", "h"), h=0.0, degree=1)
bc_X_c = DirichletBC(Q.sub(3), X_rim, rmsh.boundary_circle)
t_slope = Constant(prm["omega_circle_const"])
E_rim = Expression((("1.0", "0.0", "t * (x[0] - cx) / sqrt(pow(x[0] - cx, 2) + pow(x[1] - cy, 2))"),
                    ("0.0", "1.0", "t * (x[1] - cy) / sqrt(pow(x[0] - cx, 2) + pow(x[1] - cy, 2))")),
                   t=prm["omega_circle_const"], cx=c_r[0], cy=c_r[1], degree=1)
bc_E_c = DirichletBC(Q.sub(4), E_rim, rmsh.boundary_circle)
# rim_weak = 1: the rim condition on the frame only fixes the tangent plane (contact slope t), not the parametrization:
# E_a . n_rim = 0 (a = 1, 2) weakly (penalty), with n_rim the normal of a surface leaving the rigid PI with slope t.
# Needed after remeshing (param_remesh.py), where the reference coordinates near the rim are no longer the horizontal
# projection, so E_rim = [[1, 0, t r_x], [0, 1, t r_y]] (unit stretch) would be wrong. Same geometry for the original
# parametrization.
RIM_WEAK = int(prm.get("rim_weak", 0))
t_rim = Constant(prm["omega_circle_const"])
_rx = (xc[0] - c_r[0]) / sqrt((xc[0] - c_r[0]) ** 2 + (xc[1] - c_r[1]) ** 2)
_ry = (xc[1] - c_r[1]) / sqrt((xc[0] - c_r[0]) ** 2 + (xc[1] - c_r[1]) ** 2)
n_rim = as_vector([-t_rim * _rx, -t_rim * _ry, 1.0]) / sqrt(1.0 + t_rim ** 2)
if RIM_WEAK:
    F_static = F_static + prm.get("rim_penalty", 10.0) / rmsh.r_mesh * sum(
        ufl.dot(E[a, :], n_rim) * ufl.dot(nu_E[a, :], n_rim) for a in range(2)) * rmsh.ds_circle
    bcs = [bc_v_l, bc_v_c, bc_w_sq, bc_w_c, bc_s_r, bc_X_sq, bc_X_c]
else:
    bcs = [bc_v_l, bc_v_c, bc_w_sq, bc_w_c, bc_s_r, bc_X_sq, bc_X_c, bc_E_c]


def set_inflow(v0):
    v_far.assign(Constant((v0, 0.0)))
    v_l.assign(interpolate(Constant((v0, 0.0)), v_l.function_space()))


def set_h(h):
    h_pi.assign(h)
    X_rim.h = h


def set_slope(t):
    E_rim.t = t
    t_rim.assign(t)


# ------------------------------------------------------------------------------------------------------ vertical force
_rr = sqrt((xc[0] - c_r[0]) ** 2 + (xc[1] - c_r[1]) ** 2)


def _cutoff(a, b):
    s = ufl.max_value(0.0, ufl.min_value(1.0, (_rr - a) / (b - a)))
    return 1.0 - s * s * (3.0 - 2.0 * s)


cutoffs = [_cutoff(a, b) for a, b in ((1.5, 4.0), (2.0, 6.0), (3.0, 8.0))]


def vertical_force():
    '''vertical force of the PI on the membrane (sign: as the flux-based force of ../flow, calibrated in the checks),
    for three cutoffs'''
    out = []
    ez_t = as_vector([geo.g_c(E)[a, j] * E[j, 2] for a in range(2)])
    for phc in cutoffs:
        out.append(assemble(F_v_vol(phc * ez_t) + F_w_vol(phc * n3[2])))
    return out


# --------------------------------------------------------------------------------------------- initial state, output
P1 = FunctionSpace(mesh, "P", 1)


def set_from_monge(z_vertex_values, h, t_slope_value):
    '''X = (u, z) from IRENE P1 nodal values (vertex order of P1.tabulate_dof_coordinates), E = grad X'''
    zf = Function(P1)
    zf.vector()[:] = z_vertex_values
    Xf = project(as_vector([xc[0], xc[1], zf]), Q.sub(3).collapse())
    Ef = project(as_tensor([[1.0, 0.0, zf.dx(0)], [0.0, 1.0, zf.dx(1)]]), Q.sub(4).collapse())
    assign(psi.sub(3), Xf)
    assign(psi.sub(4), Ef)
    assign(psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), Q.sub(2).collapse()))
    set_h(h)
    set_slope(t_slope_value)


def X_vertices():
    Xf = psi.sub(3, deepcopy=True)
    V3 = Xf.function_space()
    vals = Xf.vector().get_local().reshape(-1, 3)         # P1 vector: dofs interleaved per vertex
    xy = V3.tabulate_dof_coordinates().reshape(-1, 3, 2)[:, 0, :]
    return xy, vals
