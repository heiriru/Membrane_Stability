'''
Interface between IRENE's steady problem with flows (steady_state/flow, fields v, w, sigma, z, omega, mu) and the linear
stability analysis.

IRENE's time-dependent equations (Wörthmüller et al. 2025, Eqs. (1)-(4)) are
    grad_i v^i - 2 H w = 0,
    rho (d_t v^i + ...) = grad^i sigma + viscous terms,
    rho (d_t w + ...)   = 2 kappa (Delta_LB H - 2H(H^2 - K)) + 2 sigma H + viscous terms,
    d_t z = w (N^3 - N^i d_i z).
IRENE's steady residual F = F_v + F_w + F_sigma + F_z + F_omega + F_mu + F_N is written such that the time-dependent
problem reads  m(d_t psi, test) + F(psi; test) = 0  with the mass form
    m = rho <v^i, nu_v_i> + rho <w, nu_w> + <z, nu_z>      (all with the area element sqrt(g) of the steady state);
sigma (Lagrange multiplier of the incompressibility), omega = grad z and mu = H have no time derivative.
The linearization around a steady state gives A x = lambda B x with A = -dF/dpsi, B = m (linear_stability.py).
'''
import importlib
import os

import dolfin
import numpy as np
from fenics import DirichletBC, derivative, NonlinearVariationalProblem, NonlinearVariationalSolver, split

# as in IRENE's solve.py: the automatic quadrature degree of these nonlinear forms is too high
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

import function_spaces as fsp
import switch_problem as swi
import parameters.read.solution as rpam

# parameter overrides from the environment, e.g. STABILITY_PARAMS="rho=0,eta=1,zeta=1,sigma_r_const=0.0025". They must
# be applied before IRENE's variational problem is imported, because IRENE writes the parameters into the forms as
# numbers. 'zeta' (not an IRENE parameter) is the normal friction coefficient, see below.
rpam.parameters.setdefault("zeta", 0.0)
for _item in filter(None, os.environ.get("STABILITY_PARAMS", "").split(",")):
    _key, _value = _item.split("=")
    rpam.parameters[_key.strip()] = float(_value)

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

import differential_geometry.manifold.geometry as geo

RHO = rpam.parameters["rho"]
ZETA = rpam.parameters["zeta"]

# exact_walls = 1: impose n.v = v^2 = 0 on the straight walls (top, bottom) as a Dirichlet BC on the v^2 component,
# instead of relying on IRENE's penalty term (alpha / h) <n.v n.nu_v> only, whose error does not scale with eta and
# makes flow-driven thresholds depend on alpha (see README). The penalty term is then identically zero on the walls.
bcs = list(vp.bcs)
if rpam.parameters.get("exact_walls", 0.0):
    from fenics import Constant
    bcs.append(DirichletBC(fsp.Q.sub(0).sub(1), Constant(0.0), rmsh.boundary_tb))

# z_outer: height of the outer boundary. 0 (IRENE, default): z = 0 (and w = 0) on the whole square; 1: only on the
# inflow and outflow sides (l, r), the walls (t, b) are free edges; 2: only on the inflow side (l). On a free edge the
# slope n.grad z = 0 is still imposed (IRENE's penalty) and the natural condition is zero vertical shear force,
# 2 kappa n.grad H = 0 (IRENE's F_w contains the boundary term 2 kappa n.grad(mu) nu_w on the whole square, which
# makes the weak form need a Dirichlet condition on w there: it is removed below on the free edges).
Z_OUTER = int(rpam.parameters.get("z_outer", 0))
if Z_OUTER:
    _fixed = {1: f"near(x[0], 0.0) || near(x[0], {rmsh.parameters['L']})", 2: "near(x[0], 0.0)"}[Z_OUTER]
    bcs = [bc for bc in bcs if bc not in (vp.bc_w_square, vp.bc_z_square)]
    bcs += [DirichletBC(fsp.Q.sub(1), vp.w_square, _fixed), DirichletBC(fsp.Q.sub(3), vp.z_square, _fixed)]

# pi_height: vertical condition of the PI. 0 (IRENE, default): z free on the rim and no equation for the height of the
# PI: IRENE's F_w keeps the boundary term 2 kappa n.grad(mu) nu_w of the integration by parts on the circle, so the
# discrete problem does not impose zero net vertical force on the PI (the PI is then subject to a vertical force, see
# README). 1: clamped height, z = z_pi (default 0) and w = 0 on the rim (PI anchored vertically). 2: rigid PI of free
# height (force-free): z' = h' and w' = dh'/dt on the whole rim (tied degrees of freedom) and zero net vertical force
# (the circle term of F_w is removed, the row of the tied w is the net vertical force on the PI). Option 2 is
# implemented for the linear stability of flat steady states (omega_circle = 0, where z* = 0 satisfies both).
PI_HEIGHT = int(rpam.parameters.get("pi_height", 0))
tie_groups = []
from fenics import Constant as _C
z_pi_const = _C(rpam.parameters.get("z_pi", 0.0))   # height of the clamped PI (pi_height = 1), adjustable
if PI_HEIGHT == 1:
    from fenics import Constant
    bcs += [DirichletBC(fsp.Q.sub(3), z_pi_const, rmsh.boundary_circle),
            DirichletBC(fsp.Q.sub(1), Constant(0.0), rmsh.boundary_circle)]
elif PI_HEIGHT == 2:
    if rpam.parameters["omega_circle_const"] != 0:
        raise ValueError("pi_height = 2 is implemented for flat steady states only (omega_circle_const = 0)")
    from fenics import Constant
    for _sub in (3, 1):   # z, w
        tie_groups.append(np.array(sorted(DirichletBC(fsp.Q.sub(_sub), Constant(0.0),
                                                      rmsh.boundary_circle).get_boundary_values())))

bcs_hom = []
for bc in bcs:
    bc_h = DirichletBC(bc)
    bc_h.homogenize()
    bcs_hom.append(bc_h)

import differential_geometry.boundary.geometry as bgeo
import ufl

_i, _j, _k = ufl.indices(3)
# IRENE's F_v subtracts the viscous traction term 2 eta (d.n).nu on the walls (tb), on the PI circle and (y-component)
# on the outflow (r): there the tangential velocity then has no boundary condition ("open" boundary). This is fine for
# the steady solver, but it is not dissipative: the linearized in-plane dynamics has spurious growing modes, even at
# rest. Adding these terms back makes the tangential viscous traction vanish (free slip on the walls and on the PI, where
# n.v = 0 is imposed by IRENE's penalty; traction-free outflow with sigma = sigma_r), which is dissipative.
_d = geo.d_c(fsp.v, fsp.w, fsp.omega)
_g = geo.g(fsp.omega)
outflow_correction = 2.0 * rpam.parameters['eta'] * (
    (_d[_i, 1] * _g[_i, _k] * (bgeo.n_lr(fsp.omega))[_k] * fsp.nu_v[1]) * bgeo.sqrt_deth_lr(fsp.omega) * rmsh.ds_r
    + (_d[_i, _j] * _g[_i, _k] * (bgeo.n_tb(fsp.omega))[_k] * fsp.nu_v[_j]) * bgeo.sqrt_deth_tb(fsp.omega) * (rmsh.ds_t + rmsh.ds_b)
    + (_d[_i, _j] * _g[_i, _k] * (bgeo.n_circle(fsp.omega))[_k] * fsp.nu_v[_j])
    * bgeo.sqrt_deth_circle(fsp.omega, rmsh.parameters["c_r"][:2]) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_circle)

# Normal friction zeta w (drag of the surrounding solvent, not in IRENE's model): rho D_t w = f_n - zeta w. IRENE's
# model has no dissipation of the normal motion of a flat membrane, so for rho -> 0 (lipid membranes, Re ~ 1e-9) the
# linearized dynamics is only well posed with zeta > 0. Steady states with w = 0 (e.g. flat, or any state with
# v.grad z = 0) do not depend on zeta, and neither does the location of a real (divergence) instability, where A is
# singular independently of the mass/friction.
friction = ZETA * fsp.w * fsp.nu_w * geo.sqrt_detg(fsp.omega) * rmsh.dx

# In-plane friction with the surrounding fluid (not in IRENE's model), relative to the far-field velocity v0 e_x of the
# membrane (the PI is held fixed, the membrane and the fluid around it move past it): b (v - v0 e_x). It screens the
# in-plane flow disturbance of the PI beyond ell_b = sqrt(eta / b) (Brinkman / Evans-Sackmann; plays the role of the
# Saffman-Delbrueck cut-off of the 2D Stokes paradox). Parameter 'b_friction' (default 0: IRENE's model).
B_FRICTION = rpam.parameters.get("b_friction", 0.0)
from fenics import Constant as _Constant
v_far = _Constant((0.0, 0.0))
if B_FRICTION:
    friction = friction + B_FRICTION * (fsp.v[0] - v_far[0]) * fsp.nu_v[0] * geo.sqrt_detg(fsp.omega) * rmsh.dx \
        + B_FRICTION * (fsp.v[1] - v_far[1]) * fsp.nu_v[1] * geo.sqrt_detg(fsp.omega) * rmsh.dx

if Z_OUTER:
    _n = bgeo.n_tb(fsp.omega)
    free_edge_correction = -2.0 * rpam.parameters['kappa'] * (_n[_i] * fsp.mu.dx(_i) * fsp.nu_w) \
        * bgeo.sqrt_deth_tb(fsp.omega) * rmsh.ds_tb
    if Z_OUTER == 2:
        _n = bgeo.n_lr(fsp.omega)
        free_edge_correction += -2.0 * rpam.parameters['kappa'] * (_n[_i] * fsp.mu.dx(_i) * fsp.nu_w) \
            * bgeo.sqrt_deth_lr(fsp.omega) * rmsh.ds_r
    friction = friction + free_edge_correction

if PI_HEIGHT == 2:
    friction = friction - 2.0 * rpam.parameters['kappa'] * (
        (bgeo.n_circle(fsp.omega))[_i] * fsp.mu.dx(_i) * fsp.nu_w) \
        * bgeo.sqrt_deth_circle(fsp.omega, rmsh.parameters["c_r"][:2]) * (1.0 / rmsh.parameters["r"]) * rmsh.ds_circle

OUTFLOW = "traction_free"
F_used = vp.F + outflow_correction + friction


def set_outflow(kind):
    '''"traction_free" (default: free slip on walls and PI, traction-free outflow; dissipative) or "irene" (original)'''
    global OUTFLOW, F_used
    OUTFLOW = kind
    F_used = (vp.F + outflow_correction if kind == "traction_free" else vp.F) + friction
_, _, _, _, omega_star, _ = split(fsp.psi)
sqrt_g = geo.sqrt_detg(omega_star)


def mass(trial, test):
    v, w, _, z, _, _ = split(trial)
    nu_v, nu_w, _, nu_z, _, _ = split(test)
    return (RHO * (v[0] * nu_v[0] + v[1] * nu_v[1]) + RHO * w * nu_w + z * nu_z) * sqrt_g * rmsh.dx


_v_l_unit = vp.v_l.vector().get_local().copy()
_v_l_const = rpam.parameters["v_l_const"]


def set_inflow(v0):
    '''inflow velocity v^1 = v0 on the left boundary (IRENE's v_l, scaled)'''
    v_far.assign(_Constant((v0, 0.0)))
    if _v_l_const != 0:
        vp.v_l.vector()[:] = _v_l_unit * (v0 / _v_l_const)
    else:
        vp.v_l.vector()[:] = 0.0
        # v^1 component of the P2 vector space on the left boundary
        from fenics import Constant, interpolate
        vp.v_l.assign(interpolate(Constant((v0, 0.0)), vp.v_l.function_space()))


def newton(tol=1e-10, max_iterations=30):
    problem = NonlinearVariationalProblem(F_used, fsp.psi, bcs, derivative(F_used, fsp.psi, fsp.J_psi))
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters["newton_solver"]
    prm["linear_solver"] = "mumps"
    prm["absolute_tolerance"] = tol
    prm["relative_tolerance"] = 1e-12
    prm["maximum_iterations"] = max_iterations
    prm["report"] = False
    return solver.solve()


def eigenproblem():
    return ls_module().assemble_eigenproblem(F_used, fsp.psi, bcs_hom, mass)


def eigenpairs(target=0.0, nev=12):
    '''eigenpairs of the linearization around fsp.psi (with the tied PI degrees of freedom for pi_height = 2)'''
    ls = ls_module()
    A, B = eigenproblem()
    if not tie_groups:
        return ls.solve_eigenproblem(A, B, target=target, nev=nev)
    A_r, B_r, P = ls.tie_dofs(A, B, tie_groups)
    pairs = ls.solve_eigenproblem(A_r, B_r, target=target, nev=nev)
    for p in pairs:
        p.x_real, p.x_imag = P @ p.x_real, P @ p.x_imag
    return pairs


def ls_module():
    from stability import linear_stability
    return linear_stability
