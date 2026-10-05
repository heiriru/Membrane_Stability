'''
[flow_phi add-on: IRENE's flow problem with mobile, curvature-coupled proteins; see the end of this docstring]

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

Mobile proteins (fields phi, m of the extended function space in this folder). Free energy per area
    f = 2 kappa (H - C phi)^2 + chi/2 phi^2 + eps/2 |grad phi|^2      (spontaneous curvature H0 = C phi)
    - normal force: IRENE's bending force with mu -> mu - H0, i.e. 2 kappa [Delta_LB (H - H0) + 2 (H - H0)(H^2 + H H0 - K)]:
      F_w gets  2 kappa [ g^ij d_i H0 d_j nu_w + 2 (H0 K - mu H0^2) nu_w ] sqrt(g)
      (the boundary terms of the integration by parts vanish where w is imposed: outer square, clamped / tied PI);
    - chemical potential  m = df/dphi - div(eps grad phi) = -4 kappa C (mu - C phi) + chi phi - eps Delta_LB phi;
    - conservation with advection by the membrane flow and diffusion (mobility M_phi):
      d_t phi + div(phi v) = M_phi Delta_LB m   (no diffusive flux through the boundaries; phi = 0 at the inflow when
      phi_inflow_dirichlet = 1, the default: the incoming membrane carries the mean protein density);
    - tangential force of the protein chemical potential on the membrane (Gibbs-Duhem): -phi grad m.
Parameters (STABILITY_PARAMS): C_phi, chi_phi, eps_phi, M_phi, phi_inflow_dirichlet; beta_phi (quartic term
beta/4 phi^4 of the free energy, 0 by default); phi_form (0: the forms above, exact at linear order around phi = 0;
1: consistent nonlinear forms, see the comments at protein_forms); phi_sponge (1: proteins become passive near the
outflow edge, sponge_L, sponge_chi).
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

_i, _j, _k, _l = ufl.indices(4)
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

# ---------------------------------------------------------------------------------------------------------------------
# mobile proteins
C_PHI = rpam.parameters.get("C_phi", 0.0)
from fenics import Constant as _Cst
import ufl as _ufl
chi_phi = _Cst(rpam.parameters.get("chi_phi", 1.0))
beta_phi = _Cst(rpam.parameters.get("beta_phi", 0.0))
eps_phi = rpam.parameters.get("eps_phi", 1.0)
M_phi = rpam.parameters.get("M_phi", 1.0)
# protein forms: "linear" (round 4, exact at linear order around the flat uniform state) or "consistent" (nonlinear
# terms from the free energy, see the module docstring; same linearization around phi = 0)
PHI_FORM = ("linear", "consistent")[int(rpam.parameters.get("phi_form", 0))]    # phi_form = 0 / 1
# outflow treatment of the proteins: "noflux" (natural conditions on the whole edge) or "sponge": over the last
# sponge_L before the outflow edge the proteins become passive (C -> 0, chi -> chi + sponge_chi): they are carried
# out by the flow without feedback on the shape and without demixing, as if the patch continued downstream
# "absorbing" (phi_sponge = 2): the proteins keep their properties but are removed by a sink -gamma s(x) phi over the
# last sponge_L (as if they left through the edge); unlike "sponge", this creates no chemical-potential barrier
# (raising chi in the sponge gives a drift -M grad(chi_eff) back into the patch, which matters for large M)
PHI_OUTFLOW = ("noflux", "sponge", "absorbing")[int(rpam.parameters.get("phi_sponge", 0))]
SPONGE_GAMMA = rpam.parameters.get("sponge_gamma", 0.05)
SPONGE_L = rpam.parameters.get("sponge_L", 15.0)
SPONGE_CHI = rpam.parameters.get("sponge_chi", 1.0)
_kap = rpam.parameters["kappa"]
_om, _mu, _phi, _m = fsp.omega, fsp.mu, fsp.phi, fsp.m
_xc = _ufl.SpatialCoordinate(fsp.Q.mesh())
if PHI_OUTFLOW in ("sponge", "absorbing"):
    _s_lin = _ufl.max_value(0.0, _ufl.min_value(1.0, (_xc[0] - (rmsh.parameters["L"] - SPONGE_L)) / SPONGE_L))
    sponge = _s_lin * _s_lin * (3.0 - 2.0 * _s_lin)          # smoothstep 0 -> 1
else:
    sponge = _Cst(0.0)
if PHI_OUTFLOW == "absorbing":
    C_eff = _Cst(C_PHI)
    chi_eff = chi_phi
    _sink = SPONGE_GAMMA * sponge
else:
    C_eff = C_PHI * (1.0 - sponge)
    chi_eff = chi_phi + SPONGE_CHI * sponge
    _sink = _Cst(0.0)
_H0 = C_eff * _phi
_sg = geo.sqrt_detg(_om)
_gc = geo.g_c(_om)
_b = geo.b(_om)
# bending force correction for the spontaneous curvature H0 = C phi (both forms)
_bend = 2.0 * _kap * (_gc[_i, _j] * _H0.dx(_i) * fsp.nu_w.dx(_j) + 2.0 * (_H0 * geo.K(_om) - _mu * _H0 ** 2) * fsp.nu_w)
# chemical potential m = df/dphi - eps Delta phi
_chem = (_m + 4.0 * _kap * C_eff * (_mu - C_eff * _phi) - chi_eff * _phi - beta_phi * _phi ** 3) * fsp.nu_m \
    - eps_phi * _gc[_i, _j] * _phi.dx(_i) * fsp.nu_m.dx(_j)
_diff = M_phi * _gc[_i, _j] * _m.dx(_i) * fsp.nu_phi.dx(_j) + _sink * _phi * fsp.nu_phi
if PHI_FORM == "linear":
    protein_forms = (
        _bend
        # conservation: div(phi v) = v^i d_i phi + phi (div v), and diffusion of the chemical potential
        + (fsp.v[_i] * _phi.dx(_i) + _phi * geo.Nabla_v(fsp.v, _om)[_i, _i]) * fsp.nu_phi + _diff
        + _chem
        # Gibbs-Duhem force -phi grad m on the membrane (F_v = - int force . nu_v)
        + _phi * _m.dx(_i) * fsp.nu_v[_i]
    ) * _sg * rmsh.dx
else:
    # free energy density W = 2 kappa (H - H0)^2 + g(phi) + eps/2 |grad phi|^2,  g = chi/2 phi^2 + beta/4 phi^4, with
    # the incompressibility multiplier sigma (IRENE's field). Virtual power with phi carried by the material (D phi/Dt
    # = 0 on incompressible motion) gives
    #   normal:     2 kappa [Delta (H - H0) + 2 (H - H0)(H^2 + H H0 - K)] - 2 H (sigma + g) + eps (b^ij phi_i phi_j - H |grad phi|^2)
    #   tangential: grad sigma + m grad phi = div(viscous stress)        (internal force -m grad phi, m = dW/dphi)
    # and the material derivative in the Monge frame, whose points move vertically with dz/dt = w sqrt(1 + |omega|^2):
    #   D phi / Dt = d_t phi + (v^i - a^i) d_i phi,  a^i = g^ij omega_j dz/dt   (tangential velocity of the frame)
    _g_phi = 0.5 * chi_eff * _phi ** 2 + 0.25 * beta_phi * _phi ** 4
    _bup = _ufl.as_tensor(_gc[_i, _k] * _gc[_j, _l] * _b[_k, _l], (_i, _j))
    _grad2 = _gc[_i, _j] * _phi.dx(_i) * _phi.dx(_j)
    _zdot = fsp.w * _ufl.sqrt(1.0 + _om[_i] * _om[_i])
    _a = _ufl.as_tensor(_gc[_i, _j] * _om[_j] * _zdot, (_i,))
    protein_forms = (
        _bend
        + (-2.0 * _mu * _g_phi + eps_phi * (_bup[_i, _j] * _phi.dx(_i) * _phi.dx(_j) - _mu * _grad2)) * fsp.nu_w
        + (fsp.v[_i] - _a[_i]) * _phi.dx(_i) * fsp.nu_phi + _diff
        + _chem
        - _m * _phi.dx(_i) * fsp.nu_v[_i]
    ) * _sg * rmsh.dx
PHI_INFLOW = int(rpam.parameters.get("phi_inflow_dirichlet", 1))
if PHI_INFLOW:
    from fenics import Constant
    _bc_phi = DirichletBC(fsp.Q.sub(6), Constant(0.0), rmsh.boundary_l)
    bcs.append(_bc_phi)
    _h = DirichletBC(_bc_phi)
    _h.homogenize()
    bcs_hom.append(_h)

OUTFLOW = "traction_free"
F_used = vp.F + outflow_correction + friction + protein_forms


def set_outflow(kind):
    '''"traction_free" (default: free slip on walls and PI, traction-free outflow; dissipative) or "irene" (original)'''
    global OUTFLOW, F_used
    OUTFLOW = kind
    F_used = (vp.F + outflow_correction if kind == "traction_free" else vp.F) + friction + protein_forms
_, _, _, _, omega_star, _, _, _ = split(fsp.psi)
sqrt_g = geo.sqrt_detg(omega_star)


def mass(trial, test):
    v, w, _, z, _, _, phi, _ = split(trial)
    nu_v, nu_w, _, nu_z, _, _, nu_phi, _ = split(test)
    return (RHO * (v[0] * nu_v[0] + v[1] * nu_v[1]) + RHO * w * nu_w + z * nu_z + phi * nu_phi) * sqrt_g * rmsh.dx


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
