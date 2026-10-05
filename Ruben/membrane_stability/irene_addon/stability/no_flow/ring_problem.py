'''
Interface between IRENE's steady, no-flow ring problem (steady_state/no_flow/variational_problem_bc_ring.py) and the
linear stability analysis (modules/stability/linear_stability.py).

IRENE's residual F = F_z + F_omega + F_mu + F_N is used unchanged. Only the boundary data are made adjustable, so that
the protein height h, the slope at the protein and the surface tension can be swept without re-importing IRENE:
    z = h                 on the protein circle (r = r0)          Dirichlet BC (rebuilt here)
    z = 0                 on the outer circle (r = R)             Dirichlet BC
    dz/dr = tan(alpha)    on the protein circle                   penalty term of IRENE (vp.omega_r is updated)
    dz/dr = 0             on the outer circle                     penalty term of IRENE (vp.omega_R is updated)

Dynamics used for the stability analysis (overdamped normal relaxation, Monge gauge):
    zeta * w = f_n,   dz/dt = w * sqrt(g)
where f_n = 2 kappa (Delta_LB H - 2H (H^2 - K)) + 2 sigma H is the normal elastic force per unit area and zeta a
normal friction coefficient. IRENE's F_z equals <f_n / 2, nu_z sqrt(g)>, hence
    < zeta dz/dt, nu_z > = 2 F_z    ->    mass form  m = (zeta / 2) <z, nu_z>,  residual for m(dz/dt) + F_dyn = 0: F_dyn = -F.
This is a gradient flow of the Helfrich energy: its eigenvalues are real, and their sign (stability) does not depend
on the choice of the positive friction zeta, which only sets the time unit kappa / (zeta r0^4).
'''
import dolfin
import numpy as np
from fenics import (Constant, DirichletBC, Function, assemble, split, sqrt, dx as _dx,
                    NonlinearVariationalProblem, NonlinearVariationalSolver, derivative)

# as in IRENE's steady_state/no_flow/solve.py: the automatic quadrature degree of these nonlinear forms is too high
dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

import function_spaces as fsp
import switch_problem as swi
import importlib

rmsh = importlib.import_module(swi.rmsh)
vp = importlib.import_module(swi.vp)

import differential_geometry.boundary.geometry as bgeo
import differential_geometry.manifold.geometry as geo
import physics.utils as phys
import parameters.read.solution as rpam

KAPPA = rpam.parameters["kappa"]
r0 = rmsh.parameters["r"]
R = rmsh.parameters["R"]
c_r = rmsh.parameters["c_r"][:2]
c_R = rmsh.parameters["c_R"][:2]

h_const = Constant(0.0)
bcs = [DirichletBC(fsp.Q.sub(0), h_const, rmsh.boundary_r),
       DirichletBC(fsp.Q.sub(0), Constant(0.0), rmsh.boundary_R)]
# homogeneous BCs of the perturbation: z' = 0 on both circles (the slope conditions are penalty terms in F, whose
# linearization is automatically homogeneous)
bcs_hom = [DirichletBC(fsp.Q.sub(0), Constant(0.0), rmsh.boundary_r),
           DirichletBC(fsp.Q.sub(0), Constant(0.0), rmsh.boundary_R)]

ZETA = 1.0


def mass(trial, test):
    z_trial, _, _ = split(trial)
    z_test, _, _ = split(test)
    return 0.5 * ZETA * z_trial * z_test * rmsh.dx


F_dynamics = -vp.F


def set_parameters(h=None, slope_r=None, slope_R=None, sigma=None):
    '''update the boundary data / tension used by IRENE's residual (vp.F)'''
    if h is not None:
        h_const.assign(h)
    # IRENE imposes n^i omega_i = omega_r(R), with n the unit (metric) normal pointing out of the domain:
    # n^i omega_i = -z'/sqrt(1 + z'^2) on the inner circle and +z'/sqrt(1 + z'^2) on the outer circle
    if slope_r is not None:
        vp.omega_r.vector()[:] = -slope_r / np.sqrt(1.0 + slope_r ** 2)
    if slope_R is not None:
        vp.omega_R.vector()[:] = slope_R / np.sqrt(1.0 + slope_R ** 2)
    if sigma is not None:
        fsp.sigma.vector()[:] = sigma


def flat_initial_guess():
    fsp.psi.vector().zero()


def newton(max_iterations=40, relaxation=1.0, tol=1e-10):
    J = derivative(vp.F, fsp.psi, fsp.J_psi)
    problem = NonlinearVariationalProblem(vp.F, fsp.psi, bcs, J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters["newton_solver"]
    prm["linear_solver"] = "mumps"
    prm["absolute_tolerance"] = tol
    prm["relative_tolerance"] = 1e-12
    prm["maximum_iterations"] = max_iterations
    prm["relaxation_parameter"] = relaxation
    prm["report"] = False
    prm["error_on_nonconvergence"] = True
    return solver.solve()


def solve_steady(h, h_previous=None, n_substeps=1, max_halvings=4):
    '''
    steady state at protein height h, by continuation from the current state (at h_previous); if Newton fails, the
    step is halved at most max_halvings times (the end of the branch is then reported by a RuntimeError)
    '''
    if h_previous is None:
        targets = [h]
    else:
        targets = list(np.linspace(h_previous, h, n_substeps + 1)[1:])
    min_step = abs(h - h_previous) / max(n_substeps, 1) / 2 ** max_halvings if h_previous is not None else 0.0
    backup = fsp.psi.vector().copy()
    current = h_previous
    while targets:
        target = targets[0]
        set_parameters(h=target)
        try:
            newton()
            backup = fsp.psi.vector().copy()
            current = target
            targets.pop(0)
        except RuntimeError:
            fsp.psi.vector()[:] = backup
            if current is None or abs(target - current) <= min_step * 1.0001:
                set_parameters(h=current if current is not None else h)
                raise
            targets.insert(0, 0.5 * (current + target))
    return fsp.psi


def fields():
    z, omega, mu = fsp.psi.split(deepcopy=True)
    return z, omega, mu


def force_on_protein():
    '''3D force per IRENE's print_out_force_on_boundary_bc_ring (tension + bending line forces on the protein circle)'''
    _, omega, mu = fields()
    integrand = phys.dFdl_sigma_kappa_3d(omega, mu, fsp.sigma, KAPPA, bgeo.n_circle(omega))
    line_element = bgeo.sqrt_deth_circle(omega, c_r) * (1.0 / r0)
    return np.array([assemble(component * line_element * rmsh.ds_r) for component in integrand])


def moment_term():
    '''
    Boundary term 2 kappa H n^i d_i psi of the first variation of the Helfrich energy, for a rigid vertical translation
    of the PI (psi = dh N_3, N_3 = 1/sqrt(1 + |grad z|^2)). It is not included in the line forces (17)-(18) of the PRE /
    IRENE's dFdl_sigma_kappa_3d; F_line - moment_term() reproduces the virtual-work force -dE/dh (check_force_energy.py).
    '''
    _, omega, mu = fields()
    N3 = 1.0 / sqrt(1.0 + omega[0] ** 2 + omega[1] ** 2)
    n = bgeo.n_circle(omega)
    line_element = bgeo.sqrt_deth_circle(omega, c_r) * (1.0 / r0)
    return assemble(2.0 * KAPPA * mu * (n[0] * N3.dx(0) + n[1] * N3.dx(1)) * line_element * rmsh.ds_r)


def energy():
    '''Helfrich energy E = < 2 kappa H^2 + sigma >_Omega (area element sqrt(g)), consistent with IRENE's F_z'''
    _, omega, mu = fields()
    return assemble((2.0 * KAPPA * mu ** 2 + fsp.sigma) * geo.sqrt_detg(omega) * rmsh.dx)


def residual_norm():
    b = assemble(vp.F)
    for bc in bcs_hom:
        bc.apply(b)
    return b.norm("l2")
