'''
Steady states w* = (u*, p*) of the variational problem 'vp' (a variational_problem_bc_*_steady module) and derived
quantities (forces on the cylinder, recirculation length).
'''
import numpy as np
from fenics import (NonlinearVariationalProblem, NonlinearVariationalSolver, DirichletBC, Constant, Function,
                    assemble, parameters)

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True


def newton_solve(vp, Re, verbose=False):
    '''solve F(w*) = 0 at Reynolds number Re with Newton's method, starting from the current content of vp.up'''
    vp.R.assign(Re)
    problem = NonlinearVariationalProblem(vp.F, vp.up, vp.bcs, vp.J)
    solver = NonlinearVariationalSolver(problem)
    prm = solver.parameters["newton_solver"]
    prm["absolute_tolerance"] = 1e-10
    prm["relative_tolerance"] = 1e-9
    prm["maximum_iterations"] = 25
    prm["linear_solver"] = "mumps"
    prm["report"] = verbose
    iterations, converged = solver.solve()
    return iterations, converged


def solve_steady(vp, Re, Re_start=None, n_steps=4, verbose=False):
    '''
    Steady state at Reynolds number Re. If Re_start is given, natural continuation in Re from the current solution
    at Re_start; if Newton's method fails, the continuation step is halved.
    '''
    if Re_start is None:
        vp.initial_guess()
        Re_start = min(Re, 5.0)
        newton_solve(vp, Re_start, verbose)
    Re_values = list(np.linspace(Re_start, Re, n_steps + 1)[1:]) if Re != Re_start else [Re]
    Re_current = Re_start
    backup = vp.up.vector().copy()
    while Re_values:
        Re_next = Re_values[0]
        try:
            newton_solve(vp, Re_next, verbose)
            Re_current = Re_next
            backup = vp.up.vector().copy()
            Re_values.pop(0)
        except RuntimeError:
            vp.up.vector()[:] = backup
            if abs(Re_next - Re_current) < 1e-3:
                raise
            Re_values.insert(0, 0.5 * (Re_current + Re_next))
    return vp.up


def force_on_cylinder(vp, rmsh):
    '''
    Force (F_x, F_y) exerted by the fluid on the cylinder, computed from the residual of the weak form tested with a
    function equal to e_x (e_y) on the cylinder and 0 on the other nodes (volume / Babuska-Miller formulation, more
    accurate than the boundary integral of the stress). The force on the body is minus the traction integral with the
    normal pointing out of the fluid, hence the minus sign.
    '''
    W = vp.up.function_space()
    residual = assemble(vp.F)
    force = []
    for value in [(1.0, 0.0), (0.0, 1.0)]:
        indicator = Function(W)
        DirichletBC(W.sub(0), Constant(value), rmsh.boundary_circle).apply(indicator.vector())
        force.append(-residual.inner(indicator.vector()))
    return np.array(force)


def recirculation_length(u, rmsh, x_end=None, n=4000):
    '''length of the recirculation bubble behind the cylinder, measured from its rear point, along y = c_r[1]'''
    x0 = rmsh.c_r[0] + rmsh.r
    x_end = rmsh.x_max - 1e-6 if x_end is None else x_end
    xs = np.linspace(x0 + 1e-6, x_end, n)
    ux = np.array([u(x, rmsh.c_r[1])[0] for x in xs])
    positive = np.where(ux > 0)[0]
    if len(positive) == 0 or positive[0] == 0:
        return 0.0
    i = positive[0]
    x_zero = xs[i - 1] - ux[i - 1] * (xs[i] - xs[i - 1]) / (ux[i] - ux[i - 1])
    return x_zero - x0
