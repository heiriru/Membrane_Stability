'''
Steady Navier-Stokes equations in the Schaefer-Turek (DFG) benchmark channel, in units of the cylinder diameter:
channel [0, 22] x [0, 4.1], cylinder of diameter 1 centred at (2, 2), parabolic inflow with mean velocity 1 (maximum 1.5).
Re = U_mean D / nu. The benchmark 2D-1 is Re = 20.

Boundary conditions: parabolic inflow (l), no slip on the walls (tb) and on the cylinder, do-nothing at the outflow (r).
'''
from fenics import *
import importlib
import switch_problem as swi
import function_spaces_steady as fsp
import runtime_arguments as rarg

rmsh = importlib.import_module(swi.rmsh)

R = Constant(float(getattr(rarg.args, "Re", 1.0)))
W = fsp.W
up = fsp.up
u, p = split(up)
nu_test, q_test = fsp.nu, fsp.q
J_up = fsp.J_up

v_profile_l = Expression(('4.0 * 1.5 * (x[1] - y0) * (y0 + H - x[1]) / pow(H, 2)', '0'), degree=2,
                         H=rmsh.h, y0=rmsh.y_min)

bc_u_in = DirichletBC(fsp.Q_v, v_profile_l, rmsh.boundary_l)
bc_u_w = DirichletBC(fsp.Q_v, Constant((0.0, 0.0)), rmsh.boundary_tb)
bc_u_cyl = DirichletBC(fsp.Q_v, Constant((0.0, 0.0)), rmsh.boundary_circle)
bcs = [bc_u_in, bc_u_w, bc_u_cyl]

bcs_hom = [DirichletBC(fsp.Q_v, Constant((0.0, 0.0)), boundary)
           for boundary in (rmsh.boundary_l, rmsh.boundary_tb, rmsh.boundary_circle)]

F = (dot(dot(u, nabla_grad(u)), nu_test)
     + inner(grad(u), grad(nu_test)) * (1 / R)
     - div(nu_test) * p
     - q_test * div(u)
     ) * rmsh.dx

J = derivative(F, fsp.up, fsp.J_up)


def mass(trial, test):
    u_trial, _ = split(trial)
    u_test, _ = split(test)
    return inner(u_trial, u_test) * rmsh.dx


def initial_guess():
    up.vector().zero()
    for bc in bcs:
        bc.apply(up.vector())
