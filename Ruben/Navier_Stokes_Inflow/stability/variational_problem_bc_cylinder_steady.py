'''
Steady Navier-Stokes equations for the flow past a cylinder in an (approximately) unbounded domain.
Units: cylinder diameter D = 1, free-stream velocity U = 1, Re = U D / nu.

Boundary conditions
- inflow (l):        u = (1, 0)
- top/bottom (tb):   free slip, u_y = 0 and d(u_x)/dn = 0 (far field)
- cylinder:          no slip, u = 0
- outflow (r):       natural 'do-nothing' condition, (1/Re) du/dn - p n = 0
'''
from fenics import *
import importlib
import switch_problem as swi
import function_spaces_steady as fsp
import runtime_arguments as rarg

rmsh = importlib.import_module(swi.rmsh)

# Reynolds number, change it with R.assign(...)
R = Constant(float(getattr(rarg.args, "Re", 1.0)))
W = fsp.W
up = fsp.up
u, p = split(up)
nu_test, q_test = fsp.nu, fsp.q
J_up = fsp.J_up

U_inf = Constant((1.0, 0.0))

bc_u_in = DirichletBC(fsp.Q_v, U_inf, rmsh.boundary_l)
bc_u_slip = DirichletBC(fsp.Q_v.sub(1), Constant(0.0), rmsh.boundary_tb)
bc_u_cyl = DirichletBC(fsp.Q_v, Constant((0.0, 0.0)), rmsh.boundary_circle)
bcs = [bc_u_in, bc_u_slip, bc_u_cyl]

# homogeneous version of the BCs, satisfied by the perturbations (u', p') of the steady state
bcs_hom = [DirichletBC(fsp.Q_v, Constant((0.0, 0.0)), rmsh.boundary_l),
           DirichletBC(fsp.Q_v.sub(1), Constant(0.0), rmsh.boundary_tb),
           DirichletBC(fsp.Q_v, Constant((0.0, 0.0)), rmsh.boundary_circle)]

F = (dot(dot(u, nabla_grad(u)), nu_test)
     + inner(grad(u), grad(nu_test)) * (1 / R)
     - div(nu_test) * p
     - q_test * div(u)
     ) * rmsh.dx

J = derivative(F, fsp.up, fsp.J_up)


def mass(trial, test):
    '''mass form of the time-dependent problem d/dt(u) = ...: only the velocity has a time derivative'''
    u_trial, _ = split(trial)
    u_test, _ = split(test)
    return inner(u_trial, u_test) * rmsh.dx


def initial_guess():
    '''free stream (1, 0) everywhere, used as starting point of Newton's method'''
    assign(up.sub(0), interpolate(U_inf, fsp.Q_v.collapse()))
    for bc in bcs:
        bc.apply(up.vector())
