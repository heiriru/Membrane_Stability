from fenics import *
import importlib
import switch_problem as swi
import function_spaces_steady as fsp
import runtime_arguments as rarg

rmsh = importlib.import_module(swi.rmsh)

# Physical parameter: Reynolds number. It is a Constant, so that it can be changed with R.assign(...) after the module
# has been imported (an importlib.import_module of this module returns the cached module and would not re-read it)
R = Constant( float(getattr(rarg.args, "Re", 1.0)) )
W   = fsp.W
up  = fsp.up
u, p = split(up)
nu_test, q_test = fsp.nu, fsp.q
J_up = fsp.J_up
# 1) Inflow profile as a UserExpression
class Inflow(UserExpression):
    def eval(self, values, x):
        H = rmsh.h
        U0 = 1.5
        values[0] = 4*U0*x[1]*(H - x[1])/(H*H)
        values[1] = 0.0
    def value_shape(self):
        return (2,)

V_full = fsp.Q_v.collapse()  # collapse to velocity subspace
v_l = interpolate(Inflow(degree=2), V_full)  # collapse only for interpolation

# 2) **Use the exact same string‐based boundaries** they did:
bc_u_in   = DirichletBC(fsp.Q_v, v_l,            rmsh.boundary_l)
bc_u_w    = DirichletBC(fsp.Q_v, Constant((0,0)), rmsh.boundary_tb)
bc_u_cyl  = DirichletBC(fsp.Q_v, Constant((0,0)), rmsh.boundary_circle)
bc_p_out  = DirichletBC(fsp.Q_p, Constant(0.0),   rmsh.boundary_r)

bcs = [bc_u_in, bc_u_w, bc_u_cyl, bc_p_out]

# homogeneous version of the BCs, satisfied by the perturbations (u', p') of the steady state
bcs_hom = [DirichletBC(fsp.Q_v, Constant((0, 0)), rmsh.boundary_l),
           DirichletBC(fsp.Q_v, Constant((0, 0)), rmsh.boundary_tb),
           DirichletBC(fsp.Q_v, Constant((0, 0)), rmsh.boundary_circle),
           DirichletBC(fsp.Q_p, Constant(0.0), rmsh.boundary_r)]

F = ( dot(dot(u, nabla_grad(u)), nu_test)
    + inner(grad(u), grad(nu_test))*(1/R)
    - div(nu_test)*p
    - q_test*div(u)
    ) * rmsh.dx

J = derivative(F, fsp.up, fsp.J_up)


def mass(trial, test):
    '''mass form of the time-dependent problem d/dt(u) = ...: only the velocity has a time derivative'''
    u_trial, _ = split(trial)
    u_test, _ = split(test)
    return inner(u_trial, u_test) * rmsh.dx


def initial_guess():
    up.vector().zero()
    for bc in bcs:
        bc.apply(up.vector())
