'''
Time-dependent Navier-Stokes simulation (DNS) used to cross-check the linear stability analysis.

The flow starts from the steady state w* at the given Re, receives a weak, short vertical body-force kick in the
wake (t < 1), and is integrated in time. If the steady state is linearly unstable, the perturbation grows like
exp(sigma t) cos(omega t) with the (sigma, omega) of the leading eigenvalue until it saturates (vortex shedding);
if it is stable, the perturbation decays.

Discretization: same mesh, Taylor-Hood P2/P1 spaces, BCs and forms as the steady problem; BDF2 in time with
semi-implicit convection (u^* . grad) u^{n+1}, u^* = 2 u^n - u^{n-1}; one MUMPS solve per time step.

run with:
python3 solve_dynamics.py [problem] [mesh directory] [output directory] --Re 60 --T 300 --dt 0.1
'''
import argparse
import importlib
import os
import time

import numpy as np
from fenics import (Function, TrialFunctions, TestFunctions, Constant, Expression, FacetNormal, Identity, dot, inner,
                    grad, nabla_grad, div, sym, assemble, assemble_system, LUSolver, split, sqrt, parameters)

import runtime_arguments as rarg
import switch_problem as swi
import steady_state as ss
import plot_utils as pu

parser = argparse.ArgumentParser()
parser.add_argument("--Re", type=float, required=True)
parser.add_argument("--T", type=float, default=300.0)
parser.add_argument("--dt", type=float, default=0.1)
parser.add_argument("--kick", type=float, default=1e-3, help="amplitude of the body-force kick")
parser.add_argument("--snapshot_every", type=float, default=2.0, help="time between vorticity snapshots")
parser.add_argument("--probes", default="3,0;5,0", help="points where the velocity is recorded")
options = parser.parse_args(rarg.unknown_args)

parameters["std_out_all_processes"] = False
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)

vp = importlib.import_module(swi.vp_steady)
rmsh = vp.rmsh
mesh = rmsh.lmsh.mesh
W = vp.W
Re, dt = options.Re, options.dt
print(f"DNS at Re = {Re}, dt = {dt}, T = {options.T}, dofs = {W.dim()}", flush=True)

# 1) steady state
ss.solve_steady(vp, Re)
w_base = Function(W)
w_base.assign(vp.up)
u_base = split(w_base)[0]

# 2) time stepping
w = Function(W)
w_n = Function(W)
w_nm1 = Function(W)
w_n.assign(w_base)
w_nm1.assign(w_base)
u_n, _ = split(w_n)
u_nm1, _ = split(w_nm1)
u, p = TrialFunctions(W)
v, q = TestFunctions(W)

# body force: localized vertical kick behind the cylinder, switched on smoothly for 0 < t < 1
kick = Expression(("0.0", "t < 1.0 ? A * pow(sin(pi * t), 2) * exp(-(pow(x[0] - xc, 2) + pow(x[1] - yc, 2)) / 0.5) : 0.0"),
                  degree=4, A=options.kick, t=0.0, xc=rmsh.c_r[0] + 1.5, yc=rmsh.c_r[1])

alpha, beta, gamma = Constant(1.0), Constant(1.0), Constant(0.0)  # BDF1 for the first step, then BDF2
extrapolation = Constant(1.0)  # u^* = u^n for the first step, then 2 u^n - u^{n-1}
u_star = (1.0 + extrapolation) * u_n - extrapolation * u_nm1
a = (alpha / dt * dot(u, v) + dot(dot(u_star, nabla_grad(u)), v) + inner(grad(u), grad(v)) / vp.R
     - div(v) * p - q * div(u)) * rmsh.dx
L = ((beta * dot(u_n, v) + gamma * dot(u_nm1, v)) / dt + dot(kick, v)) * rmsh.dx

solver = LUSolver("mumps")

# diagnostics
n_facet = FacetNormal(mesh)
u_w, p_w = split(w)
stress = -p_w * Identity(2) + 2.0 / vp.R * sym(grad(u_w))
force_x = -dot(dot(stress, n_facet), Constant((1.0, 0.0))) * rmsh.ds_circle
force_y = -dot(dot(stress, n_facet), Constant((0.0, 1.0))) * rmsh.ds_circle
energy = 0.5 * dot(u_w - u_base, u_w - u_base) * rmsh.dx
probes = [tuple(float(c) for c in s.split(",")) for s in options.probes.split(";")]

history = {k: [] for k in ["t", "C_D", "C_L", "E"] + [f"v_probe{i}" for i in range(len(probes))]}
snapshots_t, snapshots_vorticity = [], []
w.assign(w_base)
next_snapshot = 0.0

n_steps = int(round(options.T / dt))
t = 0.0
t_start = time.time()
for step in range(1, n_steps + 1):
    t = step * dt
    kick.t = t
    if step == 2:
        alpha.assign(1.5), beta.assign(2.0), gamma.assign(-0.5), extrapolation.assign(1.0)
    if step == 1:
        extrapolation.assign(0.0)
    A_mat, b = assemble_system(a, L, vp.bcs)
    solver.solve(A_mat, w.vector(), b)
    w_nm1.assign(w_n)
    w_n.assign(w)

    u_now = w.split(deepcopy=True)[0]
    history["t"].append(t)
    history["C_D"].append(2 * assemble(force_x))
    history["C_L"].append(2 * assemble(force_y))
    history["E"].append(assemble(energy))
    for i, point in enumerate(probes):
        history[f"v_probe{i}"].append(u_now(*point)[1])
    if t >= next_snapshot - 1e-9:
        snapshots_t.append(t)
        snapshots_vorticity.append(pu.vertex_values(pu.vorticity(u_now, mesh), mesh))
        next_snapshot += options.snapshot_every
    if step % 100 == 0 or step == n_steps:
        elapsed = time.time() - t_start
        print(f"t = {t:7.2f}  C_D = {history['C_D'][-1]:.5f}  C_L = {history['C_L'][-1]:+.3e}  "
              f"E = {history['E'][-1]:.3e}  [{elapsed:.0f} s, {elapsed / step:.2f} s/step]", flush=True)
        np.savez(os.path.join(out_dir, "history.npz"), Re=Re, dt=dt, probes=np.array(probes),
                 **{k: np.array(val) for k, val in history.items()})
        np.savez_compressed(os.path.join(out_dir, "snapshots.npz"), t=np.array(snapshots_t),
                            vorticity=np.array(snapshots_vorticity), **pu.mesh_arrays(mesh))
print("... done.", flush=True)
