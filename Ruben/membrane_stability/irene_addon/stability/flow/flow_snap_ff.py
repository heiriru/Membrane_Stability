'''
Post-snap dynamics with a force-free PI: nonlinear time stepping of IRENE's equations in which the PI height h(t) is
not imposed but follows from the vertical force balance on the (massless, rigid) PI at every time step.

The PI is clamped at height h on its rim (pi_height = 1: z = h, w = 0, contact slope t); h is solved for at each time
step such that the vertical force of the membrane on the PI vanishes. The force is the conserved vertical stress flux
(modules/stability/vertical_force.py; checked against -dE/dh to 5e-5) plus, out of equilibrium, the body force of the
normal friction in the cutoff region: d_i J^i - zeta w = 0, so  F = - int J . grad(phi) dx - int phi zeta w dx.
(At steady states w = 0 and F reduces to the static force used in flow_forcefree.py.)

1. force-free steady branch in v0 (secant on h at each v0) up to v0_start (below the fold);
2. v0 is raised to v0_run (above the fold) and m(d_t psi) + F(psi) = 0 is integrated with variable-step BDF2; at each
   step h is found by a secant iteration on F(h) = 0 (each evaluation is one implicit step);
3. stops when steady, at t_max, or when Newton fails (Monge breakdown).

run with:
STABILITY_PARAMS=zeta=1,pi_height=1 python3 flow_snap_ff.py square_b [mesh] [out] --critical_dir [force-free crit]
    --t -0.3 --v0_start_factor 0.915 --v0_run_factor 0.93
'''
import irene_paths  # noqa: F401

import argparse
import csv
import json
import os
import time

import numpy as np
from fenics import (Constant, Function, FunctionSpace, NonlinearVariationalProblem, NonlinearVariationalSolver,
                    assemble, assign, derivative, dot, interpolate, project, split, vertex_to_dof_map)
import ufl

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
from stability import vertical_force as vf

parser = argparse.ArgumentParser()
parser.add_argument("--critical_dir", required=True)
parser.add_argument("--t", type=float, default=-0.3)
parser.add_argument("--v0_start_factor", type=float, default=0.915)
parser.add_argument("--v0_run_factor", type=float, default=0.93)
parser.add_argument("--v_steps", default="0,0.2,0.4,0.6,0.7,0.8,0.85,0.88,0.9,0.91")
parser.add_argument("--dt0", type=float, default=5000.0)
parser.add_argument("--dt_max", type=float, default=1e5)
parser.add_argument("--dz_target", type=float, default=0.8)
parser.add_argument("--t_max", type=float, default=5e7)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--steady_tol", type=float, default=1e-9)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
t_wall = time.time()
v0_c = json.load(open(os.path.join(options.critical_dir, "critical.json")))["v0_c"]
dx = fp.rmsh.dx
c_r = fp.rmsh.parameters["c_r"][:2]
omega_unit = fp.vp.omega_circle.vector().get_local() / prm["omega_circle_const"]
fp.vp.omega_circle.vector()[:] = omega_unit * options.t
mesh = fsp.Q.mesh()
radii = ((1.5, 4.0), (2.0, 6.0), (3.0, 8.0))
cutoffs = [vf.cutoff(c_r, a, b, mesh) for a, b in radii]


def force(dynamic=True):
    v, w, sigma, z, omega, mu = fsp.psi.split(deepcopy=True)
    J = vf.flux(omega, mu, sigma, prm["kappa"], geo.d_c(v, w, omega), prm["eta"])
    out = []
    for phi in cutoffs:
        f = -assemble(ufl.dot(J, ufl.grad(phi)) * dx)
        if dynamic:
            f -= assemble(phi * prm["zeta"] * w * dx)
        out.append(f)
    return out


def secant(fun, x0, x1, tol=1e-9, max_it=15):
    f0, f1 = fun(x0), fun(x1)
    for _ in range(max_it):
        if abs(f1) < tol:
            return x1
        if f1 == f0:
            break
        x0, f0, x1 = x1, f1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f1 = fun(x1)
    if abs(f1) < 1e-6:
        return x1
    raise RuntimeError("secant did not converge")


# 1. force-free steady branch
fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
state = None


def steady_F(v0):
    def fun(h):
        fsp.psi.vector()[:] = state
        fp.z_pi_const.assign(h)
        fp.set_inflow(v0)
        fp.newton()
        return force(dynamic=False)[1]
    return fun


fp.z_pi_const.assign(0.0)
fp.set_inflow(0.0)
fp.newton()
state = fsp.psi.vector().get_local().copy()
h = 0.0
hist = []
targets = [float(x) * v0_c for x in options.v_steps.split(",")] + [options.v0_start_factor * v0_c]
for v0 in targets:
    hp = hist[-1][1] + (hist[-1][1] - hist[-2][1]) * (v0 - hist[-1][0]) / (hist[-1][0] - hist[-2][0]) \
        if len(hist) >= 2 else h
    h = secant(steady_F(v0), hp, hp + 1e-3)
    state = fsp.psi.vector().get_local().copy()
    hist.append((v0, h))
    print(f"steady force-free: v0/v0_c = {v0 / v0_c:.4f}, h = {h:+.5f}  [{time.time() - t_wall:.0f} s]", flush=True)

# 2. time stepping at v0_run with the force-free PI
P1 = FunctionSpace(mesh, "P", 1)
fp.set_inflow(options.v0_run_factor * v0_c)
psi_n, psi_nm1 = Function(fsp.Q), Function(fsp.Q)
dt_c, c0, c1_, c2 = Constant(options.dt0), Constant(1.0), Constant(-1.0), Constant(0.0)


def mass_rate():
    v, w, _, z, _, _ = split(fsp.psi)
    v1, w1, _, z1, _, _ = split(psi_n)
    v2, w2, _, z2, _, _ = split(psi_nm1)
    dz = (c0 * z + c1_ * z1 + c2 * z2) / dt_c
    return dz * fsp.nu_z * fp.sqrt_g * dx       # rho = 0: only the kinematic equation has a time derivative


assert fp.RHO == 0, "run with rho = 0"
F_time = mass_rate() + fp.F_used
solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F_time, fsp.psi, fp.bcs,
                                                                derivative(F_time, fsp.psi, fsp.J_psi)))
sp = solver.parameters["newton_solver"]
sp.update({"linear_solver": "mumps", "absolute_tolerance": 1e-10, "relative_tolerance": 1e-9,
           "maximum_iterations": 12, "report": False})
psi_n.vector()[:] = state
psi_nm1.vector()[:] = state
fsp.psi.vector()[:] = state


def step_with_h(hh):
    fsp.psi.vector()[:] = psi_n.vector()
    fp.z_pi_const.assign(hh)
    solver.solve()
    return force(dynamic=True)[1]


def diag():
    z = fsp.psi.sub(3, deepcopy=True).vector().get_local()
    om = fsp.psi.sub(4, deepcopy=True)
    slope = np.sqrt(np.max(project(dot(om, om), P1).vector().get_local().clip(0)))
    return dict(max_abs_z=float(np.abs(z).max()), z_min=float(z.min()), z_max=float(z.max()), max_slope=float(slope))


rows = [dict(t=0.0, dt=0.0, h=h, rate=0.0, F_spread=0.0, **diag())]
snaps_t, snaps_z = [0.0], [project(fsp.psi.sub(3), P1).vector().get_local()]
t, dt, dt_prev, n, end, first = 0.0, options.dt0, None, 0, "t_max", True
h_prev = h
while t < options.t_max and n < options.max_steps:
    if first:
        a = (1.0, -1.0, 0.0)
    else:
        om_ = dt / dt_prev
        a = ((1 + 2 * om_) / (1 + om_), -(1 + om_), om_ ** 2 / (1 + om_))
    c0.assign(a[0])
    c1_.assign(a[1])
    c2.assign(a[2])
    dt_c.assign(dt)
    z_old = psi_n.sub(3, deepcopy=True).vector().get_local()
    try:
        h_new = secant(step_with_h, h, h + (h - h_prev if n > 0 else 1e-3) + 1e-4, tol=1e-9)
    except RuntimeError:
        dt *= 0.3
        if dt < 1e-3:
            end = "newton_failed"
            break
        continue
    z_new = fsp.psi.sub(3, deepcopy=True).vector().get_local()
    dzmax = float(np.abs(z_new - z_old).max())
    if dzmax > 2 * options.dz_target and dt > 1e-2:
        dt *= 0.5
        continue
    Fs = force(dynamic=True)
    t += dt
    n += 1
    first = False
    psi_nm1.vector()[:] = psi_n.vector()
    psi_n.vector()[:] = fsp.psi.vector()
    h_prev, h = h, h_new
    rows.append(dict(t=t, dt=dt, h=h, rate=dzmax / dt, F_spread=float(max(Fs) - min(Fs)), **diag()))
    r = rows[-1]
    print(f"step {n}: t = {t:.4e}, dt = {dt:.2e}, h = {h:+.4f}, max|z| = {r['max_abs_z']:.3f} (z in [{r['z_min']:.2f}, "
          f"{r['z_max']:.2f}]), max slope = {r['max_slope']:.3f}, max|dz/dt| = {r['rate']:.2e}  "
          f"[{time.time() - t_wall:.0f} s]", flush=True)
    if n % 3 == 0:
        snaps_t.append(t)
        snaps_z.append(project(fsp.psi.sub(3), P1).vector().get_local())
    with open(os.path.join(out_dir, "time_series.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    if r["rate"] < options.steady_tol and n > 10:
        end = "steady"
        break
    dt_prev = dt
    dt = float(np.clip(dt * min(2.0, 0.8 * options.dz_target / max(dzmax, 1e-14)), 0.2 * dt, options.dt_max))
    np.savez(os.path.join(out_dir, "snapshots.npz"), x=P1.tabulate_dof_coordinates()[:, 0],
             y=P1.tabulate_dof_coordinates()[:, 1], triangles=vertex_to_dof_map(P1)[mesh.cells()],
             t=np.array(snaps_t), z=np.array(snaps_z))
snaps_t.append(t)
snaps_z.append(project(fsp.psi.sub(3), P1).vector().get_local())
np.savez(os.path.join(out_dir, "snapshots.npz"), x=P1.tabulate_dof_coordinates()[:, 0],
         y=P1.tabulate_dof_coordinates()[:, 1], triangles=vertex_to_dof_map(P1)[mesh.cells()],
         t=np.array(snaps_t), z=np.array(snaps_z), psi_final=fsp.psi.vector().get_local())
json.dump(dict(t_slope=options.t, v0_c=v0_c, v0_run_factor=options.v0_run_factor, end=end, t_end=t, steps=n,
               final=rows[-1], steady_branch=hist, wall=time.time() - t_wall),
          open(os.path.join(out_dir, "snap_ff.json"), "w"), indent=1)
print(f"ended ({end}) at t = {t:.4e}: {rows[-1]}", flush=True)
