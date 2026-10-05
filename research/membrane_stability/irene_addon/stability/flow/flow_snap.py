'''
Where does the membrane go after the snap-through? Nonlinear time stepping of IRENE's equations from the fold of the
imperfect steady branch (contact slope t != 0, PI height clamped, rho = 0, normal friction zeta).

1. the steady branch is followed by natural continuation in v0 up to v0_start (below the fold, stable);
2. the inflow is raised to v0_run (above the fold: no nearby steady state) and m(d_t psi) + F(psi) = 0 is integrated
   with BDF2 (first step backward Euler) and an adaptive time step (the step is enlarged while the change of z per
   step stays below dz_target and Newton converges quickly, and halved when Newton fails);
3. the run ends when the state is steady (max |dz/dt| * t_scale < steady_tol), when t_max is reached, or when Newton
   fails at the smallest step (the Monge description z(x, y) breaking down: steep walls / overhangs).
Optionally (--v0_back) the inflow is then lowered step by step from the final state and each steady state is solved
by Newton (hysteresis: does the snapped state survive below the fold?).

Records: time series (t, dt, max|z|, A = projection on the critical mode, max|grad z|, min sigma), snapshots of z (P1)
at selected times, and the final state.

run with:
STABILITY_PARAMS=zeta=1,pi_height=1 python3 flow_snap.py square_b [mesh] [out] --critical_dir [dir with critical_mode.npz]
    --t -0.3 --v0_start_factor 0.90 --v0_run_factor 0.92
'''
import irene_paths  # noqa: F401

import argparse
import csv
import json
import os
import time

import numpy as np
from fenics import (Constant, Function, FunctionSpace, NonlinearVariationalProblem, NonlinearVariationalSolver,
                    assemble, assign, derivative, interpolate, project, split, sqrt, dot, grad, vertex_to_dof_map)

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp

parser = argparse.ArgumentParser()
parser.add_argument("--critical_dir", required=True)
parser.add_argument("--t", type=float, default=-0.3, help="contact slope of the PI")
parser.add_argument("--v0_start_factor", type=float, default=0.90)
parser.add_argument("--v0_run_factor", type=float, default=0.92)
parser.add_argument("--n_cont", type=int, default=24)
parser.add_argument("--dt0", type=float, default=2000.0)
parser.add_argument("--dt_max", type=float, default=2e5)
parser.add_argument("--dz_target", type=float, default=0.3, help="target max |z change| per step")
parser.add_argument("--t_max", type=float, default=5e7)
parser.add_argument("--max_steps", type=int, default=3000)
parser.add_argument("--steady_tol", type=float, default=1e-9, help="max |dz/dt| below which the state is steady")
parser.add_argument("--n_snapshots", type=int, default=60)
parser.add_argument("--v0_back", default="", help="factors of v0_c for the backward path from the final state")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
t_wall = time.time()

crit = json.load(open(os.path.join(options.critical_dir, "critical.json")))
v0_c = crit["v0_c"]
dcm = np.load(os.path.join(options.critical_dir, "critical_mode.npz"))
mode = Function(fsp.Q)
mode.vector()[:] = dcm["x_real"]
z_c = mode.sub(3, deepcopy=True)
z_c.vector()[:] /= np.abs(z_c.vector().get_local()).max()
dx = fp.rmsh.dx
zc_norm = assemble(z_c * z_c * dx)

if prm["omega_circle_const"] == 0:
    raise ValueError("run with omega_circle_const != 0 (the slope is set by scaling IRENE's omega_circle)")
omega_unit = fp.vp.omega_circle.vector().get_local() / prm["omega_circle_const"]
fp.vp.omega_circle.vector()[:] = omega_unit * options.t

P1 = FunctionSpace(fsp.Q.mesh(), "P", 1)
c1 = P1.tabulate_dof_coordinates()


def diagnostics():
    z = fsp.psi.sub(3, deepcopy=True)
    om = fsp.psi.sub(4, deepcopy=True)
    s = fsp.psi.sub(2, deepcopy=True)
    zl = z.vector().get_local()
    slope = np.sqrt(np.max(project(dot(om, om), P1).vector().get_local().clip(0)))
    return dict(A=assemble(z * z_c * dx) / zc_norm, max_abs_z=float(np.abs(zl).max()), z_min=float(zl.min()),
                z_max=float(zl.max()), max_slope=float(slope), sigma_min=float(s.vector().get_local().min()))


# 1. steady branch up to v0_start
fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
fp.set_inflow(0.0)
fp.newton()
branch = []
for v in np.linspace(0, options.v0_start_factor * v0_c, options.n_cont + 1)[1:]:
    fp.set_inflow(v)
    fp.newton()
    branch.append(dict(v0_over_v0c=v / v0_c, **diagnostics()))
print(f"steady branch up to v0 = {options.v0_start_factor} v0_c: {branch[-1]}  [{time.time() - t_wall:.0f} s]",
      flush=True)

# 2. time stepping at v0_run
v_run = options.v0_run_factor * v0_c
fp.set_inflow(v_run)
psi_n, psi_nm1 = Function(fsp.Q), Function(fsp.Q)
dt_c = Constant(options.dt0)
c0, c1_, c2 = Constant(1.0), Constant(-1.0), Constant(0.0)    # BDF coefficients times dt (set below)


def mass_rate():
    v, w, _, z, _, _ = split(fsp.psi)
    v1, w1, _, z1, _, _ = split(psi_n)
    v2, w2, _, z2, _, _ = split(psi_nm1)
    dv = [(c0 * v[k] + c1_ * v1[k] + c2 * v2[k]) / dt_c for k in range(2)]
    dw = (c0 * w + c1_ * w1 + c2 * w2) / dt_c
    dz = (c0 * z + c1_ * z1 + c2 * z2) / dt_c
    return (fp.RHO * (dv[0] * fsp.nu_v[0] + dv[1] * fsp.nu_v[1]) + fp.RHO * dw * fsp.nu_w + dz * fsp.nu_z) \
        * fp.sqrt_g * dx


F_time = mass_rate() + fp.F_used
solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F_time, fsp.psi, fp.bcs,
                                                                derivative(F_time, fsp.psi, fsp.J_psi)))
sp = solver.parameters["newton_solver"]
sp["linear_solver"] = "mumps"
sp["absolute_tolerance"] = 1e-10
sp["relative_tolerance"] = 1e-9
sp["maximum_iterations"] = 12
sp["report"] = False
sp["error_on_nonconvergence"] = True

psi_n.vector()[:] = fsp.psi.vector()
psi_nm1.vector()[:] = fsp.psi.vector()
rows = [dict(t=0.0, dt=0.0, rate=0.0, **diagnostics())]
snaps_t, snaps_z = [0.0], [project(fsp.psi.sub(3), P1).vector().get_local()]
t, dt, dt_prev, n, end_reason = 0.0, options.dt0, None, 0, "t_max"
first = True
snap_every = None
while t < options.t_max and n < options.max_steps:
    # BDF2 with variable step (omega = dt / dt_prev); first step backward Euler
    if first:
        a0, a1, a2 = 1.0, -1.0, 0.0
    else:
        om = dt / dt_prev
        a0, a1, a2 = (1 + 2 * om) / (1 + om), -(1 + om), om ** 2 / (1 + om)
    c0.assign(a0)
    c1_.assign(a1)
    c2.assign(a2)
    dt_c.assign(dt)
    z_old = psi_n.sub(3, deepcopy=True).vector().get_local()
    try:
        its, _ = solver.solve()
    except RuntimeError:
        fsp.psi.vector()[:] = psi_n.vector()
        dt *= 0.25
        if dt < 1e-3 * options.dt0:
            end_reason = "newton_failed"
            print(f"Newton failed at t = {t:.4e} with dt = {dt:.2e}: stopping", flush=True)
            break
        continue
    z_new = fsp.psi.sub(3, deepcopy=True).vector().get_local()
    dz = float(np.abs(z_new - z_old).max())
    if dz > 2 * options.dz_target and dt > 1e-3 * options.dt0:
        fsp.psi.vector()[:] = psi_n.vector()          # reject: too large a change
        dt *= 0.5
        continue
    t += dt
    n += 1
    first = False
    psi_nm1.vector()[:] = psi_n.vector()
    psi_n.vector()[:] = fsp.psi.vector()
    rate = dz / dt
    rows.append(dict(t=t, dt=dt, rate=rate, **diagnostics()))
    if n % 10 == 0:
        r = rows[-1]
        print(f"step {n}: t = {t:.4e}, dt = {dt:.2e}, max|dz/dt| = {rate:.2e}, A = {r['A']:+.3f}, "
              f"max|z| = {r['max_abs_z']:.3f}, max slope = {r['max_slope']:.3f}, min sigma = {r['sigma_min']:+.4f}  "
              f"[{time.time() - t_wall:.0f} s]", flush=True)
    if n % 5 == 0:
        snaps_t.append(t)
        snaps_z.append(project(fsp.psi.sub(3), P1).vector().get_local())
    if rate < options.steady_tol and n > 20:
        end_reason = "steady"
        break
    dt_prev = dt
    # step control: aim at dz_target per step
    dt = float(np.clip(dt * min(2.0, 0.8 * options.dz_target / max(dz, 1e-14)), 0.2 * dt, options.dt_max))
    dt = min(dt, 2.0 * dt_prev)

print(f"time stepping ended ({end_reason}) at t = {t:.4e} after {n} steps: {rows[-1]}  [{time.time() - t_wall:.0f} s]",
      flush=True)
snaps_t.append(t)
snaps_z.append(project(fsp.psi.sub(3), P1).vector().get_local())
psi_final = fsp.psi.vector().get_local().copy()

# 3. backward path (hysteresis)
back = []
if options.v0_back and end_reason == "steady":
    for f in [float(x) for x in options.v0_back.split(",")]:
        fp.set_inflow(f * v0_c)
        try:
            fp.newton()
        except RuntimeError:
            print(f"backward path: no convergence at v0 = {f} v0_c", flush=True)
            break
        back.append(dict(v0_over_v0c=f, **diagnostics()))
        print(f"backward: v0 = {f} v0_c: {back[-1]}", flush=True)

with open(os.path.join(out_dir, "time_series.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
np.savez(os.path.join(out_dir, "snapshots.npz"), x=c1[:, 0], y=c1[:, 1],
         triangles=vertex_to_dof_map(P1)[fsp.Q.mesh().cells()], t=np.array(snaps_t), z=np.array(snaps_z),
         psi_final=psi_final)
json.dump(dict(v0_c=v0_c, t_slope=options.t, v0_start_factor=options.v0_start_factor,
               v0_run_factor=options.v0_run_factor, end_reason=end_reason, t_end=t, steps=n, final=rows[-1],
               steady_branch=branch, backward=back, mesh=rarg.args.input_directory,
               wall_time=time.time() - t_wall), open(os.path.join(out_dir, "snap.json"), "w"), indent=1)
print("... done.", flush=True)
