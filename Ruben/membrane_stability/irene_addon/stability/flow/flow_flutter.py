'''
Nonlinear flutter: does a force-free PI in a flowing, friction-screened membrane emit a saturated, travelling wave?

With in-plane friction (b_friction, screening length ell_b = sqrt(eta / b)) the first instability of the flat state is a
Hopf bifurcation (flutter, INTERPRETATION section 11). This script integrates IRENE's full nonlinear equations from the
flat state just above (or below) the flutter threshold v0_c, kicked by a small (or large) upstream bump, and records
whether the oscillation saturates (supercritical Hopf: a limit cycle, i.e. a wave source) or grows without bound
(subcritical: snap). Below threshold, a kick that does not decay signals a subcritical Hopf bifurcation.

The PI is rigid, horizontal and force-free (as in flow_snap_ff.py): it is clamped at height h (pi_height = 1) and h is
found at every implicit BDF2 step by a secant iteration on the vertical force (conserved stress flux plus the normal
friction body force). The contact slope is zero (flat steady state).

Records: time series of h, max|z|, ||z||_2, the height at probes on the centreline, max slope; snapshots of z (P1).
Amplitude analysis (envelope, growth rate vs amplitude = cubic Landau coefficient) in plot_round5.py.

run with:
STABILITY_PARAMS=zeta=1,pi_height=1,omega_circle_const=1,b_friction=0.0204,sigma_r_const=0.0025 \
    python3 flow_flutter.py square_b [mesh] [out] --critical_dir [flutter critical dir] --v0_factor 1.05 --kick 0.05
'''
import irene_paths  # noqa: F401

import argparse
import csv
import json
import os
import time

import numpy as np
from fenics import (Constant, Expression, Function, FunctionSpace, NonlinearVariationalProblem,
                    NonlinearVariationalSolver, assemble, assign, derivative, dot, interpolate, project, split,
                    vertex_to_dof_map)
import ufl

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
from stability import vertical_force as vf

parser = argparse.ArgumentParser()
parser.add_argument("--critical_dir", required=True)
parser.add_argument("--v0_factor", type=float, default=1.05)
parser.add_argument("--kick", type=float, default=0.05, help="height of the initial upstream bump (r0)")
parser.add_argument("--kick_x", type=float, default=-15.0, help="bump centre relative to the PI (flow along +x)")
parser.add_argument("--steps_per_period", type=int, default=28)
parser.add_argument("--periods", type=float, default=7.0)
parser.add_argument("--z_stop", type=float, default=15.0, help="stop when max|z| exceeds this (snap)")
parser.add_argument("--snap_every", type=int, default=2)
parser.add_argument("--init_from", default="", help="start from the last state (checkpoint.npz) of another run (e.g. a "
                    "saturated state, to test for hysteresis below threshold)")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
t_wall = time.time()
crit = json.load(open(os.path.join(options.critical_dir, "critical.json")))
v0_c = crit["v0_c"]
lam = np.load(os.path.join(options.critical_dir, "critical_mode.npz"))["eigenvalue"]
omega_H = abs(lam.imag)
period = 2 * np.pi / omega_H
dx = fp.rmsh.dx
c_r = fp.rmsh.parameters["c_r"][:2]
assert fp.RHO == 0, "run with rho = 0"
assert fp.PI_HEIGHT == 1, "run with pi_height = 1 (the height h is then made force-free here)"
fp.vp.omega_circle.vector()[:] = 0.0          # zero contact slope: the flat state is an exact steady state
mesh = fsp.Q.mesh()
radii = ((1.5, 4.0), (2.0, 6.0), (3.0, 8.0))
cutoffs = [vf.cutoff(c_r, a, b, mesh) for a, b in radii]
P1 = FunctionSpace(mesh, "P", 1)
xy1 = P1.tabulate_dof_coordinates()
probes_dx = [-30.0, -15.0, -8.0, -4.0, 4.0, 8.0, 15.0]


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


# flat steady state at v0 (exact: z = 0, h = 0)
v_run = options.v0_factor * v0_c
fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
fp.z_pi_const.assign(0.0)
for v in np.linspace(0, v_run, 5)[1:]:
    fp.set_inflow(v)
    fp.newton()
print(f"flat steady state at v0 = {options.v0_factor} v0_c = {v_run:.5f}; flutter period 2 pi / Im lambda = {period:.4e}"
      f"  [{time.time() - t_wall:.0f} s]", flush=True)
# kick: smooth bump upstream of the PI (the implicit steps restore omega = grad z and mu = H at once)
bump = interpolate(Expression("a * exp(-(pow(x[0] - cx, 2) + pow(x[1] - cy, 2)) / (w * w))", a=options.kick,
                              cx=c_r[0] + options.kick_x, cy=c_r[1], w=6.0, degree=4), fsp.Q_z)
assign(fsp.psi.sub(3), bump)
state = fsp.psi.vector().get_local().copy()

psi_n, psi_nm1 = Function(fsp.Q), Function(fsp.Q)
dt = period / options.steps_per_period
dt_c, c0, c1_, c2 = Constant(dt), Constant(1.0), Constant(-1.0), Constant(0.0)


def mass_rate():
    _, _, _, z, _, _ = split(fsp.psi)
    _, _, _, z1, _, _ = split(psi_n)
    _, _, _, z2, _, _ = split(psi_nm1)
    return (c0 * z + c1_ * z1 + c2 * z2) / dt_c * fsp.nu_z * fp.sqrt_g * dx


F_time = mass_rate() + fp.F_used
solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F_time, fsp.psi, fp.bcs,
                                                                derivative(F_time, fsp.psi, fsp.J_psi)))
sp = solver.parameters["newton_solver"]
sp.update({"linear_solver": "mumps", "absolute_tolerance": 1e-10, "relative_tolerance": 1e-9,
           "maximum_iterations": 15, "report": False})
psi_n.vector()[:] = state
psi_nm1.vector()[:] = state
ckpt = os.path.join(out_dir, "checkpoint.npz")
resume = None
if not os.path.exists(ckpt) and options.init_from:
    src = np.load(os.path.join(options.init_from, "checkpoint.npz"), allow_pickle=True)
    psi_n.vector()[:] = src["psi_n"]
    psi_nm1.vector()[:] = src["psi_n"]
    fp.z_pi_const.assign(float(src["h"]))
    print(f"initial state: last state of {options.init_from} (t/T = {float(src['t']) / period:.3f})", flush=True)
if os.path.exists(ckpt):
    resume = np.load(ckpt, allow_pickle=True)
    psi_n.vector()[:] = resume["psi_n"]
    psi_nm1.vector()[:] = resume["psi_nm1"]
    print(f"resuming from step {int(resume['n'])}, t/T = {float(resume['t']) / period:.3f}", flush=True)


def step_with_h(hh):
    fsp.psi.vector()[:] = psi_n.vector()
    fp.z_pi_const.assign(hh)
    solver.solve()
    return force(dynamic=True)[1]


def probe_values(zf):
    out = []
    for d in probes_dx:
        try:
            out.append(float(zf(c_r[0] + d, c_r[1])))
        except RuntimeError:
            out.append(float("nan"))
    return out


def diag():
    zf = fsp.psi.sub(3, deepcopy=True)
    z = zf.vector().get_local()
    om = fsp.psi.sub(4, deepcopy=True)
    slope = np.sqrt(np.max(project(dot(om, om), P1).vector().get_local().clip(0)))
    l2 = np.sqrt(assemble(zf * zf * dx))
    d = dict(max_abs_z=float(np.abs(z).max()), z_min=float(z.min()), z_max=float(z.max()), z_l2=float(l2),
             max_slope=float(slope))
    for k, val in zip(probes_dx, probe_values(zf)):
        d[f"z_at_{k:+g}"] = val
    return d


h, h_prev = 0.0, 0.0
fsp.psi.vector()[:] = state
n_steps = int(options.periods * options.steps_per_period)
end = "t_max"
t = 0.0
n0 = 1
if resume is not None:
    h, h_prev, t, n0 = float(resume["h"]), float(resume["h_prev"]), float(resume["t"]), int(resume["n"]) + 1
    rows = list(resume["rows"])
    snaps_t, snaps_z = list(resume["snaps_t"]), list(resume["snaps_z"])
    fsp.psi.vector()[:] = psi_n.vector()
else:
    if options.init_from:
        h = h_prev = float(src["h"])
        fsp.psi.vector()[:] = psi_n.vector()
    rows = [dict(t=0.0, h=h, F_spread=0.0, **diag())]
    snaps_t, snaps_z = [0.0], [project(fsp.psi.sub(3), P1).vector().get_local()]
for n in range(n0, n_steps + 1):
    a = (1.0, -1.0, 0.0) if n == 1 else (1.5, -2.0, 0.5)     # BDF2 (constant step), first step backward Euler
    c0.assign(a[0])
    c1_.assign(a[1])
    c2.assign(a[2])
    guess = h + (h - h_prev) if n > 1 else h
    try:
        h_new = secant(step_with_h, guess, guess + 1e-4 + 0.05 * abs(h - h_prev), tol=1e-9)
    except RuntimeError:
        end = "newton_failed"
        print(f"step {n}: no convergence (snap / Monge breakdown); stopping", flush=True)
        break
    Fs = force(dynamic=True)
    t += dt
    psi_nm1.vector()[:] = psi_n.vector()
    psi_n.vector()[:] = fsp.psi.vector()
    h_prev, h = h, h_new
    rows.append(dict(t=t, h=h, F_spread=float(max(Fs) - min(Fs)), **diag()))
    r = rows[-1]
    print(f"step {n}: t/T = {t / period:.3f}, h = {h:+.4f}, max|z| = {r['max_abs_z']:.4f}, ||z|| = {r['z_l2']:.4f}, "
          f"z(-15) = {r['z_at_-15']:+.4f}, slope = {r['max_slope']:.3f}  [{time.time() - t_wall:.0f} s]", flush=True)
    if n % options.snap_every == 0:
        snaps_t.append(t)
        snaps_z.append(project(fsp.psi.sub(3), P1).vector().get_local())
    np.savez(ckpt + ".tmp.npz", psi_n=psi_n.vector().get_local(), psi_nm1=psi_nm1.vector().get_local(), h=h,
             h_prev=h_prev, t=t, n=n, rows=np.array(rows, dtype=object), snaps_t=np.array(snaps_t),
             snaps_z=np.array(snaps_z))
    os.replace(ckpt + ".tmp.npz", ckpt)
    if n % 10 == 0 or r["max_abs_z"] > options.z_stop:
        with open(os.path.join(out_dir, "time_series.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)
        np.savez(os.path.join(out_dir, "snapshots.npz"), x=xy1[:, 0], y=xy1[:, 1],
                 triangles=vertex_to_dof_map(P1)[mesh.cells()], t=np.array(snaps_t), z=np.array(snaps_z))
    if r["max_abs_z"] > options.z_stop:
        end = "snap"
        break

with open(os.path.join(out_dir, "time_series.csv"), "w", newline="") as fh:
    wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
np.savez(os.path.join(out_dir, "snapshots.npz"), x=xy1[:, 0], y=xy1[:, 1],
         triangles=vertex_to_dof_map(P1)[mesh.cells()], t=np.array(snaps_t), z=np.array(snaps_z))
json.dump(dict(v0_c=v0_c, v0_factor=options.v0_factor, kick=options.kick, period=period, omega_H=omega_H,
               lam_c=[float(lam.real), float(lam.imag)], dt=dt, end=end, t_end=t, steps=len(rows) - 1,
               final=rows[-1], critical_dir=options.critical_dir, wall=time.time() - t_wall),
          open(os.path.join(out_dir, "flutter.json"), "w"), indent=1)
print(f"ended ({end}) at t/T = {t / period:.3f}", flush=True)
