'''
Nonlinear time stepping of IRENE's flowing membrane with mobile, curvature-coupled proteins (flow_problem.py of this
folder, consistent nonlinear forms: STABILITY_PARAMS phi_form=1, with a quartic term beta_phi > 0 so that demixing
saturates).

The membrane flows past an anchored PI (clamped, pi_height = 1, zero contact slope) at inflow velocity v0; the protein
density phi (deviation from the mean) starts from small random noise (and/or is injected at the PI rim at a total rate
Q_source: proteins recruited by the anchored PI). The equations
    m(d_t psi) + F(psi) = 0,   m = <z, nu_z> + <phi, nu_phi>   (rho = 0: z and phi are the dynamical fields)
are integrated with variable-step BDF2 (first step backward Euler); the step is adapted to the change of phi and z per
step.

Checks recorded at every step: protein number N = int phi dA (conserved without source and with no-flux edges), and
the free energy G = int [2 kappa (H - C phi)^2 + chi/2 phi^2 + beta/4 phi^4 + eps/2 |grad phi|^2 + sigma_r] dA, which
must decrease at rest (v0 = 0) for a thermodynamically consistent model.

run with (no flow, energy check):
STABILITY_PARAMS=zeta=1,omega_circle_const=0,pi_height=1,C_phi=0.25,eps_phi=1,M_phi=1,phi_form=1,beta_phi=1,chi_phi=-0.08,phi_inflow_dirichlet=0 \
    python3 phi_dynamics.py square_b [mesh] [out] --v0 0 --t_end 2e4
'''
import irene_paths  # noqa: F401

import argparse
import csv
import json
import os
import time

import numpy as np
from fenics import (Constant, Function, FunctionSpace, NonlinearVariationalProblem, NonlinearVariationalSolver,
                    assemble, assign, derivative, interpolate, project, split, vertex_to_dof_map)
import ufl

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo

parser = argparse.ArgumentParser()
parser.add_argument("--v0", type=float, default=0.0)
parser.add_argument("--t_end", type=float, default=2e4)
parser.add_argument("--dt0", type=float, default=20.0)
parser.add_argument("--dt_max", type=float, default=2000.0)
parser.add_argument("--dphi_target", type=float, default=0.02)
parser.add_argument("--dz_target", type=float, default=0.3)
parser.add_argument("--noise", type=float, default=0.01)
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--Q_source", type=float, default=0.0, help="total protein injection rate at the PI rim")
parser.add_argument("--n_snap", type=int, default=150)
parser.add_argument("--max_steps", type=int, default=3000)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
t_wall = time.time()
assert fp.RHO == 0 and fp.PHI_FORM == "consistent"
mesh = fsp.Q.mesh()
dx = fp.rmsh.dx
L = fp.rmsh.parameters["L"]

# initial state: flat steady flow at v0 (phi = 0 is a steady state)
fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
for v in (np.linspace(0, options.v0, 5)[1:] if options.v0 > 0 else [0.0]):
    fp.set_inflow(v)
    fp.newton()
# random protein noise: superposition of plane waves with wavelengths 15-90 r0
rng = np.random.default_rng(options.seed)
xy = fsp.Q_phi.tabulate_dof_coordinates()
phi0 = np.zeros(len(xy))
for _ in range(60):
    k = 2 * np.pi / rng.uniform(15, 90)
    th = rng.uniform(0, 2 * np.pi)
    phi0 += rng.standard_normal() * np.cos(k * (np.cos(th) * xy[:, 0] + np.sin(th) * xy[:, 1]) + rng.uniform(0, 2 * np.pi))
phi0 *= options.noise / np.abs(phi0).max()
if fp.PHI_INFLOW:
    phi0 *= np.clip(xy[:, 0] / 10.0, 0, 1)            # compatible with phi = 0 at the inflow
f0 = Function(fsp.Q_phi)
f0.vector()[:] = phi0
assign(fsp.psi.sub(6), f0)

psi_n, psi_nm1 = Function(fsp.Q), Function(fsp.Q)
dt_c, c0, c1_, c2 = Constant(options.dt0), Constant(1.0), Constant(-1.0), Constant(0.0)
_, _, _, z, om, mu, phi, m = split(fsp.psi)
_, _, _, z1, _, _, phi1, _ = split(psi_n)
_, _, _, z2, _, _, phi2, _ = split(psi_nm1)
sg = geo.sqrt_detg(om)
F_time = ((c0 * z + c1_ * z1 + c2 * z2) / dt_c * fsp.nu_z + (c0 * phi + c1_ * phi1 + c2 * phi2) / dt_c * fsp.nu_phi) \
    * sg * dx + fp.F_used
if options.Q_source:
    circ = assemble(Constant(1.0) * fp.rmsh.ds_circle)
    F_time = F_time - Constant(options.Q_source / circ) * fsp.nu_phi * fp.rmsh.ds_circle
solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F_time, fsp.psi, fp.bcs,
                                                                derivative(F_time, fsp.psi, fsp.J_psi)))
sp = solver.parameters["newton_solver"]
sp.update({"linear_solver": "mumps", "absolute_tolerance": 1e-10, "relative_tolerance": 1e-9,
           "maximum_iterations": 12, "report": False})
psi_n.vector()[:] = fsp.psi.vector()
psi_nm1.vector()[:] = fsp.psi.vector()

C_eff, chi_eff = fp.C_eff, fp.chi_eff
kap, eps_p, beta = prm["kappa"], fp.eps_phi, fp.beta_phi
i, j = ufl.indices(2)
energy_form = (2 * kap * (mu - C_eff * phi) ** 2 + 0.5 * chi_eff * phi ** 2 + 0.25 * beta * phi ** 4
               + 0.5 * eps_p * geo.g_c(om)[i, j] * phi.dx(i) * phi.dx(j) + prm["sigma_r_const"]) * sg * dx
number_form = phi * sg * dx
P1 = FunctionSpace(mesh, "P", 1)
xy1 = P1.tabulate_dof_coordinates()


def diag():
    ph = fsp.psi.sub(6, deepcopy=True).vector().get_local()
    zz = fsp.psi.sub(3, deepcopy=True).vector().get_local()
    return dict(G=assemble(energy_form), N=assemble(number_form), phi_min=float(ph.min()), phi_max=float(ph.max()),
                z_min=float(zz.min()), z_max=float(zz.max()))


def snap():
    return (project(fsp.psi.sub(3), P1).vector().get_local().astype(np.float32),
            project(fsp.psi.sub(6), P1).vector().get_local().astype(np.float32))


t_snap = options.t_end / options.n_snap
ckpt = os.path.join(out_dir, "checkpoint.npz")
if os.path.exists(ckpt):
    R = np.load(ckpt, allow_pickle=True)
    psi_n.vector()[:] = R["psi_n"]
    psi_nm1.vector()[:] = R["psi_nm1"]
    fsp.psi.vector()[:] = R["psi_n"]
    rows = list(R["rows"])
    S_t, S_z, S_phi = list(R["S_t"]), list(R["S_z"]), list(R["S_phi"])
    t, dt, n = float(R["t"]), float(R["dt"]), int(R["n"])
    dt_prev = float(R["dt_prev"]) if R["dt_prev"].ndim == 0 and n > 0 else None
    next_snap, first, end = float(R["next_snap"]), n == 0, "t_end"
    print(f"resuming at t = {t:.4e} (step {n})", flush=True)
else:
    rows = [dict(t=0.0, dt=0.0, **diag())]
    S_t, S_z, S_phi = [0.0], [], []
    zs, ps = snap()
    S_z.append(zs)
    S_phi.append(ps)
    next_snap = t_snap
    t, dt, dt_prev, n, first, end = 0.0, options.dt0, None, 0, True, "t_end"


def checkpoint():
    np.savez(ckpt + ".tmp.npz", psi_n=psi_n.vector().get_local(), psi_nm1=psi_nm1.vector().get_local(),
             rows=np.array(rows, dtype=object), S_t=np.array(S_t), S_z=np.array(S_z), S_phi=np.array(S_phi), t=t,
             dt=dt, n=n, dt_prev=dt_prev if dt_prev is not None else 0.0, next_snap=next_snap)
    os.replace(ckpt + ".tmp.npz", ckpt)


def save():
    with open(os.path.join(out_dir, "time_series.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez_compressed(os.path.join(out_dir, "snapshots.npz"), x=xy1[:, 0], y=xy1[:, 1],
                        triangles=vertex_to_dof_map(P1)[mesh.cells()], t=np.array(S_t), z=np.array(S_z),
                        phi=np.array(S_phi), c_r=np.array(fp.rmsh.parameters["c_r"][:2]), r=fp.rmsh.parameters["r"])


while t < options.t_end and n < options.max_steps:
    if first:
        a = (1.0, -1.0, 0.0)
    else:
        w_ = dt / dt_prev
        a = ((1 + 2 * w_) / (1 + w_), -(1 + w_), w_ ** 2 / (1 + w_))
    c0.assign(a[0])
    c1_.assign(a[1])
    c2.assign(a[2])
    dt_c.assign(dt)
    ph_old = psi_n.sub(6, deepcopy=True).vector().get_local()
    z_old = psi_n.sub(3, deepcopy=True).vector().get_local()
    fsp.psi.vector()[:] = psi_n.vector()
    try:
        solver.solve()
    except RuntimeError:
        dt *= 0.3
        if dt < 1e-2:
            end = "newton_failed"
            break
        continue
    dph = float(np.abs(fsp.psi.sub(6, deepcopy=True).vector().get_local() - ph_old).max())
    dz = float(np.abs(fsp.psi.sub(3, deepcopy=True).vector().get_local() - z_old).max())
    ratio = max(dph / options.dphi_target, dz / options.dz_target)
    if ratio > 2.0 and dt > 1e-2:
        dt *= 0.5
        continue
    t += dt
    n += 1
    first = False
    psi_nm1.vector()[:] = psi_n.vector()
    psi_n.vector()[:] = fsp.psi.vector()
    rows.append(dict(t=t, dt=dt, **diag()))
    if t >= next_snap or t >= options.t_end:
        zs, ps = snap()
        S_t.append(t)
        S_z.append(zs)
        S_phi.append(ps)
        next_snap += t_snap
    if n % 10 == 0:
        r = rows[-1]
        print(f"step {n}: t = {t:.4e}, dt = {dt:.3g}, phi in [{r['phi_min']:+.4f}, {r['phi_max']:+.4f}], z in "
              f"[{r['z_min']:+.3f}, {r['z_max']:+.3f}], G = {r['G']:.8e}, N = {r['N']:+.4e}  [{time.time() - t_wall:.0f} s]",
              flush=True)
        save()
    dt_prev = dt
    dt = float(np.clip(dt * min(1.5, 0.9 / max(ratio, 1e-12)), 0.3 * dt, options.dt_max))
    dt = min(dt, options.t_end - t) if options.t_end - t > 1e-9 else dt
    checkpoint()

save()
json.dump(dict(v0=options.v0, SL=options.v0 * L, Q_source=options.Q_source, noise=options.noise, end=end, t_end=t,
               steps=n, params={k: prm[k] for k in ("C_phi", "chi_phi", "eps_phi", "M_phi", "beta_phi", "phi_sponge",
                                                    "sigma_r_const") if k in prm},
               final=rows[-1], wall=time.time() - t_wall), open(os.path.join(out_dir, "run.json"), "w"), indent=1)
print(f"ended ({end}) at t = {t:.4e} after {n} steps", flush=True)
