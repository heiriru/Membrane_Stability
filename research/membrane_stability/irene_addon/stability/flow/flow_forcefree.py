'''
Steady branch of the flowing membrane around a force-free PI with a contact angle (the PRE's setup, t = -0.3), with
the vertical force balance on the PI imposed exactly.

IRENE's square_b leaves the PI height free but its rim equations are not a force balance (README). Here the PI height
h is an unknown: the PI is clamped at height h (pi_height = 1, z = h and w = 0 on the rim, contact slope t), and h is
determined by the force balance F(h, v0) = 0, with F the vertical force of the membrane on the PI computed from the
conserved vertical stress flux (modules/stability/vertical_force.py: Helfrich Noether current + tension + viscous
traction, domain integral; checked against -dE/dh in the ring problem to 5e-5).

Continuation: in v0 (secant on h at each v0) while it converges, then in h (secant on v0 at each h) through the fold
of the imperfect branch and along the unstable branch. Records h, v0, A (projection on the critical mode), max |z|, the
spread of F over three cutoffs (discretization check).

run with:
STABILITY_PARAMS=zeta=1,pi_height=1 python3 flow_forcefree.py square_b [mesh] [out] --t -0.3 --critical_dir [dir]
'''
import irene_paths  # noqa: F401

import argparse
import csv
import json
import os
import time

import numpy as np
from fenics import Constant, Function, assemble, assign, interpolate

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
import differential_geometry.manifold.geometry as geo
from stability import vertical_force as vf

parser = argparse.ArgumentParser()
parser.add_argument("--t", type=float, default=-0.3)
parser.add_argument("--critical_dir", required=True, help="force-free critical point of the flat PI (pi_height = 2)")
parser.add_argument("--v_steps", default="0,0.1,0.2,0.3,0.4,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.84,0.87,0.9,0.92,0.94")
parser.add_argument("--dh", type=float, default=0.25)
parser.add_argument("--h_steps", type=int, default=80)
parser.add_argument("--A_max", type=float, default=20.0)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
t_wall = time.time()
crit = json.load(open(os.path.join(options.critical_dir, "critical.json")))
v0_c = crit["v0_c"]
cm = np.load(os.path.join(options.critical_dir, "critical_mode.npz"))
mode = Function(fsp.Q)
mode.vector()[:] = cm["x_real"]
z_c = mode.sub(3, deepcopy=True)
z_c.vector()[:] /= np.abs(z_c.vector().get_local()).max()
dx = fp.rmsh.dx
zc_norm = assemble(z_c * z_c * dx)
c_r = fp.rmsh.parameters["c_r"][:2]
omega_unit = fp.vp.omega_circle.vector().get_local() / prm["omega_circle_const"]
fp.vp.omega_circle.vector()[:] = omega_unit * options.t


def solve(h, v0):
    fp.z_pi_const.assign(h)
    fp.set_inflow(v0)
    fp.newton()


def force():
    v, w, sigma, z, omega, mu = fsp.psi.split(deepcopy=True)
    d_contra = geo.d_c(v, w, omega)
    return vf.vertical_force(omega, mu, sigma, prm["kappa"], dx, c_r, d_contra=d_contra, eta=prm["eta"])


def amplitude():
    z = fsp.psi.sub(3, deepcopy=True)
    return assemble(z * z_c * dx) / zc_norm, float(np.abs(z.vector().get_local()).max())


def secant(fun, x0, x1, tol=1e-9, max_it=20):
    f0 = fun(x0)
    f1 = fun(x1)
    for _ in range(max_it):
        if abs(f1) < tol:
            return x1
        x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
        x0, f0 = x1, f1
        x1, f1 = x2, fun(x2)
    if abs(f1) < 1e-6:
        return x1
    raise RuntimeError("secant did not converge")


rows = []


def record(h, v0, mode_):
    F = force()
    A, zmax = amplitude()
    rows.append(dict(mode=mode_, v0=v0, v0_over_v0c=v0 / v0_c, SL=v0 * fp.rmsh.parameters["L"], h=h, A=A,
                     max_abs_z=zmax, F=F[1], F_spread=float(max(F) - min(F))))
    print(f"[{mode_}] v0/v0_c = {v0 / v0_c:.5f}, h = {h:+.5f}, A = {A:+.4f}, max|z| = {zmax:.4f}, F = {F[1]:+.2e} "
          f"(spread {max(F) - min(F):.1e})  [{time.time() - t_wall:.0f} s]", flush=True)
    with open(os.path.join(out_dir, "forcefree_branch.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


# start: v0 = 0
fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
state = None


def F_of_h(v0):
    def fun(h):
        fsp.psi.vector()[:] = state
        solve(h, v0)
        return force()[1]
    return fun


fp.z_pi_const.assign(0.0)
fp.set_inflow(0.0)
fp.newton()
state = fsp.psi.vector().get_local().copy()
h = secant(F_of_h(0.0), 0.0, 0.05)
state = fsp.psi.vector().get_local().copy()
record(h, 0.0, "v0")
history = [(0.0, h, state.copy())]
# 1. continuation in v0 (failed steps are halved; the branch turns near the fold, then step 2 takes over)
targets = [float(x) * v0_c for x in options.v_steps.split(",")][1:]
halvings = 0
while targets:
    v0 = targets[0]
    if len(history) >= 2:
        (va, ha, _), (vb, hb, _) = history[-2], history[-1]
        h_pred = hb + (hb - ha) * (v0 - vb) / (vb - va)
    else:
        h_pred = history[-1][1]
    try:
        state = history[-1][2].copy()
        h = secant(F_of_h(v0), h_pred, h_pred + 1e-3 * (1 + abs(h_pred)))
    except RuntimeError:
        halvings += 1
        if halvings > 4:
            print(f"v0-continuation stopped at v0 = {v0 / v0_c:.4f} v0_c", flush=True)
            break
        targets.insert(0, 0.5 * (history[-1][0] + v0))
        continue
    targets.pop(0)
    halvings = 0
    state = fsp.psi.vector().get_local().copy()
    history.append((v0, h, state.copy()))
    record(h, v0, "v0")


# 2. continuation in h: at given h, find v0 with F = 0 (adaptive step in h)
def F_of_v(h):
    def fun(v0):
        fsp.psi.vector()[:] = state
        solve(h, v0)
        return force()[1]
    return fun


sgn = np.sign(history[-1][1] - history[-2][1])
dh = 0.02
for k in range(options.h_steps):
    (va, ha, _), (vb, hb, _) = history[-2], history[-1]
    h_new = hb + sgn * dh
    v_pred = vb + (vb - va) * (h_new - hb) / (hb - ha) if hb != ha else vb
    try:
        state = history[-1][2].copy()
        v0 = secant(F_of_v(h_new), v_pred, v_pred * (1 - 1e-3))
    except RuntimeError:
        dh *= 0.5
        if dh < 1e-3:
            print(f"h-continuation stopped at h = {hb:.4f}", flush=True)
            break
        continue
    state = fsp.psi.vector().get_local().copy()
    history.append((v0, h_new, state.copy()))
    record(h_new, v0, "h")
    dh = min(dh * 1.4, options.dh)
    if abs(rows[-1]["A"]) > options.A_max or v0 <= 0:
        break

vs = np.array([r["v0_over_v0c"] for r in rows])
i_fold = int(np.argmax(vs))
fold_found = 0 < i_fold < len(vs) - 1          # an interior maximum of v0 along the branch
res = dict(t=options.t, v0_c=v0_c, fold_found=bool(fold_found), fold_v0_over_v0c=float(vs[i_fold]),
           fold_A=rows[i_fold]["A"],
           fold_h=rows[i_fold]["h"], h_at_rest=rows[0]["h"], n_points=len(rows), max_F_spread=float(
               max(r["F_spread"] for r in rows)), mesh=rarg.args.input_directory, wall=time.time() - t_wall)
json.dump(res, open(os.path.join(out_dir, "forcefree.json"), "w"), indent=1)
print("FORCEFREE:", json.dumps(res), flush=True)
