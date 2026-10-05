'''
Bifurcation diagram of the flow-driven buckling: steady branches z*(v0) with their stability.

1. Imperfect branches (--t_list): PI with contact slope t (omega = t r_hat on the rim), continuation in v0 from rest
   up to v0_max. For a pitchfork unfolded by t, the amplitude at the threshold of the perfect (t = 0) problem scales as
   A(v0_c) ~ |t|^(1/3), and a fold of the branch would indicate a subcritical bifurcation.
2. Buckled branch of the perfect problem (--postbuckling): at v0 = v0_c (1 + delta), Newton from the flat state plus
   a multiple of the critical mode; the branch is then continued in v0 upwards and downwards (a branch that exists
   below v0_c is subcritical).
The amplitude is the projection A = <z*, z_c> / <z_c, z_c> on the critical mode z_c (normalized to max |z_c| = 1), and
also max |z*|; at every point the leading eigenvalues are computed.

Needs the critical mode of the same setup (flow_critical.py output, --critical_dir). The contact slope is changed by
scaling IRENE's omega_circle, so run with omega_circle_const != 0 (default -0.3 of the csv); t = 0 is then set here.

run with:
STABILITY_PARAMS=zeta=1,pi_height=1 python3 flow_branch.py square_b [mesh] [out] --critical_dir [dir] \
    --t_list -0.1,-0.03,-0.01,-0.003 --v0_max_factor 1.5 --postbuckling
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

parser = argparse.ArgumentParser()
parser.add_argument("--critical_dir", required=True)
parser.add_argument("--t_list", default="-0.1,-0.03,-0.01,-0.003")
parser.add_argument("--v0_max_factor", type=float, default=1.5, help="continue up to this multiple of v0_c")
parser.add_argument("--n_steps", type=int, default=24, help="continuation steps up to v0_max (refined near v0_c)")
parser.add_argument("--postbuckling", action="store_true")
parser.add_argument("--deltas", default="0.02,0.05,0.1,0.2,0.3,0.5")
parser.add_argument("--nev", type=int, default=6)
parser.add_argument("--amplitude_t", default=None,
                    help="contact slopes for the amplitude-controlled continuation (e.g. 0,-0.03,-0.1): the amplitude A is "
                         "imposed and v0 is an unknown (bordered Newton), which follows the branches through folds")
parser.add_argument("--save_shapes", action="store_true", help="store z (P1) at every recorded point in shapes.npz")
parser.add_argument("--amplitudes", default="0.05,0.1,0.2,0.3,0.5,0.75,1,1.5,2,3,4,5,6,8,10")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters

crit = json.load(open(os.path.join(options.critical_dir, "critical.json")))
v0_c = crit["v0_c"]
cm = np.load(os.path.join(options.critical_dir, "critical_mode.npz"))
mode = Function(fsp.Q)
mode.vector()[:] = cm["x_real"]
z_c = mode.sub(3, deepcopy=True)
z_c.vector()[:] /= np.abs(z_c.vector().get_local()).max()
mode.vector()[:] /= np.abs(mode.sub(3, deepcopy=True).vector().get_local()).max()
dx = fp.rmsh.dx
zc_norm = assemble(z_c * z_c * dx)

if prm["omega_circle_const"] == 0:
    raise ValueError("run with omega_circle_const != 0 (the contact slope is set by scaling IRENE's omega_circle)")
omega_unit = fp.vp.omega_circle.vector().get_local() / prm["omega_circle_const"]


def set_slope(t):
    fp.vp.omega_circle.vector()[:] = omega_unit * t


def amplitude():
    z = fsp.psi.sub(3, deepcopy=True)
    return assemble(z * z_c * dx) / zc_norm, float(np.abs(z.vector().get_local()).max())


def leading():
    pairs = fp.eigenpairs(target=0.0, nev=options.nev)
    lead = max(pairs, key=lambda p: p.eigenvalue.real)
    return lead.eigenvalue


rows = []
shapes = []
if options.save_shapes:
    from fenics import FunctionSpace, project, vertex_to_dof_map
    P1 = FunctionSpace(fsp.Q.mesh(), "P", 1)


def record(kind, t, v0):
    A, zmax = amplitude()
    lam = leading()
    rows.append(dict(kind=kind, t=t, v0=v0, v0_over_v0c=v0 / v0_c, SL=prm["eta"] * v0 * fp.rmsh.parameters["L"] /
                     prm["kappa"], A=A, max_abs_z=zmax, lead_re=lam.real, lead_im=abs(lam.imag)))
    print(f"[{kind}] t = {t:+.4f}, v0/v0_c = {v0 / v0_c:.4f}: A = {A:+.5e}, max|z| = {zmax:.4e}, "
          f"leading = {lam.real:+.4e} {abs(lam.imag):+.2e} i", flush=True)
    if options.save_shapes:
        shapes.append(project(fsp.psi.sub(3, deepcopy=True), P1).vector().get_local())
    write()


def write():
    with open(os.path.join(out_dir, "branches.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    if options.save_shapes:
        c = P1.tabulate_dof_coordinates()
        np.savez(os.path.join(out_dir, "shapes.npz"), x=c[:, 0], y=c[:, 1],
                 triangles=vertex_to_dof_map(P1)[fsp.Q.mesh().cells()], z=np.array(shapes),
                 t=np.array([r["t"] for r in rows]), v0_over_v0c=np.array([r["v0_over_v0c"] for r in rows]),
                 A=np.array([r["A"] for r in rows]), lead_re=np.array([r["lead_re"] for r in rows]))


def solve_at(v0, t, previous, max_halvings=5):
    '''Newton at (v0, t) by continuation from 'previous' = (v0_p, t_p, state); halves failed steps'''
    v_p, t_p, state = previous
    fsp.psi.vector()[:] = state
    targets = [(v0, t)]
    current = (v_p, t_p)
    halvings = 0
    while targets:
        v, tt = targets[0]
        fp.set_inflow(v)
        set_slope(tt)
        try:
            fp.newton()
            state = fsp.psi.vector().get_local().copy()
            current = (v, tt)
            targets.pop(0)
        except RuntimeError:
            fsp.psi.vector()[:] = state
            halvings += 1
            if halvings > max_halvings:
                raise
            targets.insert(0, (0.5 * (current[0] + v), 0.5 * (current[1] + tt)))
    return state


def v0_path():
    # denser near v0_c
    a = options.v0_max_factor
    x = np.concatenate([np.linspace(0, 0.8, 5)[1:], np.linspace(0.8, 1.2, 17)[1:],
                        np.linspace(1.2, a, 8)[1:] if a > 1.2 else []])
    return v0_c * x[x <= a + 1e-12]


t0 = time.time()
fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
rest = fsp.psi.vector().get_local().copy()

# 1. imperfect branches (the states above v0_c of the branch with the smallest |t| are kept for the homotopy t -> 0)
t_values = [float(x) for x in options.t_list.split(",") if x]
t_small = min(t_values, key=abs) if t_values else None
kept = []
for t in t_values:
    fp.set_inflow(0.0)
    set_slope(t)
    fsp.psi.vector()[:] = rest
    fp.newton()
    state = fsp.psi.vector().get_local().copy()
    record("imperfect", t, 0.0)
    v_prev = 0.0
    for v0 in v0_path():
        try:
            state = solve_at(v0, t, (v_prev, t, state))
        except RuntimeError:
            print(f"t = {t}: continuation failed beyond v0/v0_c = {v_prev / v0_c:.4f} (fold?)", flush=True)
            break
        v_prev = v0
        record("imperfect", t, v0)
        if t == t_small and v0 > 1.01 * v0_c:
            kept.append((v0, state.copy()))
    print(f"t = {t} done [{time.time() - t0:.0f} s]", flush=True)

# 2a. buckled branch of the perfect problem by the homotopy t -> 0 at fixed v0 > v0_c
if options.postbuckling and kept:
    for v0, state in kept:
        t_prev = t_small
        try:
            for factor in (0.5, 0.25, 0.1, 0.0):
                state = solve_at(v0, factor * t_small, (v0, t_prev, state))
                t_prev = factor * t_small
        except RuntimeError:
            print(f"homotopy t -> 0 failed at v0/v0_c = {v0 / v0_c:.4f}, t = {t_prev}", flush=True)
            continue
        record("postbuckling_homotopy", 0.0, v0)

# 2b. buckled branch of the perfect problem from the flat state plus the critical mode, continued in v0
if options.postbuckling and not kept:
    set_slope(0.0)
    for sign in (+1, -1):
        found = None
        for delta in [float(x) for x in options.deltas.split(",")]:
            v0 = v0_c * (1 + delta)
            fp.set_inflow(v0)
            fsp.psi.vector()[:] = rest
            fp.newton()     # flat state at v0
            flat = fsp.psi.vector().get_local().copy()
            for a in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
                fsp.psi.vector()[:] = flat + sign * a * mode.vector().get_local()
                try:
                    fp.newton(max_iterations=60)
                except RuntimeError:
                    continue
                A, _ = amplitude()
                if abs(A) > 1e-3:
                    found = (v0, fsp.psi.vector().get_local().copy())
                    break
            if found:
                break
        if not found:
            print(f"postbuckling (sign {sign:+d}): no buckled state found for the deltas tried", flush=True)
            continue
        v_start, state_start = found
        record("postbuckling", 0.0, v_start)
        # continue up, then down
        for direction in (+1, -1):
            state, v_prev = state_start, v_start
            step = 0.02 * v0_c
            while True:
                v0 = v_prev + direction * step
                if v0 > options.v0_max_factor * v0_c or v0 < 0.3 * v0_c:
                    break
                try:
                    state = solve_at(v0, 0.0, (v_prev, 0.0, state))
                except RuntimeError:
                    print(f"postbuckling: continuation failed beyond v0/v0_c = {v_prev / v0_c:.4f}", flush=True)
                    break
                A, _ = amplitude()
                v_prev = v0
                record("postbuckling", 0.0, v0)
                if abs(A) < 1e-3:
                    print("postbuckling: back on the flat branch", flush=True)
                    break
                if direction < 0 and v0 < v0_c:
                    step = 0.01 * v0_c
# 3. amplitude-controlled continuation: unknowns (psi, v0), equations F(psi; v0) = 0 and <z, z_c> = A <z_c, z_c>
if options.amplitude_t is not None:
    from fenics import PETScMatrix, derivative, TestFunction
    from petsc4py import PETSc
    import ufl

    g_psi = assemble(z_c * fsp.nu_z * dx).get_local() / zc_norm       # d A / d psi
    J_form = derivative(fp.F_used, fsp.psi, fsp.J_psi)

    def residual(v0):
        fp.set_inflow(v0)
        b = assemble(fp.F_used)
        for bc in fp.bcs:
            bc.apply(b, fsp.psi.vector())
        return b.get_local()

    def bordered_newton(A_target, v0, tol=1e-10, max_iterations=25):
        for it in range(max_iterations):
            R = residual(v0)
            eps = 1e-6 * max(abs(v0), 1e-3)
            R_v = (residual(v0 + eps) - R) / eps                            # only the inflow rows depend on v0
            fp.set_inflow(v0)
            gval = float(np.dot(g_psi, fsp.psi.vector().get_local())) - A_target
            if np.linalg.norm(R) < tol * max(1.0, np.sqrt(len(R))) and abs(gval) < 1e-9 * max(1.0, abs(A_target)):
                return v0, it
            J = PETScMatrix()
            assemble(J_form, tensor=J)
            for bc in fp.bcs:
                bc.apply(J)
            ksp = PETSc.KSP().create()
            ksp.setOperators(J.mat())
            ksp.setType("preonly")
            ksp.getPC().setType("lu")
            ksp.getPC().setFactorSolverType("mumps")
            rhs, sol = J.mat().createVecs()
            rhs.setArray(-R)
            ksp.solve(rhs, sol)
            a_ = sol.getArray().copy()
            rhs.setArray(-R_v)
            ksp.solve(rhs, sol)
            b_ = sol.getArray().copy()
            ksp.destroy()
            dv = (-gval - np.dot(g_psi, a_)) / np.dot(g_psi, b_)
            fsp.psi.vector()[:] = fsp.psi.vector().get_local() + a_ + dv * b_
            v0 = v0 + dv
        raise RuntimeError("bordered Newton did not converge")

    amps = [float(x) for x in options.amplitudes.split(",")]
    for t in [float(x) for x in options.amplitude_t.split(",")]:
        set_slope(t)
        fp.set_inflow(0.0)
        fsp.psi.vector()[:] = rest
        fp.newton()
        A0, _ = amplitude()
        if t == 0:
            # start at the bifurcation point, along the critical mode
            fp.set_inflow(v0_c)
            fsp.psi.vector()[:] = rest
            fp.newton()
            flat = fsp.psi.vector().get_local().copy()
            history = [(0.0, v0_c, flat)]
        else:
            history = [(A0, 0.0, fsp.psi.vector().get_local().copy())]
            for v in (0.25 * v0_c, 0.5 * v0_c):
                fp.set_inflow(v)
                fp.newton()
                history.append((amplitude()[0], v, fsp.psi.vector().get_local().copy()))
        targets = [a for a in amps if a > history[-1][0] * 1.02]
        for A_target in targets:
            # predictor: secant in A (or along the critical mode from the bifurcation point)
            A1, v1, x1 = history[-1]
            if len(history) >= 2:
                A2, v2, x2 = history[-2]
                w = (A_target - A1) / (A1 - A2)
                x_pred, v_pred = x1 + w * (x1 - x2), v1 + w * (v1 - v2)
            else:
                x_pred = x1 + (A_target - A1) * mode.vector().get_local() if t == 0 else x1
                v_pred = v1 if t == 0 else 0.5 * v0_c
            fsp.psi.vector()[:] = x_pred
            try:
                v_sol, its = bordered_newton(A_target, v_pred)
            except RuntimeError:
                print(f"[amplitude] t = {t}: no convergence at A = {A_target}", flush=True)
                break
            history.append((A_target, v_sol, fsp.psi.vector().get_local().copy()))
            fp.set_inflow(v_sol)
            record("amplitude", t, v_sol)
        print(f"amplitude continuation t = {t} done [{time.time() - t0:.0f} s]", flush=True)

json.dump(dict(v0_c=v0_c, critical_dir=options.critical_dir, pi_height=fp.PI_HEIGHT, time=time.time() - t0),
          open(os.path.join(out_dir, "branches.json"), "w"), indent=1)
print("... done.", flush=True)
