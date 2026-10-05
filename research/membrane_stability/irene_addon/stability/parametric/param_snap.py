'''
Post-snap dynamics beyond the Monge gauge: time stepping of the parametric surface (param_problem.py) from a state of
the Monge run (../flow/flow_snap_ff.py snapshots), with the force-free PI (h found at every step by a secant iteration
on the vertical force) or a clamped PI (--clamped).

Validation: started from a Monge snapshot, the parametric run must follow the Monge run while the surface is a graph
(compare h(t), z_min(t), z_max(t)); it then continues past the slope at which the Monge description fails.

run with:
STABILITY_PARAMS=zeta=1,gauge_delta=0.05 python3 param_snap.py square_b [mesh] [out] --monge_run [snapff dir]
    --critical_dir [crit] --v0_factor 0.97 --start_index 4
'''
import irene_paths  # noqa: F401

import argparse
import csv
import json
import os
import time

import numpy as np
from fenics import (Constant, DirichletBC, Function, NonlinearVariationalProblem, NonlinearVariationalSolver, assemble,
                    derivative, project, split, vertex_to_dof_map)
import ufl

import runtime_arguments as rarg
import param_problem as pp
import differential_geometry.manifold.geometry as geo

parser = argparse.ArgumentParser()
parser.add_argument("--monge_run", default=None)
parser.add_argument("--report", action="store_true", help="print the Newton iterations")
parser.add_argument("--smooth", type=float, default=1.5, help="smoothing length of the transferred surface / cell size")
parser.add_argument("--transfer", default=None, help="start from param_remesh.py transfer data (new mesh = input dir)")
parser.add_argument("--remesh_aniso", type=float, default=0.0,
                    help="end the segment with end = 'remesh' when the anisotropy of the parametrization exceeds this")
parser.add_argument("--remesh_stretch", type=float, default=8.0,
                    help="... or when the largest area stretch exceeds this factor times its value at the start")
parser.add_argument("--critical_dir", required=True)
parser.add_argument("--v0_factor", type=float, default=0.97)
parser.add_argument("--t_slope", type=float, default=-0.3)
parser.add_argument("--start_index", type=int, default=4)
parser.add_argument("--clamped", action="store_true")
parser.add_argument("--dt0", type=float, default=50.0)
parser.add_argument("--dt_max", type=float, default=5e4)
parser.add_argument("--dX_target", type=float, default=0.8)
parser.add_argument("--max_steps", type=int, default=400)
parser.add_argument("--t_max", type=float, default=1e7)
parser.add_argument("--dt_resume", type=float, default=0.0, help="time step after resuming (0: the stored one)")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
t_wall = time.time()
v0_c = json.load(open(os.path.join(options.critical_dir, "critical.json")))["v0_c"]
if options.transfer:
    from matplotlib.tri import LinearTriInterpolator, Triangulation
    from scipy.spatial import cKDTree
    TR = np.load(options.transfer, allow_pickle=True)
    t_start, h = float(TR["t"]), float(TR["h"])
    _tri = Triangulation(TR["u"][:, 0], TR["u"][:, 1], TR["triangles"])
    _tree = cKDTree(TR["u"])

    def _interp(values, pts):
        f = LinearTriInterpolator(_tri, values)(pts[:, 0], pts[:, 1])
        out_ = np.asarray(f.filled(np.nan))
        bad = np.isnan(out_)
        if bad.any():
            out_[bad] = values[_tree.query(pts[bad])[1]]
        return out_

    _P1 = pp.P1
    _xy = _P1.tabulate_dof_coordinates()
    _f = lambda vals: (lambda fn: (fn.vector().set_local(_interp(vals, _xy)), fn)[1])(Function(_P1))  # noqa: E731
    Xc = [_f(TR["X_n"][:, k]) for k in range(3)]
    Vc = [_f(TR["V"][:, k]) for k in range(3)]
    from fenics import assign as _assign, as_vector as _av, as_tensor as _at
    # the interpolated surface inherits the facets of the old (coarser) triangulation: smooth them at the scale of the
    # new mesh, (X_s, nu) + (tau grad X_s, grad nu) = (X, nu) with tau = (c h)^2, the surface staying fixed on the
    # boundary (the boundary conditions are imposed below)
    from fenics import (TrialFunction as _TF, TestFunction as _Tf, CellDiameter as _CD, inner as _in, grad as _gr,
                        solve as _solve, DirichletBC as _DBC)
    _P1 = pp.P1
    _hc = _CD(pp.mesh)
    for k in range(3):
        _u, _v = _TF(_P1), _Tf(_P1)
        _Xs = Function(_P1)
        _bcs = [_DBC(_P1, Xc[k], "on_boundary")]
        _solve(_u * _v * pp.dx + (options.smooth * _hc) ** 2 * _in(_gr(_u), _gr(_v)) * pp.dx == Xc[k] * _v * pp.dx,
               _Xs, _bcs)
        Xc[k] = _Xs
    _Xf = project(_av(Xc), pp.Q.sub(3).collapse())
    if "E" in TR.files:
        _Ec = [[_f(TR["E"][:, a, k]) for k in range(3)] for a in range(2)]
        _Ef = project(_at(_Ec), pp.Q.sub(4).collapse())
    else:
        _Ef = project(_at([[Xc[k].dx(a) for k in range(3)] for a in range(2)]), pp.Q.sub(4).collapse())
    _assign(pp.psi.sub(3), _Xf)
    _assign(pp.psi.sub(4), _Ef)
    _assign(pp.psi.sub(1), _f(TR["w"]))
    _assign(pp.psi.sub(2), _f(TR["sigma"]))
    _assign(pp.psi.sub(3), _Xf)
    _assign(pp.psi.sub(4), _Ef)
    _assign(pp.psi.sub(5), _f(TR["mu"]) if "E" in TR.files else project(geo.H(pp.E), _P1))   # mean curvature
    _E = _Ef
    _g = _at([[sum(_E[a, k] * _E[b, k] for k in range(3)) for b in range(2)] for a in range(2)])
    _gi = ufl.inv(_g)
    _ve = [sum(_E[b, k] * Vc[k] for k in range(3)) for b in range(2)]
    _assign(pp.psi.sub(0), project(_av([_gi[a, 0] * _ve[0] + _gi[a, 1] * _ve[1] for a in range(2)]),
                                   pp.Q.sub(0).collapse()))
    pp.set_h(h)
    pp.set_slope(options.t_slope)
    pp.set_inflow(options.v0_factor * v0_c)
    for _bc in pp.bcs:
        _bc.apply(pp.psi.vector())
    _res = assemble(pp.F_static).get_local()
    for _bc in pp.bcs:
        _res[list(_bc.get_boundary_values().keys())] = 0.0
    print("static residual per field: " + ", ".join(f"{nm} {np.abs(_res[pp.Q.sub(k).dofmap().dofs()]).max():.3g}"
          for k, nm in enumerate(("v", "w", "sigma", "X", "E", "mu"))), flush=True)
    if options.report:
        _xyq = pp.Q.tabulate_dof_coordinates()
        for k, nm in enumerate(("v", "w", "E", "mu")):
            kk = {"v": 0, "w": 1, "E": 4, "mu": 5}[nm]
            dd = np.array(pp.Q.sub(kk).dofmap().dofs())
            top = dd[np.argsort(-np.abs(_res[dd]))[:4]]
            print(f"  largest {nm} residuals at " + "; ".join(f"({_xyq[q, 0]:.2f}, {_xyq[q, 1]:.2f}): {_res[q]:+.3g}"
                                                            for q in top), flush=True)
    _pv = pp.psi.vector().get_local()
    print(f"start from the transfer data {options.transfer} at t = {t_start:.6g} (h = {h:+.4f}); NaN: "
          f"{int(np.isnan(_pv).sum())}, field max: " + ", ".join(
              f"{nm} {np.abs(_pv[pp.Q.sub(k).dofmap().dofs()]).max():.3g}"
              for k, nm in enumerate(("v", "w", "sigma", "X", "E", "mu"))), flush=True)
else:
    snaps = np.load(os.path.join(options.monge_run, "snapshots.npz"))
    ts_m = list(csv.DictReader(open(os.path.join(options.monge_run, "time_series.csv"))))
    t_start = float(snaps["t"][options.start_index])
    h0 = min(ts_m, key=lambda r: abs(float(r["t"]) - t_start))
    h = float(h0["h"])
    pp.set_from_monge(snaps["z"][options.start_index], h, options.t_slope)
    pp.set_inflow(options.v0_factor * v0_c)
    print(f"start from the Monge state at t = {t_start:.6g} (h = {h:+.4f}), v0 = {options.v0_factor} v0_c", flush=True)

psi, Q = pp.psi, pp.Q
psi_n, psi_nm1 = Function(Q), Function(Q)
psi_n.vector()[:] = psi.vector()
psi_nm1.vector()[:] = psi.vector()
dt_c, c0, c1_, c2 = Constant(options.dt0), Constant(1.0), Constant(-1.0), Constant(0.0)
X1 = split(psi_n)[3]
X2 = split(psi_nm1)[3]
F_time = (ufl.dot((c0 * pp.X + c1_ * X1 + c2 * X2) / dt_c - pp.X_velocity, pp.nu_X)) * pp.sg * pp.dx + pp.F_static
_problem = NonlinearVariationalProblem(F_time, psi, pp.bcs, derivative(F_time, psi, pp.J_psi))


def make_solver():
    # a new solver after every failure: a diverged Newton iteration can leave the MUMPS factorization in a failed
    # state, after which every later solve fails immediately (even for tiny time steps)
    sol = NonlinearVariationalSolver(_problem)
    sol.parameters["newton_solver"].update({"linear_solver": "mumps", "absolute_tolerance": 1e-9,
                                            "relative_tolerance": 1e-9, "maximum_iterations": 20,
                                            "report": options.report, "error_on_nonconvergence": True})
    return sol


solver = make_solver()
P1 = pp.P1
xy1 = P1.tabulate_dof_coordinates()
tri = vertex_to_dof_map(P1)[pp.mesh.cells()]


def X_P1():
    return np.array([project(pp.X[c], P1).vector().get_local() for c in range(3)]).T


def diag():
    Xv = X_P1()
    nz = project(pp.n3[2], P1).vector().get_local()
    sgv = project(pp.sg, P1).vector().get_local()
    drift = np.hypot(Xv[:, 0] - xy1[:, 0], Xv[:, 1] - xy1[:, 1])
    Ev = np.array([[project(pp.E[a, k], P1).vector().get_local() for k in range(3)] for a in range(2)])  # (2, 3, n)
    gv = np.einsum("akn,bkn->nab", Ev, Ev)
    ev = np.linalg.eigvalsh(gv)
    aniso = np.sqrt(np.maximum(ev[:, 1], 1e-30) / np.maximum(ev[:, 0], 1e-30))
    return dict(z_min=float(Xv[:, 2].min()), z_max=float(Xv[:, 2].max()), min_nz=float(nz.min()),
                max_slope=float(np.sqrt(max(1.0 / max(nz.min(), 1e-6) ** 2 - 1.0, 0.0))) if nz.min() > 0 else float("inf"),
                max_drift=float(drift.max()), stretch_max=float(sgv.max()), stretch_min=float(sgv.min()),
                aniso_max=float(aniso.max())), Xv


def secant(fun, x0, x1, tol=1e-8, max_it=15):
    f0, f1 = fun(x0), fun(x1)
    for _ in range(max_it):
        if abs(f1) < tol:
            return x1
        if f1 == f0:
            break
        x0, f0, x1 = x1, f1, x1 - f1 * (x1 - x0) / (f1 - f0)
        f1 = fun(x1)
    if abs(f1) < 1e-5:
        return x1
    raise RuntimeError("secant did not converge")


def step_with_h(hh):
    psi.vector()[:] = psi_n.vector()
    pp.set_h(hh)
    solver.solve()
    return pp.vertical_force()[1]


# first: a short backward-Euler step at fixed h to put v, w, sigma, mu, E on the constraint manifold, then report the
# force at the Monge h (should be ~ 0 if the parametric force matches the Monge force balance)
dt_c.assign(1.0)
if os.path.exists(os.path.join(out_dir, "checkpoint.npz")):
    F0 = [0.0, 0.0, 0.0]
else:
    solver.solve()
    F0 = pp.vertical_force()
print(f"initial solve done; vertical force at the Monge h: {F0}  [{time.time() - t_wall:.0f} s]", flush=True)
psi_n.vector()[:] = psi.vector()
psi_nm1.vector()[:] = psi.vector()

ckpt = os.path.join(out_dir, "checkpoint.npz")
if os.path.exists(ckpt):
    R = np.load(ckpt, allow_pickle=True)
    psi_n.vector()[:] = R["psi_n"]
    psi_nm1.vector()[:] = R["psi_nm1"]
    psi.vector()[:] = R["psi_n"]
    rows, S_t, S_X = list(R["rows"]), list(R["S_t"]), list(R["S_X"])
    t, dt, dt_prev, n, h, h_prev = (float(R["t"]), float(R["dt"]), float(R["dt_prev"]), int(R["n"]), float(R["h"]),
                                    float(R["h_prev"]))
    pp.set_h(h)
    first, end = False, "max_steps"
    Xv = S_X[-1].astype(float)
    if options.dt_resume:
        dt = options.dt_resume
    print(f"resuming at t = {t:.6g} (step {n}), h = {h:+.4f}", flush=True)
else:
    d0, Xv = diag()
    rows = [dict(t=t_start, dt=0.0, h=h, F_spread=float(max(F0) - min(F0)), F_at_h=F0[1], **d0)]
    S_t, S_X = [t_start], [Xv.astype(np.float32)]
    t, dt, dt_prev, n, first, end = t_start, options.dt0, None, 0, True, "max_steps"
    h_prev = h
    rows[0]["h_frozen"] = 0


def checkpoint():
    np.savez(ckpt + ".tmp.npz", psi_n=psi_n.vector().get_local(), psi_nm1=psi_nm1.vector().get_local(),
             rows=np.array(rows, dtype=object), S_t=np.array(S_t), S_X=np.array(S_X), t=t, dt=dt,
             dt_prev=dt_prev if dt_prev is not None else dt, n=n, h=h, h_prev=h_prev)
    os.replace(ckpt + ".tmp.npz", ckpt)


def save():
    with open(os.path.join(out_dir, "time_series.csv"), "w", newline="") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        wr = csv.DictWriter(fh, fieldnames=keys, restval="")
        wr.writeheader()
        wr.writerows(rows)
    np.savez_compressed(os.path.join(out_dir, "snapshots.npz"), u=xy1, triangles=tri, t=np.array(S_t),
                        X=np.array(S_X), c_r=np.array(pp.c_r))


h_flag = 0
_s0f = os.path.join(out_dir, "stretch0.json")
stretch0 = json.load(open(_s0f))["stretch0"] if os.path.exists(_s0f) else float(diag()[0]["stretch_max"])
json.dump(dict(stretch0=stretch0), open(_s0f, "w"))
print(f"largest area stretch at the start of the segment: {stretch0:.2f}", flush=True)
while n < options.max_steps and t < options.t_max:
    if first:
        a = (1.0, -1.0, 0.0)
    else:
        om = dt / dt_prev
        a = ((1 + 2 * om) / (1 + om), -(1 + om), om ** 2 / (1 + om))
    c0.assign(a[0])
    c1_.assign(a[1])
    c2.assign(a[2])
    dt_c.assign(dt)
    X_old = Xv
    try:
        if options.clamped:
            psi.vector()[:] = psi_n.vector()
            solver.solve()
            h_new = h
        else:
            guess = h + (h - h_prev) * (dt / dt_prev if dt_prev else 0.0)
            try:
                h_new = secant(step_with_h, guess, guess + 1e-3)
                h_flag = 0
            except RuntimeError:
                # force-free search failed: take the step with the PI height of the previous step (flagged)
                solver = make_solver()
                psi.vector()[:] = psi_n.vector()
                pp.set_h(h)
                solver.solve()
                h_new, h_flag = h, 1
                print("  secant failed: step taken at the previous PI height", flush=True)
    except RuntimeError:
        solver = make_solver()
        dt *= 0.3
        print(f"  no convergence: dt -> {dt:.3g}", flush=True)
        if dt < 1e-3:
            end = "newton_failed"
            break
        continue
    d, Xv = diag()
    dX = float(np.abs(Xv - X_old).max())
    if dX > 2 * options.dX_target and dt > 1e-2:
        dt *= 0.5
        Xv = X_old
        continue
    Fs = pp.vertical_force()
    t += dt
    n += 1
    first = False
    psi_nm1.vector()[:] = psi_n.vector()
    psi_n.vector()[:] = psi.vector()
    h_prev, h = h, h_new
    rows.append(dict(t=t, dt=dt, h=h, F_spread=float(max(Fs) - min(Fs)), F_at_h=Fs[1], **d))
    rows[-1]["h_frozen"] = h_flag if not options.clamped else 1
    S_t.append(t)
    S_X.append(Xv.astype(np.float32))
    print(f"step {n}: t = {t:.6g}, dt = {dt:.3g}, h = {h:+.4f}, z in [{d['z_min']:+.3f}, {d['z_max']:+.3f}], "
          f"min n_z = {d['min_nz']:+.4f} (slope {d['max_slope']:.2f}), drift {d['max_drift']:.3f}, "
          f"stretch [{d['stretch_min']:.3f}, {d['stretch_max']:.2f}], F spread {rows[-1]['F_spread']:.2e}  "
          f"[{time.time() - t_wall:.0f} s]", flush=True)
    if n % 5 == 0:
        save()
    dt_prev = dt
    dt = float(np.clip(dt * min(2.0, 0.8 * options.dX_target / max(dX, 1e-12)), 0.2 * dt, options.dt_max))
    dt = min(dt, 2.0 * dt_prev)
    checkpoint()
    if options.remesh_aniso and (d["aniso_max"] > options.remesh_aniso or d["stretch_max"] > options.remesh_stretch * stretch0):
        end = "remesh"
        print(f"parametrization degenerates (anisotropy {d['aniso_max']:.1f}, stretch {d['stretch_max']:.1f}): remesh",
              flush=True)
        break

save()
json.dump(dict(transfer=options.transfer, v0_c=v0_c, v0_factor=options.v0_factor, t_slope=options.t_slope, start_index=options.start_index,
               t_start=t_start, clamped=options.clamped, end=end, t_end=t, steps=n, final=rows[-1],
               gauge_delta=pp.DELTA, gauge_eq=pp.D_EQ, monge_run=options.monge_run, wall=time.time() - t_wall),
          open(os.path.join(out_dir, "param_snap.json"), "w"), indent=1)
print(f"ended ({end}) at t = {t:.6g} after {n} steps", flush=True)
