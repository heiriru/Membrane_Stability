'''
Nonlinear coefficient of the long-wave (lattice) amplitude equation: homogenization of a uniformly tilted lattice.

Long-wave theory of the lattice of anchored PIs (study_lattice.py, lattice_longwave.py): for the mean height H(x, t)
    zeta H_t = d_x J(H_x) - zeta c H_x + ...,      J(u) = sigma_eff u + g u^3 + ...,
where J(u) is the macroscopic vertical stress (force per unit length) carried by the lattice when it is tilted with
slope u. The linear coefficient sigma_eff is the effective tension found from the Bloch spectrum; the cubic coefficient
g decides the nonlinear long-wave dynamics (slope equation of convective Cahn-Hilliard type):
    g > 0: slopes saturate at |u| = sqrt(-sigma_eff / g) beyond threshold (terraces / facets, supercritical);
    g < 0: no saturation (subcritical, collapse).

Cell problem: on the periodic cell, z = u x + p(x, y) with p periodic; the PIs are rigid, follow the mean surface in
height (z = u x + p constant on each rim: p = h - u (x - c_x)) and keep zero slope (omega = grad z = 0 on the rims, IRENE's PI condition for t = 0, as in the
linear lattice analysis): tilting the lattice bends every cell, which gives the bending part of sigma_eff. The normal force balance is IRENE's
nonlinear one (Monge gauge, geometry module of IRENE: Helfrich bending in mixed form with omega = grad z, mu = H, the
tension and viscous normal forces 2 sigma H + 2 eta d^{ij} b_ij), with the in-plane flow and tension frozen at the
flat base state (v, sigma from plate_lib.BaseFlow at drive D). This keeps the geometric nonlinearity, which carries
96 % of the cubic Landau coefficient of the single PI in the box (flow_landau.py: the in-plane readjustment x2 gives
the other 4 %). J(u) is the cell average of the vertical stress flux J_x (modules/stability/vertical_force.py) over the
strips away from the PI. Check: J(u)/u at small u against sigma_eff from the Bloch spectrum.

    python3 nonlinear_cell.py [lattice run dir] --Lc 8 --us 0.02,0.05,0.1,0.2,0.3
'''
import argparse
import json
import os
import sys

import numpy as np
from dolfin import (Constant, DirichletBC, Expression, FiniteElement, Function, FunctionSpace, MixedElement,
                    NonlinearVariationalProblem, NonlinearVariationalSolver, TestFunctions, VectorElement,
                    as_vector, assemble, derivative, split, TrialFunction)
import ufl

import plate_lib as pl

sys.path.insert(0, "/home/fenics/shared/modules")
import differential_geometry.manifold.geometry as geo  # noqa: E402
import differential_geometry.manifold.gauges.monge_gauge as _monge  # noqa: E402
for _name in ("e", "K", "normal", "X"):          # IRENE's command.set_gauge('monge'), manifold part only
    setattr(geo, _name, getattr(_monge, _name))
from stability import vertical_force as vf  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("run")
parser.add_argument("--Lc", type=float, default=8.0)
parser.add_argument("--us", default="0.01,0.02,0.05,0.1,0.15,0.2,0.3")
parser.add_argument("--drives", default="zero,half,lw", help="zero, half, lw: D = 0, D_lw / 2, D_lw")
parser.add_argument("--pi_height", choices=["clamped", "free"], default="free",
                    help="PI height relative to the mean surface: clamped (p = 0 on the rim) or free (rigid, force-free:"
                         " p = const on the rim, tied)")
parser.add_argument("--angle", type=float, default=0.0,
                    help="direction of the tilt in degrees from the flow (x) axis: slope vector u (cos, sin); the "
                         "cell averages of both flux components are recorded (2D slope equation)")
parser.add_argument("--kappa", type=float, default=1.0)
parser.add_argument("--eta", type=float, default=1.0)
args = parser.parse_args()
lw = [r for r in json.load(open(os.path.join(args.run, "longwave.json"))) if r["Lc"] == args.Lc][0]
info = [r for r in json.load(open(os.path.join(args.run, "lattice.json"))) if r["Lc"] == args.Lc]
info = info[0] if info else lw
geom = pl.Geometry(os.path.join(args.run, f"mesh_L{args.Lc:g}"))
mesh = geom.mesh
if lw["drive"] == "force":
    base = pl.BaseFlow(geom, f0_unit=1.0)
else:
    base = pl.BaseFlow(geom, b=1.0 / info["ell_b"] ** 2, V_unit=1.0)
sigma0 = lw["sigma0"]
kappa, eta = args.kappa, args.eta

P2 = FiniteElement("P", mesh.ufl_cell(), 2)
V2 = VectorElement("P", mesh.ufl_cell(), 2)
R = FiniteElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([P2, V2, P2, R]), constrained_domain=geom.pbc)
sol = Function(W)
p, zeta_, mu, h_pi = split(sol)
nu_n, nu_o, nu_m, nu_h = TestFunctions(W)
u_c = Constant(0.0)
D_c = Constant(0.0)
ca, sa = np.cos(np.radians(args.angle)), np.sin(np.radians(args.angle))
if abs(sa) < 1e-12:
    sa = 0.0
if abs(ca) < 1e-12:
    ca = 0.0
omega = as_vector((ca * u_c + zeta_[0], sa * u_c + zeta_[1]))
v = D_c * base.v
sigma = sigma0 + D_c * base.s
w0 = Constant(0.0)
i, j, k = ufl.indices(3)
sg = geo.sqrt_detg(omega)
dx = geom.dx
# IRENE's steady normal force balance with w = 0, rho = 0 (F_w), and the mixed constraints (F_omega, F_mu) written
# for the periodic part p (omega = u e_x + zeta, zeta = grad p)
F_w = (2.0 * kappa * (-geo.g_c(omega)[i, j] * mu.dx(i) * nu_n.dx(j) + 2.0 * mu * (mu ** 2 - geo.K(omega)) * nu_n)
       - (2.0 * sigma * mu + 2.0 * eta * geo.g_c(omega)[i, k] * geo.Nabla_v(v, omega)[j, k] * geo.b(omega)[i, j])
       * nu_n) * sg * dx
F_o = (p * (nu_o[0].dx(0) + nu_o[1].dx(1)) + zeta_[0] * nu_o[0] + zeta_[1] * nu_o[1]) * dx
F_m = (geo.H(omega) - mu) * nu_m * sg * dx
F = F_w + F_o + F_m
# PI height: force-free (free) = rigid PI whose height h relative to the mean surface is an unknown, coupled to the rim
# by a stiff penalty; the h-row sets the net penalty force (the vertical force on the PI) to zero.
# clamped: h fixed to 0 (p = 0 on the rim, Dirichlet below) and the h-row is trivial.
rim = sum((geom.ds(pl.HOLE0 + kk) for kk in range(1, len(geom.holes))), geom.ds(pl.HOLE0))
beta = Constant(1e4)
# a rigid horizontal PI has constant z = u x + p on its rim: p = h - u (x - c_x) there
xx = ufl.SpatialCoordinate(mesh)
p_rim = p + u_c * (ca * (xx[0] - geom.holes[0][0]) + sa * (xx[1] - geom.holes[0][1]))
if args.pi_height == "free":
    F = F + beta * (p_rim - h_pi) * nu_n * rim + beta * (h_pi - p_rim) * nu_h * rim
else:
    F = F + h_pi * nu_h * dx + beta * p_rim * nu_n * rim
bcs = []
zeta_rim = Constant((0.0, 0.0))   # the PIs keep zero slope (IRENE's contact condition, t = 0): omega = 0 on the rim
if args.pi_height == "free":
    # remove the neutral uniform lift (p and h together): p = 0 at the cell corner
    bcs.append(DirichletBC(W.sub(0), Constant(0.0), "near(x[0], 0.0) && near(x[1], 0.0)", method="pointwise"))
for kk in range(len(geom.holes)):
    bcs.append(DirichletBC(W.sub(1), zeta_rim, geom.facets, pl.HOLE0 + kk))
solver = NonlinearVariationalSolver(NonlinearVariationalProblem(F, sol, bcs, derivative(F, sol, TrialFunction(W))))
sp = solver.parameters["newton_solver"]
sp.update({"linear_solver": "mumps", "absolute_tolerance": 1e-11, "relative_tolerance": 1e-10,
           "maximum_iterations": 30, "report": False})

# strip average of the vertical stress flux J_x (away from the PI, |x - c| > 2.5)
cx = geom.holes[0][0]
strip = Expression("fabs(x[0] - cx) > 2.5 ? 1.0 : 0.0", cx=cx, degree=0, domain=mesh)
strip_area = assemble(strip * dx)
cy = geom.holes[0][1]
strip_y = Expression("fabs(x[1] - cy) > 2.5 ? 1.0 : 0.0", cy=cy, degree=0, domain=mesh)
strip_y_area = assemble(strip_y * dx)


def J_bar(component=0):
    d_contra = geo.d_c(v, w0, omega)
    J = vf.flux(omega, mu, sigma, kappa, d_contra, eta)
    if component == 1:
        return assemble(J[1] * strip_y * dx) / strip_y_area   # average flux density across horizontal cuts
    return assemble(J[0] * strip * dx) / strip_area * 1.0     # average flux density across vertical cuts


out = dict(Lc=args.Lc, angle=args.angle, phi=lw["phi"], drive=lw["drive"], sigma0=sigma0, D_lw=lw["D_lw"], runs=[])
for dname in args.drives.split(","):
    D = {"lw": lw["D_lw"], "half": 0.5 * lw["D_lw"], "zero": 0.0}[dname]
    D_c.assign(D)
    sol.vector().zero()
    rows = []
    failed_at = None
    for u in [float(x) for x in args.us.split(",")]:
        # continuation in u (the previous solution is the initial guess)
        u_c.assign(u)
        zeta_rim.assign(Constant((-ca * u, -sa * u)))
        try:
            solver.solve()
        except RuntimeError:
            print(f"D = {D:.5f} ({dname}): no convergence at u = {u} (the tilted cell loses stability)", flush=True)
            failed_at = u
            break
        Jx, Jy = J_bar(0), J_bar(1)
        Jb = ca * Jx + sa * Jy                                   # flux along the tilt
        E_cell = assemble((2.0 * kappa * mu ** 2 + sigma) * sg * dx) - assemble(sigma * dx)
        sig_E = 2.0 * E_cell / (geom.Lx * geom.Ly) / u ** 2          # variational check (no flow): 2 E / (A u^2)
        rows.append(dict(u=u, ux=ca * u, uy=sa * u, Jx=Jx, Jy=Jy, J=Jb, J_over_u=Jb / u, sigma_energy=sig_E, max_p=float(np.abs(sol.sub(0, deepcopy=True).vector()
                                                                         .get_local()).max())))
        print(f"D = {D:.5f} ({dname}): u = {u:.3f}: J = {Jb:+.6e}, J/u = {Jb / u:+.6e}, 2E/(A u^2) = {sig_E:+.6e}",
              flush=True)
    us = np.array([r["u"] for r in rows])
    Js = np.array([r["J"] for r in rows])
    # fit J = s u + g u^3 + g5 u^5
    nfit = min(3, len(us))
    A = np.vstack([us ** (2 * kk + 1) for kk in range(nfit)]).T
    coef = list(np.linalg.lstsq(A, Js, rcond=None)[0]) + [np.nan] * (3 - nfit)
    s_fit, g_fit, g5 = coef[:3]
    # small-slope cubic coefficient from the two smallest slopes (independent of the fit order)
    g_small = float((Js[1] / us[1] - Js[0] / us[0]) / (us[1] ** 2 - us[0] ** 2)) if len(us) >= 2 else None
    out["runs"].append(dict(drive_name=dname, D=D, rows=rows, sigma_eff_fit=float(s_fit), g_fit=float(g_fit),
                            g5_fit=float(g5), g_small_slope=g_small, failed_at_u=failed_at))
    json.dump(out, open(os.path.join(args.run, f"nonlinear_cell_L{args.Lc:g}" + (f"_a{args.angle:g}" if args.angle else "") + ".json"), "w"), indent=1)
    print(f"  fit: sigma_eff = {s_fit:+.5e}, g = {g_fit:+.5e}, g5 = {g5:+.3e}", flush=True)
json.dump(out, open(os.path.join(args.run, f"nonlinear_cell_L{args.Lc:g}" + (f"_a{args.angle:g}" if args.angle else "") + ".json"), "w"), indent=1)
