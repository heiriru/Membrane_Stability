'''
Compressed-plate model of the flow-driven buckling (PI with zero contact angle, rho = 0).

For omega_circle = 0 the steady state is flat (z* = 0, w* = 0) for every inflow v0, and the in-plane problem is the 2D
Stokes problem with the incompressibility constraint: the in-plane stress
    T_ij = sigma g_ij + 2 eta d_ij          (tension positive)
is linear in v0,  T = sigma0 I + v0 T1  (T1 does not depend on sigma0: the outflow tension sigma_r = sigma0 only shifts
sigma by a constant). Around a flat state the normal and in-plane perturbations decouple at linear order, and the normal
force balance of a perturbation z' is that of a plate under the in-plane stress T:
    zeta dz'/dt = -kappa Delta^2 z' + div(T grad z')
(clamped outer boundary z' = dz'/dn = 0, dz'/dn = 0 and zero shear force on the PI, whose height is free). This operator
is self-adjoint, so the threshold is the first v0 at which the energy
    E2[z] = kappa int (Delta z)^2 + sigma0 int |grad z|^2 + v0 int grad z . T1 grad z
stops being positive definite:  v0_c = min over z of  (kappa int (Delta z)^2 + sigma0 int |grad z|^2) / (- int grad z . T1 grad z),
a generalized eigenvalue problem for v0 (mixed Ciarlet-Raviart formulation, m = Delta z, P2 x P2).

The script
    1. computes the flat steady state at v0_ref with IRENE's full equations and extracts T1 (and its tension part
       sigma1 I and viscous part 2 eta d1),
    2. checks the model: relaxation rates of the plate at v0_ref vs the full FE eigenvalues (same zeta),
    3. solves the eigenproblem for v0_c at each sigma0 (--gammas: Gamma = sigma0 L^2 / kappa) and decomposes the Rayleigh
       quotient at the critical mode into bending, tension, and the tension / viscous parts of the flow-induced stress.

run with (same parameters and variants as flow_critical.py):
STABILITY_PARAMS=zeta=1,omega_circle_const=0 python3 flow_plate_model.py square_b [mesh] [out] --gammas 0,10,25,50,100
'''
import irene_paths  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
from fenics import (Constant, DirichletBC, FiniteElement, Function, FunctionSpace, MixedElement, PETScMatrix,
                    TestFunctions, TrialFunctions, as_matrix, assemble, assign, dot, grad, inner, interpolate,
                    project, split, sym)

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls
from stability import modes

parser = argparse.ArgumentParser()
parser.add_argument("--v0_ref", type=float, default=0.1)
parser.add_argument("--gammas", default=None, help="Gamma = sigma0 L^2 / kappa values (default: the one of the run)")
parser.add_argument("--nev", type=int, default=6)
parser.add_argument("--tag", default="")
parser.add_argument("--pi_bc", choices=["rigid", "clamped", "free", "irene", "no_flux"], default="rigid",
                    help="rigid (default): rigid PI of free height, z' = h' on the rim (one unknown), zero net vertical "
                         "force; clamped: z' = 0 on the rim (height fixed); free: zero vertical shear force on the PI (natural condition of the free-height PI); irene: "
                         "the PI rows of the normal force balance keep the boundary terms of the integration by parts, "
                         "as IRENE's F_w does on the circle (no natural condition); no_flux: bending shear force "
                         "natural, but without the vertical component n.T.grad z of the in-plane traction on the rim")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
L, W, r0 = fp.rmsh.parameters["L"], fp.rmsh.parameters["h"], fp.rmsh.parameters["r"]
c_r = fp.rmsh.parameters["c_r"]
kappa, eta, sigma0, zeta = prm["kappa"], prm["eta"], prm["sigma_r_const"], prm["zeta"]
if prm["omega_circle_const"] != 0:
    raise ValueError("the plate model needs a flat base state: run with omega_circle_const=0")

# 1. flat base state at v0_ref (full equations), and the full FE eigenvalues there
t0 = time.time()
fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(sigma0), fsp.Q_sigma))
for v in np.linspace(0, options.v0_ref, 5)[1:]:
    fp.set_inflow(v)
    fp.newton()
z_max = np.abs(fsp.psi.sub(3, deepcopy=True).vector().get_local()).max()
A, B = fp.eigenproblem()
full_pairs = ls.solve_eigenproblem(A, B, target=0.0, nev=options.nev)
full_rates = sorted([p.eigenvalue.real for p in full_pairs if abs(p.eigenvalue.imag) < 1e-10], reverse=True)
print(f"flat base state at v0 = {options.v0_ref} (max|z*| = {z_max:.1e}), full FE rates: {full_rates[:4]}  "
      f"[{time.time() - t0:.0f} s]", flush=True)

v_base, _, sigma_base, _, _, _ = fsp.psi.split(deepcopy=True)
d_base = sym(grad(v_base))
I2 = as_matrix([[1.0, 0.0], [0.0, 1.0]])
T1_sigma = (sigma_base - sigma0) / options.v0_ref * I2
T1_visc = 2.0 * eta * d_base / options.v0_ref
T1 = T1_sigma + T1_visc

# 2. plate model: mixed space (z, m = Delta z)
mesh = fsp.Q.mesh()
P2 = FiniteElement("P", mesh.ufl_cell(), 2)
V = FunctionSpace(mesh, MixedElement([P2, P2]))
z, m = TrialFunctions(V)
phi, q = TestFunctions(V)
dx = fp.rmsh.dx
fixed = {0: fp.rmsh.boundary_square, 1: f"near(x[0], 0.0) || near(x[0], {L})", 2: "near(x[0], 0.0)"}[fp.Z_OUTER]
bc = DirichletBC(V.sub(0), Constant(0.0), fixed)

a_m = (m * q + dot(grad(z), grad(q))) * dx                      # m = Delta z (dz/dn = 0 natural)
a_bend = -kappa * dot(grad(m), grad(phi)) * dx                   # = int kappa Delta^2 z phi
a_tens = dot(grad(z), grad(phi)) * dx
a_flow = dot(dot(T1, grad(z)), grad(phi)) * dx
if fp.B_FRICTION:
    # the normal force is T : grad grad z = div(T grad z) - (div T) . grad z; without friction div T = 0, with the
    # in-plane friction div T = b (v - v0 e_x) = v0 b (v1 - e_x): a non-conservative (follower-type) term, which makes
    # the plate operator non-self-adjoint (the energy decomposition below is then only indicative)
    from fenics import as_vector
    a_flow = a_flow + fp.B_FRICTION * dot(v_base / options.v0_ref - as_vector((1.0, 0.0)), grad(z)) * phi * dx
if options.pi_bc in ("irene", "no_flux"):
    from fenics import FacetNormal
    n = FacetNormal(mesh)
    ds_c = fp.rmsh.ds_circle
    if options.pi_bc == "irene":
        a_bend = a_bend + kappa * dot(grad(m), n) * phi * ds_c
    a_tens = a_tens - dot(grad(z), n) * phi * ds_c
    a_flow = a_flow - dot(dot(T1, grad(z)), n) * phi * ds_c


bcs_plate = [bc]
if options.pi_bc == "clamped":
    bcs_plate.append(DirichletBC(V.sub(0), Constant(0.0), fp.rmsh.boundary_circle))
rim_dofs = None
if options.pi_bc == "rigid":
    rim_dofs = np.array(sorted(DirichletBC(V.sub(0), Constant(0.0), fp.rmsh.boundary_circle).get_boundary_values()))


def assemble_bc(form, identity):
    M = PETScMatrix()
    assemble(form, tensor=M, keep_diagonal=True)
    for b in bcs_plate:
        if identity:
            b.apply(M)
        else:
            b.zero(M)
    return M


def solve_plate(Kf, Mf, identity_K=True):
    '''eigenpairs of K x = lambda M x (with the rim tied for the rigid PI); eigenvectors returned in the full space'''
    K, M = assemble_bc(Kf, identity_K), assemble_bc(Mf, False)
    if rim_dofs is None:
        return ls.solve_eigenproblem(K, M, target=0.0, nev=options.nev)
    K_r, M_r, P = ls.tie_dofs(K, M, [rim_dofs])
    pairs = ls.solve_eigenproblem(K_r, M_r, target=0.0, nev=options.nev)
    for p in pairs:
        p.x_real, p.x_imag = P @ p.x_real, P @ p.x_imag
    return pairs


def rates(v0, s0):
    '''relaxation rates of the plate at inflow v0 and tension s0: zeta lambda z = -(kappa Delta^2 z - div(T grad z))'''
    pairs = solve_plate(a_m - (a_bend + s0 * a_tens + v0 * a_flow), zeta * z * phi * dx)
    return sorted([p.eigenvalue.real for p in pairs], reverse=True), pairs


plate_rates, _ = rates(options.v0_ref, sigma0)
print(f"plate model rates at v0 = {options.v0_ref}: {plate_rates[:4]}", flush=True)
rel = abs(plate_rates[0] - full_rates[0]) / abs(full_rates[0])
print(f"leading rate: full {full_rates[0]:+.6e}, plate {plate_rates[0]:+.6e}, relative difference {rel:.2e}", flush=True)
rest_rates, _ = rates(0.0, sigma0)

# 3. v0_c(sigma0): (a_m + bending/tension part) x = v0 (- flow part) x
gammas = [float(g) for g in options.gammas.split(",")] if options.gammas else [sigma0 * L ** 2 / kappa]
results = []
for gamma in gammas:
    s0 = gamma * kappa / L ** 2
    pairs = solve_plate(a_m + a_bend + s0 * a_tens, -a_flow)
    positive = [p for p in pairs if p.eigenvalue.real > 0 and abs(p.eigenvalue.imag) < 1e-8 * abs(p.eigenvalue)]
    if not positive:
        print(f"Gamma = {gamma}: no positive threshold found", flush=True)
        continue
    crit = min(positive, key=lambda p: p.eigenvalue.real)
    v0_c = crit.eigenvalue.real
    mode = Function(V)
    mode.vector()[:] = crit.x_real
    z_c, m_c = mode.split(deepcopy=True)
    parts = dict(bending=assemble(kappa * m_c ** 2 * dx), tension=assemble(s0 * dot(grad(z_c), grad(z_c)) * dx),
                 flow_sigma=assemble(v0_c * dot(dot(T1_sigma, grad(z_c)), grad(z_c)) * dx),
                 flow_viscous=assemble(v0_c * dot(dot(T1_visc, grad(z_c)), grad(z_c)) * dx))
    balance = sum(parts.values()) / parts["bending"]
    # the same Rayleigh quotient with the mode frozen: linear dependence on Gamma
    z_c.set_allow_extrapolation(True)
    radii = [r for r in (2 * r0, 5 * r0, 10 * r0, 20 * r0) if r < 0.45 * min(L, W)]
    m_dom, spectrum = modes.dominant_m(z_c, c_r[:2], radii)
    row = dict(Gamma=gamma, sigma0=s0, v0_c=v0_c, SL_c=eta * v0_c * L / kappa, residual=crit.residual,
               energy_parts=parts, energy_balance=balance, azimuthal_spectrum=list(spectrum / spectrum.max()),
               dominant_m=m_dom)
    results.append(row)
    print(f"Gamma = {gamma:g}: v0_c = {v0_c:.6f}, SL_c = {row['SL_c']:.4f} (residual {crit.residual:.1e}); energy: "
          + ", ".join(f"{k} {v:+.3e}" for k, v in parts.items()) + f"; sum/bending = {balance:.1e}; "
          f"azimuthal |z_m|/max (m=0..4): {np.round(row['azimuthal_spectrum'][:5], 3)}", flush=True)
    coords_z = z_c.function_space().tabulate_dof_coordinates()
    np.savez(os.path.join(out_dir, f"plate_mode{options.tag}_G{gamma:g}.npz"), x=coords_z[:, 0], y=coords_z[:, 1],
             z=z_c.vector().get_local(), v0_c=v0_c)

# stress fields for the figures (P1 at the vertices)
P1 = FunctionSpace(mesh, "P", 1)
from fenics import vertex_to_dof_map
fields = {"sigma1": project((sigma_base - sigma0) / options.v0_ref, P1)}
for (i, j) in ((0, 0), (1, 1), (0, 1)):
    fields[f"T1_{i}{j}"] = project(T1[i, j], P1)
    fields[f"T1visc_{i}{j}"] = project(T1_visc[i, j], P1)
c1 = P1.tabulate_dof_coordinates()
np.savez(os.path.join(out_dir, f"plate_T1{options.tag}.npz"), x=c1[:, 0], y=c1[:, 1],
         triangles=vertex_to_dof_map(P1)[mesh.cells()], **{k: f.vector().get_local() for k, f in fields.items()})

summary = dict(pi_bc=options.pi_bc, L=L, W=W, r=r0, kappa=kappa, eta=eta, sigma0=sigma0, zeta=zeta, b_friction=fp.B_FRICTION,
               ell_b=float(np.sqrt(eta / fp.B_FRICTION)) if fp.B_FRICTION else None, z_outer=fp.Z_OUTER,
               v0_ref=options.v0_ref, mesh=rarg.args.input_directory, dofs_full=fsp.Q.dim(), dofs_plate=V.dim(),
               full_rates=full_rates[:4], plate_rates=plate_rates[:4], rest_rates=rest_rates[:4],
               check_leading_rate_rel_diff=rel, thresholds=results)
json.dump(summary, open(os.path.join(out_dir, f"plate_model{options.tag}.json"), "w"), indent=1)
print("PLATE:", json.dumps({k: summary[k] for k in ("L", "W", "r", "sigma0", "ell_b", "z_outer")}),
      json.dumps([(r["Gamma"], r["SL_c"]) for r in results]), flush=True)
