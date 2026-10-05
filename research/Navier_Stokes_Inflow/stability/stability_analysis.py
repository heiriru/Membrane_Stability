'''
Linear stability analysis of the steady flow past a cylinder with the finite-element functions of the steady solver.

For each Reynolds number Re:
1. solve the steady Navier-Stokes equations F(w*; Re) = 0 with Newton's method (continuation in Re),
2. build the linearized operator A = -dF/dw(w*) and the mass matrix B from the FE forms (linear_stability.py),
3. compute the eigenvalues lambda of A x = lambda B x with the largest real part (perturbations ~ exp(lambda t)).
The steady state is unstable when max Re(lambda) > 0. The critical Reynolds number Re_c, at which the leading
eigenvalue crosses the imaginary axis, is then located with secant iterations on sigma(Re) = max Re(lambda).

run with:
python3 stability_analysis.py [problem] [mesh directory] [output directory] --Re 20,30,40,45,50,60,70,80
example:
MESH_PATH="/home/fenics/shared/generate_mesh/2d/cylinder/solution_unbounded_coarse"; SOLUTION_PATH="solution/coarse"; python3 stability_analysis.py cylinder $MESH_PATH $SOLUTION_PATH
'''
import argparse
import csv
import importlib
import json
import os
import time

import numpy as np
from fenics import XDMFFile, parameters

import runtime_arguments as rarg
import switch_problem as swi
import steady_state as ss
import linear_stability as ls
import plot_utils as pu

parser = argparse.ArgumentParser()
parser.add_argument("--Re", default="20,30,40,44,48,52,56,60,70,80",
                    help="comma separated list of Reynolds numbers (ascending)")
parser.add_argument("--nev", type=int, default=8, help="eigenvalues computed around each shift")
parser.add_argument("--spectrum_Re", default="40,50,60", help="Re at which the full spectrum is stored")
parser.add_argument("--tol_Re_c", type=float, default=1e-5, help="tolerance on sigma for the secant iterations")
options = parser.parse_args(rarg.unknown_args)

parameters["std_out_all_processes"] = False
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)

vp = importlib.import_module(swi.vp_steady)
rmsh = vp.rmsh
mesh = rmsh.lmsh.mesh
W = vp.W
print(f"problem = {rarg.args.problem}, number of degrees of freedom = {W.dim()}", flush=True)

Re_list = [float(x) for x in options.Re.split(",")]
spectrum_Re = [float(x) for x in options.spectrum_Re.split(",")]


def stability_at(Re, Re_previous, shifts):
    t0 = time.time()
    ss.solve_steady(vp, Re, Re_start=Re_previous)
    t1 = time.time()
    A, B = ls.assemble_eigenproblem(vp.F, vp.up, vp.bcs_hom, vp.mass)
    pairs = ls.leading_eigenvalues(A, B, shifts=shifts, nev=options.nev)
    t2 = time.time()
    force = ss.force_on_cylinder(vp, rmsh)
    u_star = vp.up.split(deepcopy=True)[0]
    L_r = ss.recirculation_length(u_star, rmsh)
    lead = pairs[0]
    print(f"Re = {Re:8.4f}: leading eigenvalue = {lead.eigenvalue.real:+.6f} {lead.eigenvalue.imag:+.6f} i "
          f"(residual {lead.residual:.1e}), C_D = {2 * force[0]:.4f}, L_r = {L_r:.4f}   "
          f"[steady {t1 - t0:.0f} s, eigenvalues {t2 - t1:.0f} s]", flush=True)
    return pairs, force, L_r, u_star


def save_fields(tag, u_star, pair=None):
    '''base flow and (optionally) eigenmode: XDMF files for ParaView and vertex values for matplotlib'''
    data = pu.mesh_arrays(mesh)
    data["u"] = pu.vertex_values(u_star, mesh)
    data["vorticity"] = pu.vertex_values(pu.vorticity(u_star, mesh), mesh)
    with XDMFFile(os.path.join(out_dir, f"base_flow_{tag}.xdmf")) as xdmf:
        xdmf.write(u_star)
    if pair is not None:
        real, imag = ls.eigenvector_to_functions(pair, W)
        for name, f in (("real", real), ("imag", imag)):
            u_mode = f.split(deepcopy=True)[0]
            scale = np.abs(u_mode.vector().get_local()).max()
            u_mode.vector()[:] = u_mode.vector().get_local() / scale if scale > 0 else 0.0
            u_mode.rename(f"mode_{name}", f"mode_{name}")
            data[f"mode_u_{name}"] = pu.vertex_values(u_mode, mesh)
            data[f"mode_vorticity_{name}"] = pu.vertex_values(pu.vorticity(u_mode, mesh), mesh)
            with XDMFFile(os.path.join(out_dir, f"eigenmode_{name}_{tag}.xdmf")) as xdmf:
                xdmf.write(u_mode)
        data["eigenvalue"] = pair.eigenvalue
    np.savez_compressed(os.path.join(out_dir, f"fields_{tag}.npz"), **data)


rows = []
spectra = {}
Re_previous = None
for Re in Re_list:
    pairs, force, L_r, u_star = stability_at(Re, Re_previous, ls.DEFAULT_SHIFTS)
    Re_previous = Re
    lead = pairs[0]
    rows.append(dict(Re=Re, sigma=lead.eigenvalue.real, omega=abs(lead.eigenvalue.imag), residual=lead.residual,
                     C_D=2 * force[0], C_L=2 * force[1], L_r=L_r))
    spectra[f"{Re:g}"] = np.array([p.eigenvalue for p in pairs])
    if Re in spectrum_Re:
        save_fields(f"Re{Re:g}", u_star, lead)

with open(os.path.join(out_dir, "stability_vs_Re.csv"), "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
np.savez(os.path.join(out_dir, "spectra.npz"), **spectra)

# critical Reynolds number: secant iterations on sigma(Re) between the last stable and the first unstable Re
sigma = np.array([row["sigma"] for row in rows])
Re_arr = np.array([row["Re"] for row in rows])
crossing = np.where((sigma[:-1] < 0) & (sigma[1:] > 0))[0]
result = dict(problem=rarg.args.problem, mesh=rarg.args.input_directory, dofs=W.dim(), cells=mesh.num_cells())
if len(crossing) == 0:
    print("sigma does not change sign in the given range of Re: no critical Reynolds number found")
else:
    i = crossing[0]
    (Re_a, s_a), (Re_b, s_b) = (Re_arr[i], sigma[i]), (Re_arr[i + 1], sigma[i + 1])
    lam_guess = complex(rows[i + 1]["sigma"], rows[i + 1]["omega"])
    iterations = []
    Re_current = Re_arr[-1]
    for it in range(12):
        Re_new = Re_b - s_b * (Re_b - Re_a) / (s_b - s_a)
        # track the leading eigenvalue with a single shift placed next to it
        pairs, force, L_r, u_star = stability_at(Re_new, Re_current, (complex(0.05, lam_guess.imag),))
        Re_current = Re_new
        lead = pairs[0]
        lam_guess = lead.eigenvalue
        iterations.append(dict(Re=Re_new, sigma=lead.eigenvalue.real, omega=lead.eigenvalue.imag))
        (Re_a, s_a), (Re_b, s_b) = (Re_b, s_b), (Re_new, lead.eigenvalue.real)
        if abs(lead.eigenvalue.real) < options.tol_Re_c:
            break
    save_fields("Re_c", u_star, lead)
    result.update(Re_c=Re_new, omega_c=abs(lead.eigenvalue.imag), St_c=abs(lead.eigenvalue.imag) / (2 * np.pi),
                  sigma_at_Re_c=lead.eigenvalue.real, secant_iterations=iterations)
    print(f"\nCRITICAL REYNOLDS NUMBER: Re_c = {Re_new:.4f}, omega_c = {abs(lead.eigenvalue.imag):.5f}, "
          f"St_c = omega_c / (2 pi) = {abs(lead.eigenvalue.imag) / (2 * np.pi):.5f}", flush=True)

with open(os.path.join(out_dir, "critical_Re.json"), "w") as f:
    json.dump(result, f, indent=2)
print("... done.", flush=True)
