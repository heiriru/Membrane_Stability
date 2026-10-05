'''
Consistency check of the vertical force on the protein inclusion (PI).

For a PI translated rigidly by dh at fixed contact angle, virtual work gives the vertical force exerted by the
membrane on the PI as  F_virtual = -dE/dh  (E = Helfrich energy of the steady state). This script compares
    * F_virtual from central finite differences of E(h),
    * F_line: IRENE's line-force integral (dFdl_sigma_kappa_3d on the PI circle, = Eqs. (17)-(18) of the PRE),
    * F_reaction: the reaction of the Dirichlet constraint z = h, i.e. the residual of IRENE's weak form tested with a
      (removed)

run with:
python3 check_force_energy.py ring [mesh directory] [output directory] --tan_alpha 0.5 --h 0,0.5,1,-0.5,-1
'''
import irene_paths  # noqa: F401

import argparse
import os

import numpy as np
from fenics import assemble, sqrt
import differential_geometry.boundary.geometry as bgeo

import runtime_arguments as rarg
import ring_problem as rp
import function_spaces as fsp

parser = argparse.ArgumentParser()
parser.add_argument("--tan_alpha", type=float, default=0.5)
parser.add_argument("--h", default="0,0.5,1,-0.5,-1")
parser.add_argument("--eps", type=float, default=0.01)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)



def moment_term():
    '''boundary term 2 kappa H n^i d_i psi of the energy variation for a rigid vertical translation (psi = dh N_3)'''
    _, omega, mu = rp.fields()
    N3 = 1.0 / sqrt(1.0 + omega[0] ** 2 + omega[1] ** 2)
    n = bgeo.n_circle(omega)
    line_element = bgeo.sqrt_deth_circle(omega, rp.c_r) * (1.0 / rp.r0)
    return assemble(2.0 * rp.KAPPA * mu * (n[0] * N3.dx(0) + n[1] * N3.dx(1)) * line_element * rp.rmsh.ds_r)


rp.set_parameters(h=0.0, slope_r=options.tan_alpha, slope_R=0.0)
rp.flat_initial_guess()
rp.solve_steady(0.0, h_previous=0.0)
h_current = 0.0
lines = ["h,F_virtual,F_line_z,moment_term"]
print("      h      F_virtual=-dE/dh   F_line (IRENE, PRE Eq.17-18)   moment term   F_line - M   F_line + M")
for h in sorted([float(x) for x in options.h.split(",")], key=abs):
    rp.solve_steady(h, h_previous=h_current, n_substeps=4)
    h_current = h
    state = fsp.psi.vector().copy()
    F_line = rp.force_on_protein()[2]
    M = moment_term()
    energies = []
    for dh in (-options.eps, options.eps):
        rp.solve_steady(h + dh, h_previous=h)
        energies.append(rp.energy())
        fsp.psi.vector()[:] = state
        rp.set_parameters(h=h)
    F_virtual = -(energies[1] - energies[0]) / (2 * options.eps)
    print(f"  {h:+6.2f}   {F_virtual:+.6e}      {F_line:+.6e}                {M:+.4e}  {F_line - M:+.4e}  {F_line + M:+.4e}", flush=True)
    lines.append(f"{h},{F_virtual},{F_line},{M}")
open(os.path.join(out_dir, "check_force_energy.csv"), "w").write("\n".join(lines) + "\n")
