'''
Check of the vertical force from the conserved stress flux (modules/stability/vertical_force.py) against the
virtual-work force F = -dE/dh in IRENE's ring problem (exact for this variational problem).

run with:
python3 check_vertical_force.py ring [mesh] [out] --tan_alpha 0.5 --heights -6,-3,0,2,4
'''
import irene_paths  # noqa: F401

import argparse
import json
import os

import numpy as np
from fenics import assemble, sqrt

import runtime_arguments as rarg
import ring_problem as rp
import function_spaces as fsp
from stability import vertical_force as vf

parser = argparse.ArgumentParser()
parser.add_argument("--tan_alpha", type=float, default=0.5)
parser.add_argument("--heights", default="-6,-3,0,2,4")
parser.add_argument("--eps", type=float, default=0.01)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)

rp.set_parameters(h=0.0, slope_r=options.tan_alpha, slope_R=0.0)
rp.flat_initial_guess()
rp.solve_steady(0.0)
h_prev = 0.0
rows = []
for h in [float(x) for x in options.heights.split(",")]:
    rp.solve_steady(h, h_previous=h_prev, n_substeps=max(1, int(abs(h - h_prev) / 0.25)))
    h_prev = h
    state = fsp.psi.vector().copy()
    E = []
    for dh in (-options.eps, options.eps):
        rp.solve_steady(h + dh, h_previous=h)
        E.append(rp.energy())
        fsp.psi.vector()[:] = state
        rp.set_parameters(h=h)
    F_virtual = -(E[1] - E[0]) / (2 * options.eps)
    z, omega, mu = rp.fields()
    # check of the sign convention of H: mu vs 1/2 g^ij z_ij / sqrt(g)
    g = 1 + omega[0] ** 2 + omega[1] ** 2
    H_om = 0.5 * ((1 + omega[1] ** 2) * omega[0].dx(0) + (1 + omega[0] ** 2) * omega[1].dx(1)
                  - omega[0] * omega[1] * (omega[0].dx(1) + omega[1].dx(0))) / g ** 1.5
    H_check = assemble(mu * H_om * rp.rmsh.dx) / assemble(mu * mu * rp.rmsh.dx)
    F_dom = vf.vertical_force(omega, mu, fsp.sigma, rp.KAPPA, rp.rmsh.dx, rp.c_r)
    rows.append(dict(h=h, F_virtual=F_virtual, F_domain=F_dom, H_check=H_check))
    print(f"h = {h:+.2f}: F_virtual = {F_virtual:+.6f}, F_domain = {np.round(F_dom, 6)}, <mu H(omega)>/<mu^2> = "
          f"{H_check:.4f}", flush=True)
json.dump(rows, open(os.path.join(out_dir, "check_vertical_force.json"), "w"), indent=1)
