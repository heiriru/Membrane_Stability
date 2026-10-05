'''
Vertical force on the PI in IRENE's square_b setup (the "free height" PI of Ferraro & Castellana, PRE).

IRENE's normal force balance F_w contains, on the PI circle, the boundary term 2 kappa n.grad(mu) nu_w of the
integration by parts of the bending term, and the tension and viscous normal-force terms are not integrated by parts.
The rows of the rim degrees of freedom are then not a force balance: nothing in the discrete problem imposes that the net
vertical force on the PI vanishes, although z is free on the rim. This script shows it on a critical mode of
flow_critical.py (flat base state, omega_circle = 0), and it decomposes IRENE's normal-force work on the mode
    <F_w(z'), z'> = (bending) + (tension) + (viscous) + (rim term)
and compares it with the energy of the compressed-plate model (flow_plate_model.py):
the rim term equals (net vertical force on the PI, int 2 kappa dmu'/dn) x (PI displacement): a spurious force doing
work on the PI, which lowers the threshold.

run with:
STABILITY_PARAMS=zeta=1,omega_circle_const=0 python3 check_pi_force.py square_b [mesh] [out] --mode_dir [critical dir]
'''
import irene_paths  # noqa: F401

import argparse
import json
import os

import numpy as np
from fenics import FacetNormal, Function, as_matrix, assemble, dot, grad, inner, sym

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp

parser = argparse.ArgumentParser()
parser.add_argument("--mode_dir", required=True)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
kappa, eta, sigma0 = prm["kappa"], prm["eta"], prm["sigma_r_const"]

d = np.load(os.path.join(options.mode_dir, "critical_mode.npz"))
fsp.psi.vector()[:] = d["base"]
mode = Function(fsp.Q)
mode.vector()[:] = d["x_real"]
_, _, _, z, om, mu = mode.split(deepcopy=True)
vb, _, sb, _, _, _ = fsp.psi.split(deepcopy=True)
n = FacetNormal(fsp.Q.mesh())
dx, ds_c = fp.rmsh.dx, fp.rmsh.ds_circle
G = sym(grad(vb))

rim_length = assemble(1.0 * ds_c(domain=fsp.Q.mesh()))
res = dict(
    v0=float(d["v0"]), eigenvalue=float(np.real(d["eigenvalue"])),
    net_bending_shear_on_PI=assemble(2 * kappa * dot(grad(mu), n) * ds_c),
    abs_bending_shear_on_PI=assemble(abs(2 * kappa * dot(grad(mu), n)) * ds_c),
    PI_mean_displacement=assemble(z * ds_c) / rim_length,
    rim_slope_omega=assemble(abs(dot(om, om)) ** 0.5 * ds_c) / rim_length,
    irene_bending=assemble(-2 * kappa * dot(grad(mu), grad(z)) * dx),
    irene_rim_term=assemble(2 * kappa * dot(grad(mu), n) * z * ds_c),
    irene_tension=assemble(-2 * sb * mu * z * dx),
    irene_viscous=assemble(-2 * eta * inner(G, sym(grad(om))) * z * dx),
    plate_bending=assemble(kappa * (2 * mu) ** 2 * dx),
    plate_tension=assemble(sigma0 * dot(grad(z), grad(z)) * dx),
    plate_flow=assemble(dot(dot((sb - sigma0) * as_matrix([[1, 0], [0, 1]]) + 2 * eta * G, grad(z)), grad(z)) * dx),
)
res["rim_term_over_force_times_displacement"] = res["irene_rim_term"] / (
    res["net_bending_shear_on_PI"] * res["PI_mean_displacement"])
res["plate_energy_total"] = res["plate_bending"] + res["plate_tension"] + res["plate_flow"]
for k, v in res.items():
    print(f"{k:40s} {v:+.5e}")
json.dump(res, open(os.path.join(out_dir, "check_pi_force.json"), "w"), indent=1)
