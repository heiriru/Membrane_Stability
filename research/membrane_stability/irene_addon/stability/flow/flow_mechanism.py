'''
Fields that explain a flow-driven buckling instability: for the steady state at inflow v0, the surface tension sigma,
the in-plane stress of the membrane T_ij = sigma g_ij + 2 eta d_ij (tension positive), its smallest principal value
(negative = compression along that direction), the velocity, the shape z*, and the leading eigenmode z'.
Saved as P1 nodal values (npz) for plotting.

run with:
STABILITY_PARAMS=zeta=1,omega_circle_const=0 python3 flow_mechanism.py square_b [mesh] [out] --v0 0.165
'''
import irene_paths  # noqa: F401

import argparse
import os

import numpy as np
from fenics import Constant, assign, interpolate, project, FunctionSpace, sqrt

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--v0", type=float, required=True)
parser.add_argument("--n_steps", type=int, default=10)
parser.add_argument("--tag", default="")
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters

fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
for v in np.linspace(0, options.v0, options.n_steps + 1):
    fp.set_inflow(v)
    fp.newton()

pairs = fp.eigenpairs(target=0.0, nev=8)
lead = max(pairs, key=lambda p: p.eigenvalue.real)
print(f"v0 = {options.v0}: leading eigenvalue {lead.eigenvalue}", flush=True)
mode = ls.eigenvector_to_functions(lead, fsp.Q)[0]

v, w, sigma, z, omega, mu = fsp.psi.split(deepcopy=True)
P1 = FunctionSpace(fsp.Q.mesh(), "P", 1)
# flat-membrane in-plane stress (the base states used here are flat or weakly curved)
eta = prm["eta"]
d11 = v[0].dx(0)
d22 = v[1].dx(1)
d12 = 0.5 * (v[0].dx(1) + v[1].dx(0))
T11 = sigma + 2 * eta * d11
T22 = sigma + 2 * eta * d22
T12 = 2 * eta * d12
T_min = 0.5 * (T11 + T22) - sqrt(0.25 * (T11 - T22) ** 2 + T12 ** 2)
fields = dict(
    sigma=project(sigma, P1), T_min=project(T_min, P1), T11=project(T11, P1), T22=project(T22, P1),
    vx=project(v[0], P1), vy=project(v[1], P1), z=project(z, P1), mode=project(mode.sub(3), P1))
coords = P1.tabulate_dof_coordinates()
cells = fsp.Q.mesh().cells()
# map P1 dofs to vertices for a triangulation
from fenics import dof_to_vertex_map, vertex_to_dof_map
v2d = vertex_to_dof_map(P1)
np.savez(os.path.join(out_dir, f"mechanism{options.tag}.npz"), x=coords[:, 0], y=coords[:, 1],
         triangles=v2d[cells], eigenvalue=lead.eigenvalue, v0=options.v0, eta=eta, sigma_r=prm["sigma_r_const"],
         L=fp.rmsh.parameters["L"], r=fp.rmsh.parameters["r"],
         **{k: f.vector().get_local() for k, f in fields.items()})
s = fields["sigma"].vector().get_local()
t = fields["T_min"].vector().get_local()
print(f"sigma: min {s.min():.4e} max {s.max():.4e}; smallest principal stress T_min: min {t.min():.4e}; "
      f"area fraction with sigma < 0: {(s < 0).mean():.3f}", flush=True)
