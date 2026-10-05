'''
Steady state and leading eigenmodes at one inflow velocity, stored on a P1 triangulation for figures and animations.

Stores the base state (in-plane velocity, tension, height) and the nev_store leading eigenmodes (real and imaginary
parts of all components that matter for plotting: z, w, v): a complex eigenvalue (flutter) is animated as
Re[(x_r + i x_i) exp(i omega t)].

run with:
STABILITY_PARAMS=zeta=1,omega_circle_const=0,pi_height=2,b_friction=0.04 python3 flow_snapshot.py square_b [mesh] [out] \
    --v0 0.21
'''
import irene_paths  # noqa: F401

import argparse
import json
import os

import numpy as np
from fenics import Constant, Function, FunctionSpace, assign, interpolate, project, vertex_to_dof_map

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp

parser = argparse.ArgumentParser()
parser.add_argument("--v0", type=float, required=True)
parser.add_argument("--n_substeps", type=int, default=8)
parser.add_argument("--nev", type=int, default=10)
parser.add_argument("--nev_store", type=int, default=4)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters

fsp.psi.vector().zero()
assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
for v in np.linspace(0, options.v0, options.n_substeps + 1)[1:]:
    fp.set_inflow(v)
    fp.newton()

P1 = FunctionSpace(fsp.Q.mesh(), "P", 1)
coords = P1.tabulate_dof_coordinates()
out = dict(x=coords[:, 0], y=coords[:, 1], triangles=vertex_to_dof_map(P1)[fsp.Q.mesh().cells()], v0=options.v0)


def fields(u, prefix):
    v, w, sigma, z, _, _ = u.split(deepcopy=True)
    out[prefix + "vx"] = project(v[0], P1).vector().get_local()
    out[prefix + "vy"] = project(v[1], P1).vector().get_local()
    out[prefix + "w"] = project(w, P1).vector().get_local()
    out[prefix + "sigma"] = project(sigma, P1).vector().get_local()
    out[prefix + "z"] = project(z, P1).vector().get_local()


fields(fsp.psi, "base_")
pairs = sorted(fp.eigenpairs(target=0.0, nev=options.nev), key=lambda p: -p.eigenvalue.real)[:options.nev_store]
eig = []
for k, p in enumerate(pairs):
    eig.append([p.eigenvalue.real, p.eigenvalue.imag, p.residual])
    for part, vec in (("re", p.x_real), ("im", p.x_imag)):
        u = Function(fsp.Q)
        u.vector()[:] = vec
        fields(u, f"mode{k}_{part}_")
out["eigenvalues"] = np.array(eig)
np.savez(os.path.join(out_dir, "snapshot.npz"), **out)
info = dict(v0=options.v0, eigenvalues=eig, b_friction=fp.B_FRICTION, pi_height=fp.PI_HEIGHT,
            mesh=rarg.args.input_directory)
json.dump(info, open(os.path.join(out_dir, "snapshot.json"), "w"), indent=1)
print("SNAPSHOT:", json.dumps(info), flush=True)
