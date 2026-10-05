'''
Least stable eigenvalues of a steady state of any IRENE no-flow problem (z, omega, mu formulation), with the
overdamped normal dynamics of ring_problem.py: mass (zeta/2) <z, nu_z>, residual -F, homogeneous Dirichlet BCs obtained
by homogenizing IRENE's BCs. Parameters are read by IRENE from parameters_bc_<problem>.csv in this folder.

run with:
python3 static_spectrum.py square_a [mesh directory] [output directory] --nev 12
'''
import irene_paths  # noqa: F401

import argparse
import importlib
import os

import dolfin
import numpy as np
from fenics import DirichletBC, derivative, NonlinearVariationalProblem, NonlinearVariationalSolver, split

dolfin.parameters["form_compiler"]["quadrature_degree"] = 4

import runtime_arguments as rarg
import switch_problem as swi
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--nev", type=int, default=12)
parser.add_argument("--target", type=float, default=0.0)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)

vp = importlib.import_module(swi.vp)
rmsh = vp.rmsh

solver = NonlinearVariationalSolver(NonlinearVariationalProblem(vp.F, fsp.psi, vp.bcs, derivative(vp.F, fsp.psi, fsp.J_psi)))
solver.parameters["newton_solver"]["linear_solver"] = "mumps"
solver.parameters["newton_solver"]["absolute_tolerance"] = 1e-10
solver.solve()

bcs_hom = []
for bc in vp.bcs:
    bc_h = DirichletBC(bc)
    bc_h.homogenize()
    bcs_hom.append(bc_h)


def mass(trial, test):
    return 0.5 * split(trial)[0] * split(test)[0] * rmsh.dx


A, B = ls.assemble_eigenproblem(-vp.F, fsp.psi, bcs_hom, mass)
pairs = ls.solve_eigenproblem(A, B, target=options.target, nev=options.nev)
values = np.array([p.eigenvalue.real for p in pairs])
print("static eigenvalues (zeta = 1):", " ".join(f"{v:.6e}" for v in values))
np.savetxt(os.path.join(out_dir, "static_spectrum.csv"), values, header="lambda_static", comments="")
