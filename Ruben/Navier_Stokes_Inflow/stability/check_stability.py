'''
Stable or unstable? One steady solve + one eigenvalue solve at a single Reynolds number, no time integration.

1. Newton: steady state w* with F(w*) = 0
2. eigenvalues of A x = lambda B x, A = -dF/dw(w*), B = mass matrix
3. verdict: stable if max Re(lambda) < 0, unstable otherwise; the eigenvalue also gives the growth rate
   sigma = Re(lambda) and the frequency omega = Im(lambda) of the perturbation that will appear.

run with:
python3 check_stability.py cylinder /home/fenics/shared/generate_mesh/2d/cylinder/solution_unbounded_coarse solution/verdict --Re 60
'''
import argparse
import importlib
import time

import runtime_arguments as rarg
import switch_problem as swi
import steady_state as ss
import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--Re", type=float, required=True)
options = parser.parse_args(rarg.unknown_args)

vp = importlib.import_module(swi.vp_steady)
t0 = time.time()
ss.solve_steady(vp, options.Re)
t1 = time.time()
A, B = ls.assemble_eigenproblem(vp.F, vp.up, vp.bcs_hom, vp.mass)
lead = ls.leading_eigenvalues(A, B)[0]
t2 = time.time()

lam = lead.eigenvalue
verdict = "UNSTABLE" if lam.real > 0 else "STABLE"
print(f"\nRe = {options.Re:g}: leading eigenvalue lambda = {lam.real:+.5f} {lam.imag:+.5f} i  ->  {verdict}")
if lam.real > 0:
    print(f"   perturbations grow like exp({lam.real:.4f} t): amplitude x10 every {2.302585 / lam.real:.1f} time units, "
          f"oscillating with Strouhal number {abs(lam.imag) / 6.283185:.4f}")
else:
    print(f"   perturbations decay like exp({lam.real:.4f} t): amplitude /10 every {2.302585 / -lam.real:.1f} time units")
print(f"   time: steady state {t1 - t0:.0f} s + eigenvalues {t2 - t1:.0f} s, no time integration")
