'''
Check of the steady Navier-Stokes solver (the base flow of the stability analysis) against the Schaefer-Turek (DFG)
benchmark 2D-1, Re = 20, in the channel with a cylinder (problem 'channel', lengths in units of D, U_mean = 1).
Reference values (Schaefer & Turek 1996; John 2004): C_D = 5.57953523384, C_L = 0.010618948146,
pressure difference between front and back of the cylinder Delta p = 0.11752016697 / U_mean^2 = 2.93800417.

run with:
python3 check_dfg_benchmark.py channel /home/fenics/shared/generate_mesh/2d/cylinder/solution_dfg solution/checks
'''
import importlib

import runtime_arguments as rarg
import switch_problem as swi
import steady_state as ss

vp = importlib.import_module(swi.vp_steady)
rmsh = vp.rmsh
ss.solve_steady(vp, 20.0)
force = ss.force_on_cylinder(vp, rmsh)
_, p_star = vp.up.split(deepcopy=True)
p_star.set_allow_extrapolation(True)
c = rmsh.c_r
values = dict(C_D=2 * force[0], C_L=2 * force[1], Delta_p=p_star(c[0] - rmsh.r, c[1]) - p_star(c[0] + rmsh.r, c[1]))
reference = dict(C_D=5.57953523384, C_L=0.010618948146, Delta_p=0.11752016697 / 0.2 ** 2)
tolerance = dict(C_D=5e-3, C_L=5e-2, Delta_p=5e-3)
passed = True
print(f"DFG benchmark 2D-1 (Re = 20), {vp.W.dim()} dofs:")
for key in values:
    rel = abs(values[key] - reference[key]) / abs(reference[key])
    passed &= rel < tolerance[key]
    print(f"   {key:8s} = {values[key]:.6f}   reference {reference[key]:.6f}   relative error {rel:.1e}")
print("CHECK DFG BENCHMARK:", "PASSED" if passed else "FAILED")
