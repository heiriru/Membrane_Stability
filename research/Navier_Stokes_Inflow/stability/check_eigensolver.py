'''
Check of the eigenvalue machinery of linear_stability.py on problems with known eigenvalues, set up exactly like the
Navier-Stokes problem (a residual form F, a mass form, homogeneous Dirichlet BCs).

1. Heat equation on the unit square, u_t = Laplace(u), u = 0 on the boundary:
       lambda_{m,n} = -pi^2 (m^2 + n^2)                                        (real, symmetric problem)
2. Two coupled fields with a rotation, u1_t = Laplace(u1) - c u2, u2_t = Laplace(u2) + c u1:
       lambda_{m,n} = -pi^2 (m^2 + n^2) +/- i c                                (complex pairs, non-symmetric)
   This tests the complex shift-and-invert (the one used for the cylinder) and the non-symmetric SLEPc solver.

run with:
python3 check_eigensolver.py [output directory]
'''
import os
import sys

import numpy as np
from fenics import (UnitSquareMesh, FunctionSpace, FiniteElement, MixedElement, Function, TestFunction, TestFunctions,
                    DirichletBC, Constant, inner, grad, split, dx)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import linear_stability as ls

out_dir = sys.argv[1] if len(sys.argv) > 1 else "solution/checks"
os.makedirs(out_dir, exist_ok=True)

mesh = UnitSquareMesh(40, 40)
exact_real = np.sort(np.array([-np.pi ** 2 * (m ** 2 + n ** 2) for m in range(1, 6) for n in range(1, 6)]))[::-1]
passed = True

# 1) heat equation
V = FunctionSpace(mesh, "P", 2)
w = Function(V)
phi = TestFunction(V)
F = inner(grad(w), grad(phi)) * dx
A, B = ls.assemble_eigenproblem(F, w, [DirichletBC(V, Constant(0.0), "on_boundary")], lambda tr, te: tr * te * dx)
pairs_slepc = ls.solve_eigenproblem(A, B, target=0.0, nev=6)
pairs_shift = ls.solve_eigenproblem_complex_shift(A, B, shift=-40.0 + 0.0j, nev=6)
lam_slepc = np.array([p.eigenvalue for p in pairs_slepc])[:6]
lam_shift = np.array([p.eigenvalue for p in pairs_shift])[:6]
print("1) heat equation: computed (SLEPc, complex shift) vs exact")
for k in range(6):
    print(f"   {lam_slepc[k].real:12.6f}  {lam_shift[k].real:12.6f}   exact {exact_real[k]:12.6f}")
error_1 = max(np.max(np.abs(lam_slepc - exact_real[:6]) / np.abs(exact_real[:6])),
              np.max(np.abs(lam_shift - exact_real[:6]) / np.abs(exact_real[:6])))
print(f"   max relative error = {error_1:.2e}")
passed &= error_1 < 1e-4

# 2) two fields coupled by a rotation
c = 10.0
W = FunctionSpace(mesh, MixedElement([FiniteElement("P", mesh.ufl_cell(), 2)] * 2))
w = Function(W)
u1, u2 = split(w)
phi1, phi2 = TestFunctions(W)
F = (inner(grad(u1), grad(phi1)) + c * u2 * phi1 + inner(grad(u2), grad(phi2)) - c * u1 * phi2) * dx


def mass(trial, test):
    a1, a2 = split(trial)
    b1, b2 = split(test)
    return (a1 * b1 + a2 * b2) * dx


bcs = [DirichletBC(W.sub(i), Constant(0.0), "on_boundary") for i in range(2)]
A, B = ls.assemble_eigenproblem(F, w, bcs, mass)
pairs = ls.leading_eigenvalues(A, B, shifts=(-20.0 + 10j, -50.0 + 10j, -80.0 + 10j), nev=6)
computed = np.array([p.eigenvalue for p in pairs])
exact_complex = np.unique(np.round(exact_real, 8))[::-1][:4] + 1j * c
print("2) rotation-coupled system: leading computed eigenvalues vs exact (Im >= 0)")
errors = []
for lam in exact_complex:
    k = np.argmin(np.abs(computed - lam))
    errors.append(abs(computed[k] - lam) / abs(lam))
    print(f"   {computed[k].real:12.6f} {computed[k].imag:+10.6f} i   exact {lam.real:12.6f} {lam.imag:+10.6f} i")
pairs_slepc = ls.solve_eigenproblem(A, B, target=-20.0, nev=8)
computed_slepc = np.array([p.eigenvalue for p in pairs_slepc])
k = np.argmin(np.abs(computed_slepc - exact_complex[0]))
errors.append(abs(computed_slepc[k] - exact_complex[0]) / abs(exact_complex[0]))
print(f"   SLEPc (real arithmetic, conjugate pairs): {computed_slepc[k]:.6f}")
error_2 = max(errors)
print(f"   max relative error = {error_2:.2e}")
passed &= error_2 < 1e-4

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(np.arange(1, 7), exact_real[:6], "o", mfc="none", ms=10, label="exact $-\\pi^2(m^2+n^2)$")
axes[0].plot(np.arange(1, 7), lam_shift.real, "x", ms=8, label="FE + shift-invert")
axes[0].set_xlabel("index")
axes[0].set_ylabel(r"$\lambda$")
axes[0].set_title("heat equation (real spectrum)")
axes[0].legend()
axes[1].plot(exact_complex.real, exact_complex.imag, "o", mfc="none", ms=10, label="exact")
axes[1].plot(exact_complex.real, -exact_complex.imag, "o", mfc="none", ms=10, color="C0")
axes[1].plot(computed.real, computed.imag, "x", ms=8, color="C1", label="FE + complex shift-invert")
axes[1].plot(computed_slepc.real, computed_slepc.imag, "+", ms=10, color="C2", label="FE + SLEPc (real)")
axes[1].set_xlim(exact_complex.real.min() - 20, 0)
axes[1].set_xlabel(r"Re $\lambda$")
axes[1].set_ylabel(r"Im $\lambda$")
axes[1].set_title("rotation-coupled system (complex pairs)")
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "check_eigensolver.png"), dpi=150)

print("CHECK EIGENSOLVER:", "PASSED" if passed else "FAILED")
