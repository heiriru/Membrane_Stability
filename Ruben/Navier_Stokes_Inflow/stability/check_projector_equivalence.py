'''
Check that the mixed (u, p) eigenvalue problem of linear_stability.py is the same as the eigenvalue problem of the
Leray-projected linearized operator P L (the approach of stability_analysis_test.py), and why the projector used
there, P_old = I - G (DG)^{-1} D with DG the P1 Laplacian, does not give the right eigenvalues.

Discrete setting (interior dofs): M du/dt = L u + G p, D u = 0, with G = D^T.
The discrete Leray projector, orthogonal in the M inner product and exact at the discrete level, is
    P = I - M^{-1} G (D M^{-1} G)^{-1} D,       P^2 = P,   D P = 0.
The eigenvalues of the mixed problem [[L, G], [D, 0]] x = lambda [[M, 0], [0, 0]] x are the eigenvalues of
P M^{-1} L on the divergence-free subspace range(P) (the remaining eigenvalues of P M^{-1} L are 0).
P_old replaces the discrete Schur complement D M^{-1} G by the Laplacian matrix and omits M^{-1}: it is not a
projector (P_old^2 != P_old) and P_old u is not discretely divergence free.

Dense linear algebra: use a coarse mesh, e.g. the mesh of this folder (224 cells).

run with:
python3 check_projector_equivalence.py square mesh solution/checks --Re 50
'''
import argparse
import importlib
import os

import numpy as np
import scipy.linalg as sla
from fenics import TrialFunction, TestFunction, inner, grad, assemble, DirichletBC, Constant
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import runtime_arguments as rarg
import switch_problem as swi
import steady_state as ss
import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--Re", type=float, default=50.0)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)

vp = importlib.import_module(swi.vp_steady)
rmsh = vp.rmsh
W = vp.W
ss.solve_steady(vp, options.Re)

# mixed problem
A, B = ls.assemble_eigenproblem(vp.F, vp.up, vp.bcs_hom, vp.mass)
A_d, B_d = ls.petsc_to_scipy(A).toarray(), ls.petsc_to_scipy(B).toarray()
lam_mixed = sla.eig(A_d, B_d, right=False)
lam_mixed = lam_mixed[np.isfinite(lam_mixed) & (np.abs(lam_mixed) < 1e8)]

# blocks on the interior dofs
bc_dofs = np.unique(np.concatenate([list(bc.get_boundary_values().keys()) for bc in vp.bcs_hom])).astype(int)
dofs_u = np.setdiff1d(np.array(W.sub(0).dofmap().dofs()), bc_dofs)
dofs_p = np.setdiff1d(np.array(W.sub(1).dofmap().dofs()), bc_dofs)
L = A_d[np.ix_(dofs_u, dofs_u)]
G = A_d[np.ix_(dofs_u, dofs_p)]
D = A_d[np.ix_(dofs_p, dofs_u)]
M = B_d[np.ix_(dofs_u, dofs_u)]
print(f"n_u = {len(dofs_u)}, n_p = {len(dofs_p)}, |G - D^T| = {np.abs(G - D.T).max():.1e}")

# discrete Leray projector
M_inv_G = np.linalg.solve(M, G)
P = np.eye(len(dofs_u)) - M_inv_G @ np.linalg.solve(D @ M_inv_G, D)
lam_proj = np.linalg.eigvals(P @ np.linalg.solve(M, L))
lam_proj = lam_proj[np.abs(lam_proj) > 1e-8 * np.abs(lam_proj).max()]

# projector of stability_analysis_test.py (P1 Laplacian instead of the discrete Schur complement, no M^{-1})
Q = W.sub(1).collapse()
p_tr, q_te = TrialFunction(Q), TestFunction(Q)
DG_full = ls.petsc_to_scipy(assemble(-inner(grad(p_tr), grad(q_te)) * rmsh.dx)).toarray()
# map the interior pressure dofs of W to the dofs of the collapsed space Q
_, collapse_map = W.sub(1).collapse(collapsed_dofs=True)
W_to_Q = {w_dof: q_dof for q_dof, w_dof in collapse_map.items()}
q_idx = np.array([W_to_Q[d] for d in dofs_p])
DG = DG_full[np.ix_(q_idx, q_idx)]
D_old = -D  # b_D = -q div(du)
P_old = np.eye(len(dofs_u)) - D_old.T @ np.linalg.solve(DG, D_old)
lam_old = np.linalg.eigvals(np.linalg.solve(M, P_old @ L))

rng = np.random.default_rng(0)
x = rng.standard_normal(len(dofs_u))
print(f"discrete Leray projector: |P^2 - P| = {np.abs(P @ P - P).max():.1e}, |D P x| / |D x| = "
      f"{np.linalg.norm(D @ P @ x) / np.linalg.norm(D @ x):.1e}")
print(f"old projector:            |P^2 - P| = {np.abs(P_old @ P_old - P_old).max():.1e}, |D P x| / |D x| = "
      f"{np.linalg.norm(D @ P_old @ x) / np.linalg.norm(D @ x):.1e}")


def leading(lams, k):
    return lams[np.argsort(-lams.real)][:k]


k = 12
lead_mixed, lead_proj, lead_old = leading(lam_mixed, k), leading(lam_proj, k), leading(lam_old, k)
matching = max(np.min(np.abs(lam_proj - lam)) / abs(lam) for lam in lead_mixed)
print(f"number of finite eigenvalues: mixed = {len(lam_mixed)}, P M^-1 L (non-zero) = {len(lam_proj)}, "
      f"expected n_u - n_p = {len(dofs_u) - len(dofs_p)}")
print(f"leading eigenvalues (Re = {options.Re}):")
for a, b, c in zip(lead_mixed, lead_proj, lead_old):
    print(f"   mixed {a.real:+10.5f}{a.imag:+10.5f}i   P M^-1 L {b.real:+10.5f}{b.imag:+10.5f}i   "
          f"old P {c.real:+10.5f}{c.imag:+10.5f}i")
print(f"max relative distance of the {k} leading mixed eigenvalues to the spectrum of P M^-1 L = {matching:.1e}")
print("CHECK PROJECTOR EQUIVALENCE:", "PASSED" if matching < 1e-6 else "FAILED")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(lam_mixed.real, lam_mixed.imag, "o", mfc="none", ms=8, label="mixed (u, p) problem (this work)")
ax.plot(lam_proj.real, lam_proj.imag, "x", ms=6, label=r"discrete Leray projector $P M^{-1} L$")
ax.plot(lam_old.real, lam_old.imag, "+", ms=6, alpha=0.7, label=r"old projector $I - G(DG)^{-1}D$")
ax.set_xlim(lead_mixed.real.min() * 1.6 - 0.5, max(0.5, lead_old.real.max() + 0.5))
ax.set_ylim(-3, 3)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel(r"Re $\lambda$")
ax.set_ylabel(r"Im $\lambda$")
ax.set_title(f"'square' problem, Re = {options.Re:g}: leading eigenvalues")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "check_projector_equivalence.png"), dpi=150)
