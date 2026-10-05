'''
Mobile, curvature-coupled proteins carried by the membrane flow: linear stability of the flat, uniform state.

Free energy (kappa = 1, lengths r0; phi = deviation of the protein area fraction, C0 = spontaneous curvature per unit
phi, chi = osmotic stiffness (< 0 inside the spinodal of the proteins alone), eps = gradient (line-tension) energy):
    F = int [ 1/2 (Delta z - C0 phi)^2 + sigma/2 |grad z|^2 + chi/2 phi^2 + eps/2 |grad phi|^2 ].
Dynamics: the shape relaxes against the normal friction, the proteins are conserved and advected by the in-plane flow
U e_x, the shape is not (a tangential flow does not move the surface):
    zeta dz/dt = -dF/dz,      dphi/dt + U d_x phi = M Delta dF/dphi.
For a mode exp(i k.x + lambda t):
    zeta lambda z = -(k^4 + sigma k^2) z - C0 k^2 phi
    lambda phi + i U k_x phi = -M k^2 [ C0 k^2 z + (C0^2 + chi + eps k^2) phi ].
Without flow (U = 0, Leibler / Andelman type): unstable iff chi + eps k^2 < -sigma C0^2 / (k^2 + sigma) for some k.
The tension penalizes the curvature of long-wave protein clusters (+C0^2 at k -> 0: macroscopic demixing only for
chi < -C0^2), less so at finite k: for C0 > sqrt(eps sigma) the first instability is at
    chi_c0 = eps sigma - 2 C0 sqrt(eps sigma),   k*^2 = C0 sqrt(sigma / eps) - sigma,
a modulated (microphase-separated) state of curved protein-rich domains with wavelength 2 pi / k*.
With flow, the relative drift between the advected proteins and the static shape makes the eigenvalues complex: the
threshold chi_c(U) and the selected wavevector (along the flow: travelling bands; across: static stripes parallel to
the flow) are computed numerically (max over k of Re lambda, bisection on chi).

    python3 analysis_mobile.py [out]
'''
import json
import os
import sys

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/runs/mobile"
os.makedirs(out, exist_ok=True)


def eig(kx, ky, chi, U, C0, sigma, eps, M, zeta):
    k2 = kx ** 2 + ky ** 2
    J = np.array([[-(k2 ** 2 + sigma * k2) / zeta, -C0 * k2 / zeta],
                  [-M * C0 * k2 ** 2, -M * k2 * (C0 ** 2 + chi + eps * k2) - 1j * U * kx]], dtype=complex)
    return np.linalg.eigvals(J)


def growth(chi, U, p, nk=121, kmax=None, angles=None):
    kmax = kmax or 3 * np.sqrt(p["C0"] * np.sqrt(p["sigma"] / p["eps"]))
    ks = np.linspace(1e-3, kmax, nk)
    best = (-np.inf, None, None)
    for th in (np.linspace(0, np.pi / 2, 19) if angles is None else angles):   # angle of k to the flow
        for k in ks:
            lam = eig(k * np.cos(th), k * np.sin(th), chi, U, **p)
            i = np.argmax(lam.real)
            if lam[i].real > best[0]:
                best = (lam[i].real, (k, th), lam[i])
    return best


def chi_c(U, p, angles=None):
    lo, hi = -5.0, 5.0     # unstable at lo, stable at hi
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if growth(mid, U, p, angles=angles)[0] > 0:
            lo = mid
        else:
            hi = mid
    g, (k, th), lam = growth(lo, U, p, angles=angles)
    return 0.5 * (lo + hi), k, th, lam


res = []
for C0, eps, M in ((0.2, 1.0, 1.0), (0.2, 1.0, 10.0), (0.5, 1.0, 1.0)):
    p = dict(C0=C0, sigma=0.0025, eps=eps, M=M, zeta=1.0)
    chi0 = p["eps"] * p["sigma"] - 2 * C0 * np.sqrt(p["eps"] * p["sigma"])
    kstar0 = np.sqrt(max(C0 * np.sqrt(p["sigma"] / p["eps"]) - p["sigma"], 0))
    rows = []
    for U in (0.0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1):
        c, k, th, lam = chi_c(U, p)
        ca, ka, _, la = chi_c(U, p, angles=[0.0])          # modulations along the flow only
        rows.append(dict(U=U, chi_c=c, k=k, angle_deg=float(np.degrees(th)), omega=float(lam.imag),
                         chi_c_along_flow=ca, k_along=ka, omega_along=float(la.imag),
                         band_speed_along=float(la.imag / ka) if ka > 0 else 0.0,
                         phase_speed=float(lam.imag / (k * np.cos(th))) if np.cos(th) > 1e-9 and k > 0 else 0.0))
        print(f"C0 = {C0}, eps = {eps}, M = {M}: U = {U:g}: chi_c = {c:+.5f} (analytic U = 0: {chi0:+.5f}), "
              f"k* = {k:.4f} (analytic {kstar0:.4f}), angle {np.degrees(th):.0f} deg; along the flow only: chi_c = "
              f"{ca:+.5f}, band speed {la.imag / ka:+.2e} (U = {U:g})",
              flush=True)
    res.append(dict(params=p, chi_c0_analytic=chi0, k_star0_analytic=kstar0, scan=rows))
json.dump(res, open(os.path.join(out, "mobile.json"), "w"), indent=1)
