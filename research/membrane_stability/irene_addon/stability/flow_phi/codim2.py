'''
Codimension-two point where the static buckling of the flowing membrane and the oscillatory protein instability meet
(mobile proteins in IRENE's flowing membrane, flow_problem.py of this folder), and the weakly nonlinear (centre-
manifold) normal form there.

Stage "locate" (--stage locate). At fixed mobility M, the neutral curve in the (chi, v0) plane has a static part (a
real eigenvalue crosses zero: buckling, mostly z) and an oscillatory part (a complex pair crosses the imaginary axis:
the protein mode travelling with the flow). For each chi of --chi_list the threshold v0 of each branch is bisected
separately (the leading real eigenvalue and the leading complex pair); the codimension-two point (chi*, v0*) is where
the two thresholds coincide, refined by a secant iteration on v0_s(chi) - v0_h(chi). If the frequency of the complex
pair goes to zero there instead, the point is a Bogdanov-Takens point (reported).

Stage "normal_form" (--stage normal_form, at --chi --v0). The flat state psi0 (z = 0, phi = 0) is symmetric under
S: (z, omega, w, mu, phi, m) -> -(...), (v, sigma) -> (v, sigma); both critical modes are odd. With the dynamics
B psi_t = A d + 1/2 B2(d, d) + 1/6 C3(d, d, d) (A = -J, B2 = -D2 F, C3 = -D3 F, d = psi - psi0), the centre manifold
    d = x q0 + (z q1 + c.c.) + h200 x^2/2 + h110 x z + h101 x zbar + h011 |z|^2 + (h020 z^2/2 + c.c.) + O(3)
(A q0 = 0, A q1 = i omega B q1; the h's lie in the even subspace (v, sigma) by the symmetry, where B = 0) gives the
truncated normal form (no quadratic terms by the symmetry)
    dx/dt = mu1 x + a x^3 + b x |z|^2,
    dz/dt = (mu2 + i omega) z + c x^2 z + d |z|^2 z,
with, for left eigenvectors y0 (y0^T A = 0) and y1 (y1^T A = i omega y1^T B), normalized y0^T B q0 = y1^T B q1 = 1,
    h200 = -A^-1 B2(q0, q0),  h011 = -A^-1 B2(q1, q1bar),  h110 = (i omega B - A)^-1 B2(q0, q1),
    h020 = (2 i omega B - A)^-1 B2(q1, q1),
    a = y0^T [B2(q0, h200)/2 + C3(q0, q0, q0)/6],
    b = y0^T [B2(q0, h011) + B2(q1, h101) + B2(q1bar, h110) + C3(q0, q1, q1bar)],
    c = y1^T [B2(q0, h110) + B2(q1, h200)/2 + C3(q0, q0, q1)/2],
    d = y1^T [B2(q1, h011) + B2(q1bar, h020)/2 + C3(q1, q1, q1bar)/2].
mu1 and mu2 are the real eigenvalue and the real part of the complex pair (linear in chi - chi*, v0 - v0*; their
gradients are computed by finite differences). The modes are normalized to max |z| = 1 (x and |z| in units of r0).
The coefficients are computed for beta = 0 and beta = 1 (the quartic term enters C3 only, linearly).

run with:
STABILITY_PARAMS=zeta=1,omega_circle_const=0,pi_height=1,C_phi=0.25,eps_phi=1,M_phi=10,phi_sponge=2,phi_form=1 \
    python3 codim2.py square_b [mesh] [out] --stage locate --chi_list 0.25,0.15,0.1,0.05,0
'''
import irene_paths  # noqa: F401

import argparse
import json
import os
import time

import numpy as np
from fenics import (Constant, Function, FunctionSpace, PETScMatrix, TrialFunction, as_backend_type, assemble, assign,
                    derivative, interpolate, project, vertex_to_dof_map)
from petsc4py import PETSc

import runtime_arguments as rarg
import flow_problem as fp
import function_spaces as fsp
from stability import linear_stability as ls

parser = argparse.ArgumentParser()
parser.add_argument("--stage", choices=["locate", "grid", "normal_form", "bt", "bt_locate"], default="locate")
parser.add_argument("--v0_list", default="0.1,0.12,0.14,0.16,0.18,0.2")
parser.add_argument("--chi_list", default="0.25,0.15,0.1,0.05,0")
parser.add_argument("--chi", type=float, default=None)
parser.add_argument("--v0", type=float, default=None)
parser.add_argument("--nev", type=int, default=12)
parser.add_argument("--im_tol", type=float, default=1e-7, help="|Im lambda| below this: real eigenvalue")
parser.add_argument("--v0_max", type=float, default=0.4)
options = parser.parse_args(rarg.unknown_args)
out_dir = rarg.args.output_directory
os.makedirs(out_dir, exist_ok=True)
prm = fp.rpam.parameters
t0 = time.time()
assert fp.PI_HEIGHT == 1, "clamped PI (pi_height = 1): the time stepping (phi_dynamics.py) uses the same condition"
states = {}


def steady(v0):
    key = round(v0, 10)
    if key in states:
        fsp.psi.vector()[:] = states[key]
        fp.set_inflow(v0)
        return
    fsp.psi.vector().zero()
    assign(fsp.psi.sub(2), interpolate(Constant(prm["sigma_r_const"]), fsp.Q_sigma))
    for v in np.linspace(0, v0, 6)[1:] if v0 > 0 else [0.0]:
        fp.set_inflow(v)
        fp.newton()
    states[key] = fsp.psi.vector().get_local().copy()


spectra = {}


def spectrum(v0, chi):
    key = (round(v0, 10), round(chi, 10))
    if key not in spectra:
        fp.chi_phi.assign(chi)
        steady(v0)
        spectra[key] = fp.eigenpairs(target=0.0, nev=options.nev)
    fp.chi_phi.assign(chi)
    steady(v0)
    return spectra[key]


def branches(v0, chi):
    pairs = spectrum(v0, chi)
    real = [p for p in pairs if abs(p.eigenvalue.imag) < options.im_tol]
    cplx = [p for p in pairs if p.eigenvalue.imag > options.im_tol]
    lr = max(real, key=lambda p: p.eigenvalue.real) if real else None
    lc = max(cplx, key=lambda p: p.eigenvalue.real) if cplx else None
    return lr, lc


def growth(v0, chi, kind):
    lr, lc = branches(v0, chi)
    p = lr if kind == "real" else lc
    return (p.eigenvalue.real, p.eigenvalue.imag) if p is not None else (-np.inf, 0.0)


def threshold(chi, kind, n=10, guess=None):
    lo, hi = (0.0, 0.05) if guess is None else (0.85 * guess, 1.1 * guess)
    if growth(lo, chi, kind)[0] > 0:
        if guess is not None:
            return threshold(chi, kind, n)
        return 0.0, growth(lo, chi, kind)[1]
    while growth(hi, chi, kind)[0] < 0:
        lo, hi = hi, hi * 1.3
        if hi > options.v0_max:
            return None, None
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        if growth(mid, chi, kind)[0] > 0:
            hi = mid
        else:
            lo = mid
    vc = 0.5 * (lo + hi)
    return vc, growth(vc, chi, kind)[1]


L = fp.rmsh.parameters["L"]
res = dict(M=fp.M_phi, C=fp.C_PHI, eps=fp.eps_phi, pi_height=fp.PI_HEIGHT, phi_outflow=fp.PHI_OUTFLOW,
           phi_form=fp.PHI_FORM, sigma0=prm["sigma_r_const"])

if options.stage == "grid":
    # leading eigenvalues on a (chi, v0) grid: where the static and oscillatory branches meet
    res["grid"] = []
    for v0 in [float(v) for v in options.v0_list.split(",") if v]:
        for chi in [float(c) for c in options.chi_list.split(",") if c]:
            pairs = sorted(spectrum(v0, chi), key=lambda p: -p.eigenvalue.real)
            ev = [p.eigenvalue for p in pairs if p.eigenvalue.imag >= -1e-12][:5]
            res["grid"].append(dict(v0=v0, chi=chi, eigenvalues=[[z.real, z.imag] for z in ev]))
            print(f"v0 = {v0:.3f}, chi = {chi:+.3f}: " + ", ".join(f"{z.real:+.2e}{z.imag:+.1e}i" for z in ev)
                  + f"  [{time.time() - t0:.0f} s]", flush=True)
            json.dump(res, open(os.path.join(out_dir, "codim2_grid.json"), "w"), indent=1)
    raise SystemExit

def leading_pair(v0, chi):
    pairs = sorted(spectrum(v0, chi), key=lambda p: -p.eigenvalue.real)
    if abs(pairs[0].eigenvalue.imag) > options.im_tol:
        lam = pairs[0].eigenvalue
        return lam, np.conj(lam)
    reals = [p.eigenvalue.real for p in pairs if abs(p.eigenvalue.imag) < options.im_tol]
    cplx = [p.eigenvalue for p in pairs if p.eigenvalue.imag > options.im_tol]
    # the second member: the next real eigenvalue, unless a complex pair is closer to merging (larger real part)
    return complex(reals[0]), complex(reals[1])


def unfolding(v0, chi):
    l1, l2 = leading_pair(v0, chi)
    return np.array([np.real(-l1 * l2), np.real(l1 + l2)]), (l1, l2)


if options.stage == "bt_locate":
    # Newton on (mu1, mu2) = (-lambda1 lambda2, lambda1 + lambda2) = 0 in (chi, v0): the Bogdanov-Takens point
    x = np.array([options.chi if options.chi is not None else 0.075, options.v0 if options.v0 is not None else 0.18])
    scale = np.array([1e-10, 1e-5])         # typical sizes of mu1, mu2
    hist = []
    for it in range(8):
        F, lp = unfolding(x[1], x[0])
        hist.append(dict(chi=float(x[0]), v0=float(x[1]), mu1=float(F[0]), mu2=float(F[1]),
                         lambda1=[lp[0].real, lp[0].imag], lambda2=[lp[1].real, lp[1].imag]))
        print(f"it {it}: chi = {x[0]:+.5f}, v0 = {x[1]:.5f}: mu1 = {F[0]:+.3e}, mu2 = {F[1]:+.3e} "
              f"(lambda = {lp[0]:.3e}, {lp[1]:.3e})  [{time.time() - t0:.0f} s]", flush=True)
        res["bt_locate"] = hist
        json.dump(res, open(os.path.join(out_dir, "codim2_bt_locate.json"), "w"), indent=1)
        if abs(F[0]) < 1e-3 * scale[0] * 10 and abs(F[1]) < 5e-3 * scale[1] * 10:
            break
        Jm = np.zeros((2, 2))
        for k, h in enumerate((2e-3, 2e-3 * x[1])):
            xp = x.copy()
            xp[k] += h
            Jm[:, k] = (unfolding(xp[1], xp[0])[0] - F) / h
        dx = -np.linalg.solve(Jm / scale[:, None], F / scale)
        dx = np.clip(dx, [-0.03, -0.02], [0.03, 0.02])
        x = x + dx
    res["bt_point"] = hist[-1]
    json.dump(res, open(os.path.join(out_dir, "codim2_bt_locate.json"), "w"), indent=1)
    print("BTPOINT:", json.dumps(hist[-1]), flush=True)
    raise SystemExit

if options.stage == "locate":
    res["scan"] = []
    gs, gh = None, None
    for chi in [float(c) for c in options.chi_list.split(",") if c]:
        vs, _ = threshold(chi, "real", guess=gs)
        vh, om = threshold(chi, "complex", guess=gh)
        gs, gh = vs or gs, vh or gh
        res["scan"].append(dict(chi=chi, v0_static=vs, v0_hopf=vh, omega_hopf=om))
        print(f"chi = {chi:+.4f}: static v0 = {vs}, Hopf v0 = {vh} (omega = {om})  [{time.time() - t0:.0f} s]", flush=True)
        json.dump(res, open(os.path.join(out_dir, "codim2_locate.json"), "w"), indent=1)
    # crossing: secant on D(chi) = v0_s - v0_h between the bracketing scan points
    sc = [r for r in res["scan"] if r["v0_static"] is not None and r["v0_hopf"] is not None]
    for r1, r2 in zip(sc[:-1], sc[1:]):
        d1, d2 = r1["v0_static"] - r1["v0_hopf"], r2["v0_static"] - r2["v0_hopf"]
        if d1 * d2 < 0:
            ca, cb, da, db = r1["chi"], r2["chi"], d1, d2
            for _ in range(5):
                cm = cb - db * (cb - ca) / (db - da)
                vs, _ = threshold(cm, "real", n=12, guess=0.5 * (r1["v0_static"] + r2["v0_static"]))
                vh, om = threshold(cm, "complex", n=12, guess=0.5 * (r1["v0_hopf"] + r2["v0_hopf"]))
                dm = vs - vh
                print(f"  secant: chi = {cm:+.5f}: v0_s = {vs:.6f}, v0_h = {vh:.6f}, omega = {om:.3e}", flush=True)
                if abs(dm) < 2e-5:
                    break
                ca, da, cb, db = cb, db, cm, dm
            res["codim2"] = dict(chi=cm, v0=0.5 * (vs + vh), SL=0.5 * (vs + vh) * L, omega=om,
                                 v0_static=vs, v0_hopf=vh)
            print("CODIM2:", json.dumps(res["codim2"]), flush=True)
            break
    json.dump(res, open(os.path.join(out_dir, "codim2_locate.json"), "w"), indent=1)
    raise SystemExit

# ------------------------------------------------------------------------------------------------------ normal form
chi_s, v0_s = options.chi, options.v0
Q = fsp.Q
odd = np.concatenate([np.array(Q.sub(k).dofmap().dofs()) for k in (1, 3, 4, 5, 6, 7)])
even = np.concatenate([np.array(Q.sub(k).dofmap().dofs()) for k in (0, 2)])
bc_dofs = np.array(sorted(set().union(*[set(bc.get_boundary_values().keys()) for bc in fp.bcs_hom])), dtype=int)

if options.stage == "normal_form":
    # unfolding: gradients of mu1 (real eigenvalue) and mu2 (Re of the complex pair) in (chi, v0)
    dchi, dv = 2e-3, 1e-3 * v0_s
    lam = {}
    for key, (c_, v_) in dict(c=(chi_s, v0_s), cp=(chi_s + dchi, v0_s), cm=(chi_s - dchi, v0_s),
                              vp=(chi_s, v0_s + dv), vm=(chi_s, v0_s - dv)).items():
        lr, lc = branches(v_, c_)
        lam[key] = (lr.eigenvalue, lc.eigenvalue)
        print(f"{key}: chi = {c_:+.5f}, v0 = {v_:.6f}: real {lr.eigenvalue.real:+.3e}, complex {lc.eigenvalue:.3e}", flush=True)
    grad = dict(dmu1_dchi=(lam["cp"][0].real - lam["cm"][0].real) / (2 * dchi),
                dmu1_dv0=(lam["vp"][0].real - lam["vm"][0].real) / (2 * dv),
                dmu2_dchi=(lam["cp"][1].real - lam["cm"][1].real) / (2 * dchi),
                dmu2_dv0=(lam["vp"][1].real - lam["vm"][1].real) / (2 * dv))
    mu1_0, mu2_0 = lam["c"][0].real, lam["c"][1].real
    omega = lam["c"][1].imag
    print("unfolding gradients:", grad, flush=True)

# critical modes at the point
fp.chi_phi.assign(chi_s)
steady(v0_s)
psi0 = fsp.psi.vector().get_local().copy()
A_mat, B_mat = fp.eigenproblem()
Am, Bm = as_backend_type(A_mat).mat(), as_backend_type(B_mat).mat()
A_s, B_s = ls.petsc_to_scipy(Am), ls.petsc_to_scipy(Bm)


def zpart(x):
    f = Function(Q)
    f.vector()[:] = np.ascontiguousarray(np.real(x))
    zr = f.sub(3, deepcopy=True).vector().get_local()
    f.vector()[:] = np.ascontiguousarray(np.imag(x)) if np.iscomplexobj(x) else 0.0 * np.real(x)
    zi = f.sub(3, deepcopy=True).vector().get_local()
    return zr + 1j * zi


if options.stage == "normal_form":
    lr, lc = branches(v0_s, chi_s)
    q0 = lr.x_real.copy()
    q0 /= np.abs(zpart(q0)).max() * np.sign(zpart(q0).real[np.argmax(np.abs(zpart(q0)))])
    q1 = lc.x_real + 1j * lc.x_imag
    lam1 = lc.eigenvalue
    if np.linalg.norm(A_s @ np.conj(q1) - lam1 * (B_s @ np.conj(q1))) < np.linalg.norm(A_s @ q1 - lam1 * (B_s @ q1)):
        q1 = np.conj(q1)      # sign convention of the imaginary part (real-arithmetic SLEPc)
    print(f"direct residual |A q1 - lambda B q1| / |B q1| = "
          f"{np.linalg.norm(A_s @ q1 - lam1 * (B_s @ q1)) / np.linalg.norm(B_s @ q1):.1e}", flush=True)
    zq1 = zpart(q1)
    q1 /= zq1[np.argmax(np.abs(zq1))]                 # max |z| = 1, real at its maximum
    print(f"modes: real {lr.eigenvalue:.3e} (residual {lr.residual:.1e}), complex {lc.eigenvalue:.3e} "
          f"(residual {lc.residual:.1e}); odd/even norms q0 {np.linalg.norm(q0[odd]):.2e}/{np.linalg.norm(q0[even]):.2e}, "
          f"q1 {np.linalg.norm(q1[odd]):.2e}/{np.linalg.norm(q1[even]):.2e}", flush=True)

    # left eigenvectors from the transposed problem
    At, Bt = Am.copy(), Bm.copy()
    At.transpose()
    Bt.transpose()
    adj = ls.solve_eigenproblem(At, Bt, target=0.0, nev=options.nev)
    a0 = min(adj, key=lambda p: abs(p.eigenvalue - lr.eigenvalue))
    a1 = min([p for p in adj if p.eigenvalue.imag > 0], key=lambda p: abs(p.eigenvalue - lc.eigenvalue))
    y0 = a0.x_real.copy()
    y1 = a1.x_real + 1j * a1.x_imag
    if np.linalg.norm(A_s.T @ np.conj(y1) - lam1 * (B_s.T @ np.conj(y1))) < np.linalg.norm(A_s.T @ y1 - lam1 * (B_s.T @ y1)):
        y1 = np.conj(y1)      # left eigenvector: y1^T A = i omega y1^T B
    y0[bc_dofs] = 0.0
    y1[bc_dofs] = 0.0
    y0 /= y0 @ (B_s @ q0)
    y1 /= y1 @ (B_s @ q1)
    print(f"adjoint eigenvalues {a0.eigenvalue:.3e}, {a1.eigenvalue:.3e}; y0^T B q1 = {abs(y0 @ (B_s @ q1)):.1e}, "
          f"y1^T B q0 = {abs(y1 @ (B_s @ q0)):.1e}", flush=True)

# multilinear forms (B2 = -D2 F, C3 = -D3 F) at psi0
U1, U2, U3 = Function(Q), Function(Q), Function(Q)
d2_form = derivative(derivative(fp.F_used, fsp.psi, U1), fsp.psi, U2)
d3_form = derivative(derivative(derivative(fp.F_used, fsp.psi, U1), fsp.psi, U2), fsp.psi, U3)


def _vec(form):
    b = assemble(form).get_local()
    b[bc_dofs] = 0.0
    return -b


def _parts(x):
    # contiguous copies: a strided numpy view (np.real / np.imag of a complex array) assigned to a dolfin vector is
    # read as if it were contiguous, which scrambles the values
    out = [(np.ascontiguousarray(np.real(x)), 1.0)]
    if np.iscomplexobj(x) and np.abs(np.imag(x)).max() > 0:
        out.append((np.ascontiguousarray(np.imag(x)), 1j))
    return out


def B2(a, b):
    fsp.psi.vector()[:] = psi0
    tot = 0.0
    for xa, fa in _parts(a):
        for xb, fb in _parts(b):
            U1.vector()[:] = xa
            U2.vector()[:] = xb
            tot = tot + fa * fb * _vec(d2_form)
    return tot


def C3(a, b, c):
    fsp.psi.vector()[:] = psi0
    tot = 0.0
    for xa, fa in _parts(a):
        for xb, fb in _parts(b):
            for xc, fc in _parts(c):
                U1.vector()[:] = xa
                U2.vector()[:] = xb
                U3.vector()[:] = xc
                tot = tot + fa * fb * fc * _vec(d3_form)
    return tot


ksp = PETSc.KSP().create()
ksp.setOperators(Am)
ksp.setType("preonly")
ksp.getPC().setType("lu")
ksp.getPC().setFactorSolverType("mumps")
_r, _s = Am.createVecs()


def solve_A(rhs):
    out = np.zeros(len(rhs), dtype=complex)
    for part, f in _parts(rhs):
        _r.setArray(part)
        ksp.solve(_r, _s)
        out = out + f * _s.getArray()
    return out


def clean_even(h):
    h = np.array(h, dtype=complex)
    h[odd] = 0.0                     # the h's are even; removes the round-off amplified along the singular modes
    return h


def solve_shift(shift, rhs):
    '''(shift B - A) h = rhs'''
    sol = ls.ComplexShiftedSolver(A_s, B_s, shift)
    h = -sol.solve(np.asarray(rhs, dtype=complex))
    sol.destroy()
    return h


if options.stage == "bt":
    # Bogdanov-Takens point (static branch ends where the real pair merges into the oscillatory one): normal form from
    # the two leading eigenvalues lambda1, lambda2 (a real pair or a complex pair, both close to 0) and their right /
    # left eigenvectors phi_i, psi_i (psi_i^T B phi_j = delta_ij). In the basis
    #     q0 = (lambda2 phi1 - lambda1 phi2) / (lambda2 - lambda1),  q1 = (phi2 - phi1) / (lambda2 - lambda1)
    # (real; the Jordan chain A q0 = 0, A q1 = B q0 in the limit) the linear dynamics is the companion form
    #     xi0' = xi1,  xi1' = mu1 xi0 + mu2 xi1,   mu1 = -lambda1 lambda2,  mu2 = lambda1 + lambda2,
    # with the dual rows y0 = psi1 + psi2, y1 = lambda1 psi1 + lambda2 psi2. The even sector (v, sigma) has no mass
    # (rho = 0), so it is slaved instantaneously: h(xi) = -A^-1 B2(W xi, W xi) / 2, and the reduced field is
    #     xi' = C xi + Y [B2(W xi, h(xi)) + C3(W xi, W xi, W xi) / 6]   (cubic; no quadratic terms by the symmetry).
    # Writing xi0' = xi1 + f(xi), xi1' = mu1 xi0 + mu2 xi1 + g(xi), the Z2-symmetric Bogdanov-Takens normal form
    #     x'' = mu1 x + mu2 x' + a x^3 + b x^2 x'
    # has a = g_30, b = g_21 + 3 f_30 (coefficients of xi0^3 and xi0^2 xi1).
    fp.chi_phi.assign(chi_s)
    steady(v0_s)
    pairs = sorted(spectrum(v0_s, chi_s), key=lambda p: -p.eigenvalue.real)
    p1 = pairs[0]
    if abs(p1.eigenvalue.imag) > options.im_tol:
        lam1 = p1.eigenvalue if p1.eigenvalue.imag > 0 else np.conj(p1.eigenvalue)
        phi1 = p1.x_real + 1j * p1.x_imag
        if np.linalg.norm(A_s @ np.conj(phi1) - lam1 * (B_s @ np.conj(phi1))) < np.linalg.norm(A_s @ phi1 - lam1 * (B_s @ phi1)):
            phi1 = np.conj(phi1)
        lam2, phi2 = np.conj(lam1), np.conj(phi1)
    else:
        reals = [p for p in pairs if abs(p.eigenvalue.imag) < options.im_tol]
        lam1, lam2 = reals[0].eigenvalue.real, reals[1].eigenvalue.real
        phi1, phi2 = reals[0].x_real.copy(), reals[1].x_real.copy()
    # normalize both eigenvectors to max |z| = 1 with the same sign (continuous through the merging)
    if np.iscomplexobj(phi1):
        zz = zpart(phi1)
        phi1 = phi1 / zz[np.argmax(np.abs(zz))]
        phi2 = np.conj(phi1)
    else:
        def _znorm(x):
            f_ = Function(Q)
            f_.vector()[:] = x
            z_ = f_.sub(3, deepcopy=True).vector().get_local()
            return x / z_[np.argmax(np.abs(z_))]
        phi1, phi2 = _znorm(phi1).astype(complex), _znorm(phi2).astype(complex)
    At, Bt = Am.copy(), Bm.copy()
    At.transpose()
    Bt.transpose()
    adj = ls.solve_eigenproblem(At, Bt, target=0.0, nev=options.nev)

    def left(lam_, phi_):
        cand = min(adj, key=lambda p: abs(p.eigenvalue - lam_) if abs(lam_.imag) < options.im_tol
                   else abs(p.eigenvalue - lam_) + 1e9 * (p.eigenvalue.imag * lam_.imag < 0))
        y = cand.x_real + 1j * cand.x_imag
        if np.linalg.norm(A_s.T @ np.conj(y) - lam_ * (B_s.T @ np.conj(y))) < np.linalg.norm(A_s.T @ y - lam_ * (B_s.T @ y)):
            y = np.conj(y)
        y[bc_dofs] = 0.0
        return y / (y @ (B_s @ phi_))

    psi1 = left(complex(lam1), phi1)
    psi2 = np.conj(psi1) if abs(complex(lam1).imag) > options.im_tol else left(complex(lam2), phi2)
    nrm = lambda x: float(np.linalg.norm(x))  # noqa: E731
    print(f"norms: phi {nrm(phi1):.3e} {nrm(phi2):.3e}, psi {nrm(psi1):.3e} {nrm(psi2):.3e}; psi^T B phi = "
          f"{[[complex(a_ @ (B_s @ b_)) for b_ in (phi1, phi2)] for a_ in (psi1, psi2)]}; |phi1 - phi2| = "
          f"{nrm(phi1 - phi2):.3e}", flush=True)
    q0 = np.ascontiguousarray(np.real((lam2 * phi1 - lam1 * phi2) / (lam2 - lam1)))
    q1 = np.ascontiguousarray(np.real((phi2 - phi1) / (lam2 - lam1)))
    y0 = np.ascontiguousarray(np.real(psi1 + psi2))
    y1 = np.ascontiguousarray(np.real(lam1 * psi1 + lam2 * psi2))
    mu1, mu2 = float(np.real(-lam1 * lam2)), float(np.real(lam1 + lam2))
    print(f"norms q0 {nrm(q0):.3e} q1 {nrm(q1):.3e} y0 {nrm(y0):.3e} y1 {nrm(y1):.3e}; y0 B q0 {y0 @ (B_s @ q0)}; "
          f"psi1 B q0 {complex(psi1 @ (B_s @ q0))}; lam {lam1!r} {lam2!r}", flush=True)
    G = np.array([[y0 @ (B_s @ q0), y0 @ (B_s @ q1)], [y1 @ (B_s @ q0), y1 @ (B_s @ q1)]])
    C_lin = np.array([[y0 @ (A_s @ q0), y0 @ (A_s @ q1)], [y1 @ (A_s @ q0), y1 @ (A_s @ q1)]])
    print(f"lambda1 = {complex(lam1):.4e}, lambda2 = {complex(lam2):.4e}: mu1 = {mu1:+.4e}, mu2 = {mu2:+.4e}; "
          f"Y^T B W = {G.tolist()}, Y^T A W = {C_lin.tolist()}", flush=True)
    H00 = clean_even(-0.5 * solve_A(B2(q0, q0)))
    H01 = clean_even(-solve_A(B2(q0, q1)))
    H11 = clean_even(-0.5 * solve_A(B2(q1, q1)))
    Nq = dict(
        n30=B2(q0, H00), n21=B2(q0, H01) + B2(q1, H00), n12=B2(q0, H11) + B2(q1, H01), n03=B2(q1, H11))
    out = {}
    for beta in (0.0, 1.0):
        fp.beta_phi.assign(beta)
        N = dict(n30=Nq["n30"] + C3(q0, q0, q0) / 6.0, n21=Nq["n21"] + C3(q0, q0, q1) / 2.0,
                 n12=Nq["n12"] + C3(q0, q1, q1) / 2.0, n03=Nq["n03"] + C3(q1, q1, q1) / 6.0)
        f = {k: float(np.real(y0 @ v)) for k, v in N.items()}
        g = {k: float(np.real(y1 @ v)) for k, v in N.items()}
        # at the BT point (mu = 0): a = g30, b = g21 + 3 f30; slightly off it, the near-identity transformation
        # y -> y + f adds O(mu) corrections (reported separately; they vanish at the point)
        a_bt, b_bt = g["n30"], g["n21"] + 3 * f["n30"]
        a_eff = g["n30"] - mu2 * f["n30"] + mu1 * f["n21"]
        b_eff = g["n21"] + 3 * f["n30"] - mu2 * f["n21"] + 2 * mu1 * f["n12"]
        out[str(beta)] = dict(f=f, g=g, a=a_bt, b=b_bt, a_eff=a_eff, b_eff=b_eff)
        print(f"beta = {beta}: a = {a_bt:+.4e}, b = {b_bt:+.4e}; with the O(mu) corrections a = {a_eff:+.4e}, "
              f"b = {b_eff:+.4e} (f = {f}, g = {g})  [{time.time() - t0:.0f} s]", flush=True)
    fp.beta_phi.assign(0.0)
    zq0, zq1 = np.real(zpart(q0)), np.real(zpart(q1))
    res.update(zmax_q0=float(np.abs(zq0).max()), zmax_q1=float(np.abs(zq1).max()))
    f_ = Function(Q)
    P1_ = FunctionSpace(Q.mesh(), "P", 1)
    modes = {}
    for nm, vec in (("q0", q0), ("q1", q1)):
        f_.vector()[:] = np.ascontiguousarray(vec)
        modes[f"{nm}_z"] = project(f_.sub(3), P1_).vector().get_local()
        modes[f"{nm}_phi"] = project(f_.sub(6), P1_).vector().get_local()
    xy_ = P1_.tabulate_dof_coordinates()
    np.savez(os.path.join(out_dir, f"bt_modes_chi{chi_s:g}_v{v0_s:g}.npz"), x=xy_[:, 0], y=xy_[:, 1],
             triangles=vertex_to_dof_map(P1_)[Q.mesh().cells()], **modes)
    res.update(stage="bt", chi=chi_s, v0=v0_s, SL=v0_s * L, lambda1=[complex(lam1).real, complex(lam1).imag],
               lambda2=[complex(lam2).real, complex(lam2).imag], mu1=mu1, mu2=mu2, gram=G.tolist(),
               linear_check=C_lin.tolist(), coefficients=out, wall=time.time() - t0)
    json.dump(res, open(os.path.join(out_dir, f"codim2_bt_chi{chi_s:g}_v{v0_s:g}.json"), "w"), indent=1)
    print("BT:", json.dumps({k: v for k, v in res.items() if k in ("chi", "v0", "mu1", "mu2", "coefficients")}), flush=True)
    raise SystemExit


t1 = time.time()
b00 = B2(q0, q0)
b01 = B2(q0, q1)
b11c = B2(q1, np.conj(q1))
b11 = B2(q1, q1)
sym = max(np.linalg.norm(v[odd]) / max(np.linalg.norm(v), 1e-300) for v in (b00, b01, b11c, b11))
print(f"quadratic terms: largest odd fraction {sym:.1e} (0 by symmetry)  [{time.time() - t1:.0f} s]", flush=True)
h200 = clean_even(-solve_A(b00))
h011 = clean_even(-solve_A(b11c))
h110 = clean_even(solve_shift(1j * omega, b01))
h020 = clean_even(solve_shift(2j * omega, b11))
h101 = np.conj(h110)
quad = dict(
    a=y0 @ (0.5 * B2(q0, h200)),
    b=y0 @ (B2(q0, h011) + B2(q1, h101) + B2(np.conj(q1), h110)),
    c=y1 @ (B2(q0, h110) + 0.5 * B2(q1, h200)),
    d=y1 @ (B2(q1, h011) + 0.5 * B2(np.conj(q1), h020)))
coef = {}
for beta in (0.0, 1.0):
    fp.beta_phi.assign(beta)
    cub = dict(a=y0 @ C3(q0, q0, q0) / 6.0, b=y0 @ C3(q0, q1, np.conj(q1)),
               c=y1 @ C3(q0, q0, q1) / 2.0, d=y1 @ C3(q1, q1, np.conj(q1)) / 2.0)
    coef[beta] = {k: complex(quad[k] + cub[k]) for k in quad}
    coef[beta]["parts"] = {k: dict(quadratic=complex(quad[k]), cubic=complex(cub[k])) for k in quad}
    print(f"beta = {beta}: " + ", ".join(f"{k} = {coef[beta][k]:.4e}" for k in "abcd") + f"  [{time.time() - t1:.0f} s]",
          flush=True)
fp.beta_phi.assign(0.0)


def classify(cf):
    a, b, c, d = cf["a"].real, cf["b"].real, cf["c"].real, cf["d"].real
    det = a * d - b * c
    return dict(a=a, b=b, Re_c=c, Re_d=d, det=det, theta=b / d if d else None, delta=c / a if a else None,
                static_branch="supercritical" if a < 0 else "subcritical",
                hopf_branch="supercritical" if d < 0 else "subcritical",
                mixed_modes=("stable mixed mode (buckled membrane with travelling protein waves) between the two "
                             "secondary bifurcation lines" if (a < 0 and d < 0 and det > 0) else
                             "no stable mixed mode: bistability between the pure static and oscillatory states"
                             if (a < 0 and d < 0) else "at least one primary branch subcritical"))


res.update(chi=chi_s, v0=v0_s, SL=v0_s * L, omega=omega, period=2 * np.pi / omega if omega else None, mu1=mu1_0,
           mu2=mu2_0, gradients=grad, symmetry_check=sym,
           coefficients={str(k): dict(**{kk: [cf[kk].real, cf[kk].imag] for kk in "abcd"},
                                      parts={kk: {p: [q.real, q.imag] for p, q in cf["parts"][kk].items()}
                                             for kk in "abcd"}) for k, cf in coef.items()},
           classification={str(k): classify(cf) for k, cf in coef.items()}, wall=time.time() - t0)
json.dump(res, open(os.path.join(out_dir, "codim2_normal_form.json"), "w"), indent=1)
print("NORMALFORM:", json.dumps(res["classification"]), flush=True)
