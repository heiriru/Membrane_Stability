'''
Two-dimensional long-wave (slope) equation of a lattice of anchored PIs, with all coefficients from the full model.

For the mean height H(x, y, t) of the lattice (flow along x):
    zeta H_t = d_x J_x(grad H) + d_y J_y(grad H) - (K_xx d_x^4 + K_m d_x^2 d_y^2 + K_yy d_y^4) H - zeta c H_x,
with the slope u = grad H and the macroscopic vertical stress
    J_x = u_x (sigma_xx + a1 u_x^2 + a2 u_y^2 + a3 u_x^4 + a4 u_x^2 u_y^2 + a5 u_y^4),
    J_y = u_y (sigma_yy + b1 u_y^2 + b2 u_x^2 + b3 u_y^4 + b4 u_x^2 u_y^2 + b5 u_x^4).
Coefficients:
  - sigma_xx = -eps sigma_rest (eps = D / D_lw - 1, as in the 1D equation, slope_equation.py), sigma_yy(D) linear in D
    between the Bloch values at rest and at D_lw; K's from the Bloch spectrum in every direction (lattice_transverse.py);
  - a's and b's from the homogenized nonlinear cell tilted in the directions 0, 30, 45, 60, 90 degrees
    (nonlinear_cell.py --angle) at the threshold drive, rescaled to the Bloch normalization like the 1D coefficients.
In a periodic domain the drift is a Galilean shift and is dropped. The 1D equation is the special case u_y = 0.

Analyses:
  1. coefficients and checks (sigma_yy from the tilted cell against the Bloch value; symmetry a2 = b2 of a potential
     J = dW/du);
  2. transverse stability of the uniform terrace slope u_s e_x: the slope perturbation along y grows if
     D_yy = J_y / u_y (u_s, 0) = sigma_yy + b2 u_s^2 + b5 u_s^4 < 0 (zigzag / facets) and is stable otherwise;
  3. periodic 2D runs from noise (pseudo-spectral in H, IMEX with a stabilizing Laplacian as in 1D): dense (Lc = 6)
     at eps = 0.05, 0.2 and dilute (Lc = 8) at eps = 0.05; diagnostics: wall length, orientation of the slope field,
     mean |u_x|, |u_y|.

    python3 slope_equation_2d.py [coefficient dir with nonlinear_cell_L*_a*.json, nonlinear_cell_L*.json,
                                  longwave.json, transverse.json] [out dir]
'''
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
cdir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "round6", "lattice_2d")
out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "round6", "slope2d")
only = sys.argv[3].split(",") if len(sys.argv) > 3 else None
os.makedirs(out, exist_ok=True)
t0 = time.time()
lw = {r["Lc"]: r for r in json.load(open(os.path.join(cdir, "longwave.json")))}
tr = {r["Lc"]: r for r in json.load(open(os.path.join(cdir, "transverse.json")))}


U_FIT = 0.2    # largest tilt used in the fits (beyond it some tilted cells at the threshold drive leave their branch)


def cell_rows(Lc, drive):
    rows = []
    for fn in sorted(os.listdir(cdir)):
        if not fn.startswith(f"nonlinear_cell_L{Lc:g}") or not fn.endswith(".json"):
            continue
        d = json.load(open(os.path.join(cdir, fn)))
        ang = d.get("angle", 0.0)
        for r in d["runs"]:
            if r["drive_name"] != drive:
                continue
            for x in r["rows"]:
                ux = x.get("ux", x["u"] * np.cos(np.radians(ang)))
                uy = x.get("uy", x["u"] * np.sin(np.radians(ang)))
                Jx = x.get("Jx", x["J"] if ang == 0 else None)
                Jy = x.get("Jy", 0.0 if ang == 0 else None)
                if x["u"] <= U_FIT + 1e-9:
                    rows.append(dict(angle=ang, ux=ux, uy=uy, Jx=Jx, Jy=Jy))
    return rows


def coefficients(Lc):
    b = lw[Lc]
    t = tr[Lc]["drives"]
    rest0 = [r for r in cell_rows(Lc, "zero") if r["angle"] == 0]
    us = np.array([r["ux"] for r in rest0])
    Js = np.array([r["Jx"] for r in rest0])
    s_rest_cell = np.linalg.lstsq(np.vstack([us, us ** 3, us ** 5]).T, Js, rcond=None)[0][0]
    scale = b["sigma_eff_at_rest"] / s_rest_cell
    rows = cell_rows(Lc, "lw")
    ux = np.array([r["ux"] for r in rows])
    uy = np.array([r["uy"] for r in rows])
    Jx = np.array([r["Jx"] for r in rows], dtype=float)
    Jy = np.array([r["Jy"] for r in rows], dtype=float)
    mx = np.abs(ux) > 1e-9
    Ax = np.vstack([np.ones(mx.sum()), ux[mx] ** 2, uy[mx] ** 2, ux[mx] ** 4, ux[mx] ** 2 * uy[mx] ** 2, uy[mx] ** 4]).T
    cx = np.linalg.lstsq(Ax, Jx[mx] / ux[mx], rcond=None)[0] * scale
    my = np.abs(uy) > 1e-9
    Ay = np.vstack([np.ones(my.sum()), uy[my] ** 2, ux[my] ** 2, uy[my] ** 4, ux[my] ** 2 * uy[my] ** 2, ux[my] ** 4]).T
    cy = np.linalg.lstsq(Ay, Jy[my] / uy[my], rcond=None)[0] * scale
    resx = float(np.sqrt(np.mean((Ax @ (cx / scale) - Jx[mx] / ux[mx]) ** 2)))
    resy = float(np.sqrt(np.mean((Ay @ (cy / scale) - Jy[my] / uy[my]) ** 2)))
    return dict(Lc=Lc, phi=b["phi"], sigma_rest=b["sigma_eff_at_rest"], c=b["drift_speed"], scale=float(scale),
                syy_rest=t["rest"]["sigma_yy"], syy_lw=t["lw"]["sigma_yy"], sxx_lw_bloch=t["lw"]["sigma_xx"],
                K_xx=t["lw"]["K_xx"], K_m=t["lw"]["K_mixed"], K_yy=t["lw"]["K_yy"],
                s_x_cell=float(cx[0]), a1=float(cx[1]), a2=float(cx[2]), a3=float(cx[3]), a4=float(cx[4]),
                a5=float(cx[5]), s_y_cell=float(cy[0]), b1=float(cy[1]), b2=float(cy[2]), b3=float(cy[3]),
                b4=float(cy[4]), b5=float(cy[5]), fit_rms_x=resx, fit_rms_y=resy,
                n_rows=int(len(rows)), u_max=float(np.max(np.hypot(ux, uy))))


def sig(c, eps):
    return -eps * c["sigma_rest"], c["syy_lw"] + eps * (c["syy_lw"] - c["syy_rest"])


def J2(ux, uy, c, eps):
    sx, sy = sig(c, eps)
    ux2, uy2 = ux * ux, uy * uy
    Jx = ux * (sx + c["a1"] * ux2 + c["a2"] * uy2 + c["a3"] * ux2 ** 2 + c["a4"] * ux2 * uy2 + c["a5"] * uy2 ** 2)
    Jy = uy * (sy + c["b1"] * uy2 + c["b2"] * ux2 + c["b3"] * uy2 ** 2 + c["b4"] * ux2 * uy2 + c["b5"] * ux2 ** 2)
    return Jx, Jy


def terrace_slope(c, eps):
    '''stable uniform slope along x (largest root of J_x(u, 0)/u = 0 with J_x' > 0)'''
    sx, _ = sig(c, eps)
    roots = np.roots([c["a3"], c["a1"], sx]) if c["a3"] else np.array([-sx / c["a1"]])
    u2 = sorted(r.real for r in np.atleast_1d(roots) if abs(r.imag) < 1e-12 and r.real > 0)
    for x in reversed(u2):
        u = np.sqrt(x)
        if sx + 3 * c["a1"] * u ** 2 + 5 * c["a3"] * u ** 4 > 0:
            return float(u)
    return None


co = {}
for Lc in sorted(tr):
    if not any(f.startswith(f"nonlinear_cell_L{Lc:g}_a") for f in os.listdir(cdir)):
        continue
    try:
        co[Lc] = coefficients(Lc)
    except (KeyError, ValueError, np.linalg.LinAlgError, ZeroDivisionError) as err:
        print(f"Lc = {Lc:g}: no 2D coefficients ({err})", flush=True)
        continue
    c = co[Lc]
    analysis = {}
    for eps in (0.05, 0.1, 0.2):
        us = terrace_slope(c, eps)
        Dyy = None if us is None else float(sig(c, eps)[1] + c["b2"] * us ** 2 + c["b5"] * us ** 4)
        analysis[f"{eps:g}"] = dict(u_s=us, D_yy=Dyy, transverse="unstable (zigzag)" if (Dyy is not None and Dyy < 0)
                                    else "stable" if Dyy is not None else None)
    c["terrace"] = analysis
    print(f"Lc = {Lc:g} (phi = {c['phi']:.3f}): sigma_yy rest/lw (Bloch) = {c['syy_rest']:+.4f}/{c['syy_lw']:+.4f}, "
          f"cell s_y at lw = {c['s_y_cell']:+.4f}; a1 = {c['a1']:+.3f}, a2 = {c['a2']:+.3f}, b1 = {c['b1']:+.3f}, "
          f"b2 = {c['b2']:+.3f}; K = {c['K_xx']:.3f}/{c['K_m']:.3f}/{c['K_yy']:.3f}; terraces {analysis}", flush=True)
json.dump({f"{k:g}": v for k, v in co.items()}, open(os.path.join(out, "coefficients_2d.json"), "w"), indent=1)


def run2d(c, eps, L=300.0, N=256, t_end=4e5, dt=2.0, seed=1, n_out=100, H0=None, name=""):
    ckdir = os.environ.get("SLOPE2D_CKPT", "/tmp/claude-0/runs/r6/slope2d_ckpt")   # resumable runs, outside the repo
    os.makedirs(ckdir, exist_ok=True)
    ck = os.path.join(ckdir, f"ckpt_{name}.npz")
    k1 = 2 * np.pi * np.fft.fftfreq(N, L / N)
    kx, ky = np.meshgrid(k1, k1, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    K4 = c["K_xx"] * kx ** 4 + c["K_m"] * kx ** 2 * ky ** 2 + c["K_yy"] * ky ** 4
    A_st = 2.0 * max(c["sigma_rest"], abs(sig(c, eps)[1]), 0.05)
    rng = np.random.default_rng(seed)
    if os.path.exists(ck):
        R = np.load(ck)
        H, n0 = R["H"], int(R["n"])
        ts, Hs, stats = list(R["ts"]), list(R["Hs"]), list(R["stats"])
    else:
        H = 1e-3 * rng.standard_normal((N, N)) if H0 is None else H0.copy()
        n0, ts, Hs, stats = 0, [], [], []
    nt = int(t_end / dt)
    every = max(1, nt // n_out)
    Hh = np.fft.fft2(H)
    for n in range(n0, nt + 1):
        if n % every == 0:
            H = np.real(np.fft.ifft2(Hh))
            ux, uy = np.real(np.fft.ifft2(1j * kx * Hh)), np.real(np.fft.ifft2(1j * ky * Hh))
            ts.append(n * dt)
            Hs.append(H.astype(np.float32))
            # walls: sign changes of u_x along x (terrace edges), per unit area; orientation of the slope field
            s = np.sign(ux)
            walls = np.count_nonzero(s != np.roll(s, 1, axis=0)) / N   # wall crossings per line along x
            stats.append([n * dt, float(np.mean(np.abs(ux))), float(np.mean(np.abs(uy))), walls,
                          float(np.abs(ux).max()), float(np.abs(uy).max())])
            np.savez(ck + ".tmp.npz", H=H, n=n, ts=np.array(ts), Hs=np.array(Hs), stats=np.array(stats))
            os.replace(ck + ".tmp.npz", ck)
        ux, uy = np.real(np.fft.ifft2(1j * kx * Hh)), np.real(np.fft.ifft2(1j * ky * Hh))
        Jx, Jy = J2(ux, uy, c, eps)
        div = 1j * kx * np.fft.fft2(Jx) + 1j * ky * np.fft.fft2(Jy)
        Hh = (Hh + dt * (div + A_st * k2 * Hh)) / (1 + dt * (K4 + A_st * k2))
        Hh[0, 0] = 0.0
        if n % 500 == 0 and (not np.all(np.isfinite(Hh)) or np.abs(ux).max() > 5):
            print(f"  blow-up at t = {n * dt}", flush=True)
            break
    x = np.arange(N) * L / N
    return dict(x=x, t=np.array(ts), H=np.array(Hs), stats=np.array(stats), eps=eps, Lc=c["Lc"])


for name, Lc, eps in (("dense_eps0.05", 6.0, 0.05), ("dense_eps0.2", 6.0, 0.2), ("dilute_eps0.05", 8.0, 0.05),
                      ("dense_eps0.2_local", 6.0, 0.2)):
    if Lc not in co or (only and name not in only):
        continue
    if name.endswith("_local"):
        # localized initial bump: the terrace region invades the flat lattice; across the flow (y) as a pulled front
        # with the linear spreading speed v_y = 2 sqrt(sigma_yy lambda_max), lambda_max = sigma_xx^2 / (4 K_xx)
        N_, L_ = 512, 1200.0
        xx = np.arange(N_) * L_ / N_
        X_, Y_ = np.meshgrid(xx, xx, indexing="ij")
        H0 = 0.01 * np.exp(-((X_ - L_ / 2) ** 2 + (Y_ - L_ / 2) ** 2) / 50.0)
        r = run2d(co[Lc], eps, name=name, H0=H0, t_end=6000.0, dt=1.0, n_out=60, L=L_, N=N_)
        sx, sy = sig(co[Lc], eps)
        lam_max = sx ** 2 / (4 * co[Lc]["K_xx"])
        # front position across the flow: where the envelope max_x |u_x| falls to a fixed small level
        yf = []
        for H_ in r["H"]:
            ux_ = np.gradient(H_, L_ / N_, axis=0)
            env = np.abs(ux_).max(axis=0)
            above = np.flatnonzero(env > 1e-4)
            yf.append(float((above.max() - above.min()) * L_ / N_ / 2) if len(above) else 0.0)
        r["y_front"] = np.array(yf)
        tt = r["t"]
        m = (tt > 1500) & (np.array(yf) < 0.45 * L_)
        v_meas = float(np.polyfit(tt[m], np.array(yf)[m], 1)[0]) if m.sum() > 3 else None
        r["v_y_measured"], r["v_y_linear"] = v_meas, float(2 * np.sqrt(sy * lam_max))
        print(f"localized: lateral front speed {v_meas} (linear spreading 2 sqrt(sigma_yy lambda_max) = "
              f"{r['v_y_linear']:.4f})", flush=True)
    else:
        r = run2d(co[Lc], eps, name=name)
    np.savez_compressed(os.path.join(out, f"periodic2d_{name}.npz"), **r)
    st = r["stats"][-1]
    print(f"2D {name}: <|u_x|> = {st[1]:.4f}, <|u_y|> = {st[2]:.4f}, walls per line = {st[3]:.1f}, max|u_x| = {st[4]:.3f}, "
          f"max|u_y| = {st[5]:.3f}  [{time.time() - t0:.0f} s]", flush=True)
