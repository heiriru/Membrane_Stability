'''
Protein inclusion (PI) pulled out of a ring-bounded membrane beyond the Monge range: axisymmetric shape equations in
arclength parametrization (overhangs and tubes allowed), solved as a boundary value problem (scipy solve_bvp) and
continued in the PI height h.

Energy (as in IRENE's ring problem): E = 2 pi int [ kappa/2 (psi' + sin psi / r)^2 + sigma ] r ds  between the PI rim
(r = r0, z = h, tangent angle psi = atan(t), t = tan alpha the contact slope dz/dr) and the ring (r = R, z = 0,
psi = 0). With Lagrange multipliers gamma (r' = cos psi) and eta (z' = sin psi) the Euler-Lagrange equations are
    psi'   = u
    u'     = -cos psi u / r + sin psi cos psi / r^2 + (gamma sin psi - eta cos psi) / (kappa r)
    r'     = cos psi,   z' = sin psi
    gamma' = kappa/2 (u^2 - sin^2 psi / r^2) + sigma,   eta' = 0,
with the free total length S fixed by the Hamiltonian
    H = kappa r/2 (u^2 - sin^2 psi / r^2) - sigma r + gamma cos psi + eta sin psi = 0.
The force on the PI is F = -dE/dh = 2 pi eta (checked against finite differences of E and against IRENE's virtual-work
force in the Monge range). Units: lengths r0, energies kappa, forces kappa / r0.

Under displacement control (h imposed) the axisymmetric branch is followed; under force control a state is unstable
where dF/dh > 0 (Schur complement, as for the FE analysis), i.e. beyond the force extrema. For a large ring the force
approaches the tether force f0 = 2 pi sqrt(2 kappa sigma) and the tube radius r_t = sqrt(kappa / (2 sigma)).

run with:
    python3 ring_tube.py --R 10 --tan 0.5 --h_max 60 --out results/tube_R10_tan0.5
'''
import argparse
import csv
import json
import os

import numpy as np
from scipy.integrate import solve_bvp, trapezoid

parser = argparse.ArgumentParser()
parser.add_argument("--R", type=float, default=10.0)
parser.add_argument("--tan", type=float, default=0.5)
parser.add_argument("--sigma", type=float, default=0.0025)
parser.add_argument("--kappa", type=float, default=1.0)
parser.add_argument("--r0", type=float, default=1.0)
parser.add_argument("--h_max", type=float, default=60.0)
parser.add_argument("--h_min", type=float, default=None, help="(negative) lowest height (default -h_max)")
parser.add_argument("--dh", type=float, default=0.25)
parser.add_argument("--dh_max", type=float, default=2.0)
parser.add_argument("--n_nodes", type=int, default=400)
parser.add_argument("--tol", type=float, default=1e-7)
parser.add_argument("--profiles_every", type=float, default=None, help="store profiles every this much h")
parser.add_argument("--out", required=True)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
kap, sig, r0, R = args.kappa, args.sigma, args.r0, args.R


def rhs(tau, Y, p):
    S, eta = p
    psi, u, r, z, gam = Y
    c, s = np.cos(psi), np.sin(psi)
    du = -c * u / r + s * c / r ** 2 + (gam * s - eta * c) / (kap * r)
    dgam = 0.5 * kap * (u ** 2 - s ** 2 / r ** 2) + sig
    return S * np.vstack([u, du, c, s, dgam])


def hamiltonian(Y, eta):
    psi, u, r, z, gam = Y
    return 0.5 * kap * r * (u ** 2 - np.sin(psi) ** 2 / r ** 2) - sig * r + gam * np.cos(psi) + eta * np.sin(psi)


class BC:
    def __init__(self):
        self.h, self.psi0 = 0.0, 0.0

    def __call__(self, Ya, Yb, p):
        return np.array([Ya[2] - r0, Ya[3] - self.h, Ya[0] - self.psi0, Yb[2] - R, Yb[3], Yb[0],
                         hamiltonian(Ya, p[1])])


bc = BC()


def energy(sol):
    tau = np.linspace(0, 1, 4001)
    psi, u, r, z, gam = sol.sol(tau)
    S = sol.p[0]
    dens = (0.5 * kap * (u + np.sin(psi) / r) ** 2 + sig) * r
    return 2 * np.pi * S * trapezoid(dens, tau)


def solve(guess, h, psi0):
    bc.h, bc.psi0 = h, psi0
    tau, Y, p = guess
    s = solve_bvp(rhs, bc, tau, Y, p=p, tol=args.tol, max_nodes=200000, verbose=0)
    return s


# flat start (psi0 = 0, h = 0), then the contact angle
tau = np.linspace(0, 1, args.n_nodes)
r_lin = r0 + (R - r0) * tau
Y0 = np.vstack([0 * tau, 0 * tau, r_lin, 0 * tau, sig * r_lin])
p0 = np.array([R - r0, 0.0])
sol = solve((tau, Y0, p0), 0.0, 0.0)
assert sol.success, sol.message
for a in np.linspace(0, np.arctan(args.tan), 11)[1:]:
    sol = solve((sol.x, sol.y, sol.p), 0.0, a)
    assert sol.success, sol.message
psi0 = np.arctan(args.tan)
start = sol


def record(sol, h):
    tau = np.linspace(0, 1, 2001)
    psi, u, r, z, gam = sol.sol(tau)
    S, eta = sol.p
    # neck: smallest radius beyond the PI region (after the profile has turned away from the PI)
    return dict(h=h, F=2 * np.pi * eta, energy=energy(sol), length=S, max_psi=float(np.abs(psi).max()),
                overhang=bool(np.abs(psi).max() > np.pi / 2 + 1e-9), r_min=float(r.min()), r_max=float(r.max()),
                z_min=float(z.min()), z_max=float(z.max()), nodes=int(sol.x.size), H_err=float(
                np.abs(hamiltonian(sol.sol(tau), eta)).max()))


rows, profiles = [], {}
every = args.profiles_every or max(1.0, round(args.h_max / 30))
h_lo = -args.h_max if args.h_min is None else args.h_min
for direction, h_end in ((+1, args.h_max), (-1, h_lo)):
    sol, h, dh = start, 0.0, args.dh
    if direction == +1:
        rows.append(record(sol, 0.0))
        profiles["0"] = sol.sol(np.linspace(0, 1, 801))
    next_profile = every
    while direction * h < direction * h_end - 1e-12:
        h_try = h + direction * min(dh, abs(h_end - h))
        new = solve((sol.x, sol.y, sol.p), h_try, psi0)
        if not new.success or abs(new.p[0]) > 50 * (R + abs(h_try)):
            dh *= 0.5
            if dh < 1e-4:
                print(f"R = {R}, tan = {args.tan}: continuation stopped at h = {h:.4f} ({new.message})", flush=True)
                break
            continue
        sol, h = new, h_try
        rows.append(record(sol, h))
        if abs(h) >= next_profile - 1e-9:
            profiles[f"{h:g}"] = sol.sol(np.linspace(0, 1, 801))
            next_profile += every
        dh = min(dh * 1.3, args.dh_max)
rows.sort(key=lambda r: r["h"])

# force check: F = -dE/dh by finite differences on the computed branch
hs = np.array([r["h"] for r in rows])
Es = np.array([r["energy"] for r in rows])
Fs = np.array([r["F"] for r in rows])
F_fd = -np.gradient(Es, hs)
inner = slice(2, -2)
force_check = float(np.max(np.abs(F_fd[inner] - Fs[inner])) / max(np.abs(Fs).max(), 1e-12))

# force extrema (force-control stability boundaries) and the tether force
dF = np.gradient(Fs, hs)
ext = [dict(h=float(hs[i]), F=float(Fs[i]), kind="max" if dF[i] > 0 >= dF[i + 1] else "min")
       for i in range(len(hs) - 1) if (dF[i] > 0) != (dF[i + 1] > 0)]
f0 = 2 * np.pi * np.sqrt(2 * kap * sig)
info = dict(R=R, tan_alpha=args.tan, sigma=sig, kappa=kap, r0=r0, tether_force=f0, tube_radius=np.sqrt(kap / (2 * sig)),
            h_range=[float(hs.min()), float(hs.max())], extrema=ext, force_check_rel=force_check,
            first_overhang_h={"up": next((r["h"] for r in rows if r["overhang"] and r["h"] > 0), None),
                              "down": next((r["h"] for r in rows[::-1] if r["overhang"] and r["h"] < 0), None)},
            F_end={"up": rows[-1]["F"], "down": rows[0]["F"]})
with open(os.path.join(args.out, "branch.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
np.savez(os.path.join(args.out, "profiles.npz"), **{k: v for k, v in profiles.items()})
json.dump(info, open(os.path.join(args.out, "info.json"), "w"), indent=1)
print("TUBE:", json.dumps(info), flush=True)
