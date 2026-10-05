'''
Unfolding of the Z2-symmetric Bogdanov-Takens normal form computed by codim2.py --stage bt:
    x'' = mu1 x + mu2 x' + a x^3 + b x^2 x',      a < 0, b < 0 (both primary bifurcations supercritical).
Scaling x = alpha X, t = tau T with tau = b / a, alpha^2 = |a| / b^2 (a, b < 0) gives
    X'' = nu1 X + nu2 X' - X^3 - X^2 X',      nu1 = mu1 tau^2, nu2 = mu2 tau.
The attractors of the scaled system are found by integrating from several initial conditions on a grid of directions in
the (nu1, nu2) plane: the flat state (origin), the static buckled states (X = +-sqrt(nu1)), a small oscillation around
the flat state, a large oscillation around both buckled states, or coexistence (bistability). The physical points of
codim2.py (grid of leading eigenvalues, mu1 = -lambda1 lambda2, mu2 = lambda1 + lambda2) and the time-stepping
checks are placed in the same plane.

    python3 bt_unfolding.py [codim2 normal-form json] [out json]
'''
import json
import sys

import numpy as np
from scipy.integrate import solve_ivp


def rhs(_, y, n1, n2):
    x, v = y
    return [v, n1 * x + n2 * v - x ** 3 - x ** 2 * v]


def attractors(n1, n2, T=400.0):
    found = set()
    for x0, v0 in ((0.05, 0.0), (0.0, 0.05), (2.0, 0.0), (0.0, 2.5), (-1.0, 1.5), (3.0, 3.0)):
        s = solve_ivp(rhs, (0, T), [x0, v0], args=(n1, n2), rtol=1e-8, atol=1e-10, max_step=0.5)
        tail = s.y[:, s.t > 0.7 * T]
        xr = tail[0].max() - tail[0].min()
        if xr < 1e-3 and abs(tail[1]).max() < 1e-3:
            xm = tail[0].mean()
            found.add("flat" if abs(xm) < 1e-2 else "buckled")
        elif tail[0].min() < 0 < tail[0].max():
            # an oscillation through x = 0: around the flat state only, or around both buckled states
            found.add("large oscillation" if (n1 > 0 and tail[0].max() > np.sqrt(max(n1, 0))) else "small oscillation")
        else:
            found.add("one-sided oscillation")
    return sorted(found)


if __name__ == "__main__":
    nf = [json.load(open(p)) for p in sys.argv[1].split(",")]
    out = dict(points=[], diagram=[])
    for d in nf:
        c = d["coefficients"]["1.0"]
        a, b = c["a"], c["b"]
        tau = b / a
        alpha = np.sqrt(-a) / abs(b)
        out["points"].append(dict(chi=d["chi"], v0=d["v0"], a=a, b=b, tau=tau, alpha=alpha,
                                  zmax_q0=d.get("zmax_q0"), nu1=d["mu1"] * tau ** 2, nu2=d["mu2"] * tau))
        print(f"chi = {d['chi']}, v0 = {d['v0']}: a = {a:.3e}, b = {b:.3e}, tau = {tau:.3e}, alpha = {alpha:.3e}, "
              f"(nu1, nu2) = ({d['mu1'] * tau ** 2:+.3g}, {d['mu2'] * tau:+.3g})", flush=True)
    angles = list(np.linspace(0, 2 * np.pi, 49)[:-1]) + list(np.arctan(np.linspace(0.70, 0.85, 16)))
    for ang in angles:
        for rad in (0.3, 1.0):
            n1, n2 = rad * np.cos(ang), rad * np.sin(ang)
            att = attractors(n1, n2)
            out["diagram"].append(dict(nu1=n1, nu2=n2, attractors=att))
            print(f"nu = ({n1:+.3f}, {n2:+.3f}): {att}", flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
