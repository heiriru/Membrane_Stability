'''
Force maximum (snap-through under force control) of a PI pulled out of a ring-bounded membrane, over ring sizes and
tensions, with the axisymmetric arclength shape equations of ring_tube.py (overhangs allowed). Dimensionless input:
R / r0 and ell / r0 = sqrt(kappa / sigma) / r0, contact slope tan alpha; output F_max / (kappa / r0), h at F_max,
F_max / f0 (f0 = 2 pi sqrt(2 kappa sigma), the tether force).

    python3 ring_force_table.py [out]
'''
import json
import os
import subprocess
import sys

import numpy as np

out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/claude-0/runs/ring_table"
os.makedirs(out, exist_ok=True)
here = os.path.dirname(os.path.abspath(__file__))
rows = []
for R in (5, 10, 30, 100):
    for ell in (2.0, 4.5, 6.3, 9.0, 14.1, 20.0, 28.3, 44.7):
        sigma = 1.0 / ell ** 2
        h_max = max(4 * R, 8 * ell, 40)
        d = os.path.join(out, f"R{R}_ell{ell:g}")
        subprocess.run([sys.executable, os.path.join(here, "ring_tube.py"), "--R", str(R), "--tan", "0.5",
                        "--sigma", str(sigma), "--h_max", str(h_max), "--h_min", "0", "--dh", "0.25",
                        "--dh_max", str(max(0.5, R / 40)), "--out", d], check=False, capture_output=True)
        try:
            info = json.load(open(os.path.join(d, "info.json")))
        except FileNotFoundError:
            print(f"R = {R}, ell = {ell}: failed", flush=True)
            continue
        import csv
        br = list(csv.DictReader(open(os.path.join(d, "branch.csv"))))
        h = np.array([float(r["h"]) for r in br])
        F = np.array([float(r["F"]) for r in br])
        up = h >= 0
        # the pulling force is -F (F = force of the membrane on the PI, negative when pulled up)
        i = int(np.argmax(-F[up]))
        f_max, h_at = float(-F[up][i]), float(h[up][i])
        f0 = info["tether_force"]
        reached = i < up.sum() - 1          # the maximum is interior (not the end of the computed range)
        rows.append(dict(R=R, ell=ell, sigma=sigma, F_max=f_max, h_at_F_max=h_at, F_max_over_f0=f_max / f0,
                         tether_force=f0, maximum_reached=bool(reached), h_range=info["h_range"],
                         F_end=float(-F[up][-1])))
        print(json.dumps(rows[-1]), flush=True)
        json.dump(rows, open(os.path.join(out, "ring_table.json"), "w"), indent=1)
