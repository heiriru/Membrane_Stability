'''
Thresholds in physical units (writes results/round3/physical_tables.md and .json).

Conversion: r0 = PI radius, kappa = bending rigidity, sigma0 = tension, eta = 2D membrane viscosity;
force unit kappa / r0, velocity unit kappa / (eta r0); Gamma = sigma0 L^2 / kappa.
    1. single PI in a membrane patch of size L (PRE box, plate model, force-free PI): SL_c(Gamma) for Gamma up to 1e4
       (computed here) -> v_c = SL_c kappa / (eta L) and the drag on the PI at the threshold F_c = v_c x drag(v0 = 1);
    2. lattice of anchored PIs (study_lattice.py / lattice_longwave.py): mean velocity U_c and drag per PI at the
       threshold vs area fraction phi;
    3. PI pulled out of a ring (ring_force_table.py): force maximum (snap-through under force control) and the height
       where it is reached, vs ring size, kappa and sigma.
Physical parameters: kappa = 10, 20, 50 kT (T = 300 K), sigma = 1e-6, 1e-5, 1e-4 N/m, eta = 1e-8 Pa s m (also
1e-9), r0 = 10 nm (5 nm for the flow table), L = 0.5, 1, 2 um.

    python3 physical_tables.py
'''
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
R3 = os.path.join(HERE, "..", "results", "round3")
kT = 1.380649e-23 * 300
out_md = []


def J(p):
    p = os.path.join(R3, p)
    return json.load(open(p)) if os.path.exists(p) else None


# 1. single PI: SL_c(Gamma) and drag at threshold, box L = 100 r0 (plate model, rigid force-free PI)
cache = os.path.join(R3, "box_gamma_scan.json")
if not os.path.exists(cache):
    import plate_lib as pl
    L = 100.0
    geo = pl.Geometry(pl.make_mesh("/tmp/claude-0/runs/box_gamma_mesh", L, L, [(50, 50)], h_min=0.15, h_max=3.0))
    base = pl.BaseFlow(geo)
    plate = pl.Plate(base, pi_bc="rigid")
    scan = []
    for G in (0, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000):
        c = plate.drive_threshold(G / L ** 2)
        scan.append(dict(Gamma=G, SL_c=c.eigenvalue.real * L, drag_c=c.eigenvalue.real * base.F_drag))
        print(scan[-1], flush=True)
    json.dump(dict(L_over_r0=L, drag_per_v0=base.F_drag, scan=scan), open(cache, "w"), indent=1)
box = json.load(open(cache))
Gs = np.array([r["Gamma"] for r in box["scan"]])
SLs = np.array([r["SL_c"] for r in box["scan"]])
out_md.append("## 1. Single PI in a membrane patch (box L = 100 r0, force-free PI, plate model)\n")
out_md.append("| Gamma = sigma0 L^2/kappa | SL_c | drag on the PI at threshold [kappa/r0] |\n|---|---|---|")
for r in box["scan"]:
    out_md.append(f"| {r['Gamma']:g} | {r['SL_c']:.2f} | {r['drag_c']:.3f} |")
out_md.append("\nIn physical units (L/r0 = 100 fixed; at other L/r0 add about +4.8 to SL_c per doubling at fixed Gamma):\n")
out_md.append("| r0 | L | kappa | sigma0 [N/m] | Gamma | v_c (eta = 1e-8 Pa s m) [um/s] | v_c (eta = 1e-9) [um/s] |"
              " drag at threshold [pN] |\n|---|---|---|---|---|---|---|---|")
rows1 = []
for r0 in (10e-9, 5e-9):
    L = 100 * r0
    for kap in (10, 20, 50):
        K = kap * kT
        for s in (1e-6, 1e-5, 1e-4):
            G = s * L ** 2 / K
            if G > Gs.max():
                continue
            SL = float(np.interp(np.log1p(G), np.log1p(Gs), SLs))
            drag = float(np.interp(np.log1p(G), np.log1p(Gs), [r["drag_c"] for r in box["scan"]]))
            v8 = SL * K / (1e-8 * L) * 1e6
            v9 = SL * K / (1e-9 * L) * 1e6
            Fc = drag * K / r0 * 1e12
            rows1.append(dict(r0_nm=r0 * 1e9, L_um=L * 1e6, kappa_kT=kap, sigma=s, Gamma=G, SL_c=SL, v_c_eta1e8=v8,
                              v_c_eta1e9=v9, drag_pN=Fc))
            out_md.append(f"| {r0 * 1e9:g} nm | {L * 1e6:g} um | {kap} kT | {s:g} | {G:.1f} | {v8:.0f} | {v9:.0f} |"
                          f" {Fc:.1f} |")

# 2. lattice
out_md.append("\n## 2. Lattice of anchored PIs (uniform tangential force; r0 = 10 nm, kappa = 10 kT, eta = 1e-8 Pa s m)\n")
out_md.append("| sigma0 [N/m] | area fraction phi | cell / r0 | U_c [um/s] | drag per PI at threshold [pN] | source |"
              "\n|---|---|---|---|---|---|")
K, r0 = 10 * kT, 10e-9
vu, fu = K / (1e-8 * r0) * 1e6, K / r0 * 1e12
rows2 = []
for name, s_phys in (("lat_force_s1e-7", 1e-7), ("lat_force", 1e-6), ("lat_force_s1e-5", 1e-5)):
    lw = J(os.path.join(name, "longwave.json")) or []
    grid = {r["Lc"]: r for r in (J(os.path.join(name, "lattice.json")) or [])}
    for r in sorted(lw, key=lambda r: r["Lc"]):
        rows2.append(dict(sigma=s_phys, phi=r["phi"], Lc=r["Lc"], U_c=r["U_lw"] * vu, drag=r["drag_lw"] * fu))
        out_md.append(f"| {s_phys:g} | {r['phi']:.4f} | {r['Lc']:g} | {r['U_lw'] * vu:.0f} | {r['drag_lw'] * fu:.1f} |"
                      f" long-wave threshold |")
    if not lw:
        for Lc, r in sorted(grid.items()):
            out_md.append(f"| {s_phys:g} | {r['phi']:.4f} | {Lc:g} | {r['U_c'] * vu:.0f} | {r['drag_c'] * fu:.1f} |"
                          f" Bloch grid |")

# 3. ring
tab = J("ring_table/ring_table.json") or []
out_md.append("\n## 3. PI pulled out of a ring (tan alpha = 0.5, r0 = 10 nm): force maximum and tether force\n")
out_md.append("| ring radius | kappa | sigma [N/m] | ell = sqrt(kappa/sigma) / r0 | F_max [pN] | h at F_max [nm] |"
              " tether force [pN] | F_max / tether |\n|---|---|---|---|---|---|---|---|")
rows3 = []
ells = np.array(sorted({r["ell"] for r in tab})) if tab else np.array([])
for R in (5, 10, 30, 100):
    for kap in (10, 20, 50):
        K = kap * kT
        for s in (1e-6, 1e-5, 1e-4):
            ell = np.sqrt(K / s) / r0
            if not len(ells):
                continue
            j = int(np.argmin(np.abs(np.log(ells / ell))))
            if abs(np.log(ells[j] / ell)) > 0.12:
                continue
            row = [r for r in tab if r["R"] == R and r["ell"] == ells[j]]
            if not row:
                continue
            row = row[0]
            fu = K / r0 * 1e12
            rows3.append(dict(R_nm=R * 10, kappa_kT=kap, sigma=s, ell=ells[j], F_max_pN=row["F_max"] * fu,
                              h_nm=row["h_at_F_max"] * 10, tether_pN=row["tether_force"] * fu,
                              reached=row["maximum_reached"]))
            out_md.append(f"| {R * 10:g} nm | {kap} kT | {s:g} | {ells[j]:g} | {row['F_max'] * fu:.1f}"
                          f"{'' if row['maximum_reached'] else ' (end of range)'} | {row['h_at_F_max'] * 10:.0f} |"
                          f" {row['tether_force'] * fu:.2f} | {row['F_max_over_f0']:.2f} |")
open(os.path.join(R3, "physical_tables.md"), "w").write("# Thresholds in physical units\n\n" + "\n".join(out_md) + "\n")
json.dump(dict(single=rows1, lattice=rows2, ring=rows3), open(os.path.join(R3, "physical_tables.json"), "w"),
          indent=1)
print("\n".join(out_md))
