# Round 3: status and what is left

**All queued runs finished and are in the repository (commit "Round 3 complete").** Item 1 below is done; items 2–3
are done except the remaining INTERPRETATION sections 16a–d (wrinkles, Bloch and amplitude equations, written in
short form in the take-home messages and sections 11–12) and a final reordering of Part II. Item 4 is the open physics.

## Done (code, results in `results/round3/`, figures `figures/r3_*.png`, `figures/gifs/gif5–7`)

- **Weakly nonlinear theory** (`flow_landau.py`):
  - perfect branch 1 − v0/v0_c = 0.0017966 A², against 0.0018 from the continuation;
  - Koiter fold law 0.231 |t|^(2/3) with nothing fitted (0.0107 / 0.0223 / 0.0497 / 0.103 against
    0.011 / 0.022 / 0.048 / 0.092);
  - 96 % of the cubic coefficient is geometric.
- **Post-snap dynamics** (`flow_snap.py`, t = −0.3, v0 = 0.92 and 0.97 v0_c):
  - slow drift past the fold, then collapse into a pit about 20 r0 deep just upstream of the PI;
  - near-vertical wall (slope 15), and the Monge description breaks down;
  - no saturated state; the continuation of the unstable branch also ends at A ≈ 21.
- **Force-free PI with a contact angle** (`vertical_force.py` with a Noether-current domain integral, checked against
  −dE/dh to 5e-5; `flow_forcefree.py`):
  - folds at 0.9816 / 0.9586 / 0.919 v0_c for t = −0.03 / −0.1 / −0.3 (exponent 0.64);
  - the membrane stays nearly flat until SL ≈ 23;
  - **the PRE's large flow deformations (IRENE square_b) come from the missing force balance.**
- **Divergence/flutter map** (`plate/study_flutter_map.py`, plate model, 8 IRENE checks agree to about 0.2 %):
  - flutter comes first for ℓ_b ≈ 3–15 at Γ = 0, ≈ 3.2–13 at Γ = 25, and ≈ 4.2–7.5 at Γ = 100;
  - Γ = 10 and Γ = 50 were still running (partial data in the repo).
- **Local analysis:** absolute instability for G = g√κ/s^1.5 < G* = 1.622 (saddle point, checked against the impulse
  response). G ≪ G* in the compressed region, but WKB is marginal at threshold.
- **Lattice of PIs** (`plate/study_lattice.py`, periodic cell plus Bloch waves; validated against Sangani–Acrivos to
  3e-4, and Bloch periodicity):
  - the threshold depends on the protein density, not on a box: U_c = 1.03 → 0.29 κ/(η r0) for φ = 8.7 % → 0.4 %;
  - a uniform force and friction (ℓ_b = 5, 20) give the same values within about 2 %;
  - the instability is **long-wave**: the effective tension of the lattice goes to zero, and the whole lattice
    undulates with a long wavelength, travelling upstream at about f0/ζ (checked: Re λ/q² changes sign).
- **Several PIs in the box** (`plate/study_multi.py`, 24 configurations):
  - SL_c varies by a factor 4 (rows across the flow are the most dangerous);
  - the total drag at threshold stays at 1.05–1.5 κ/r0 (4–6 pN);
  - the critical mode is always one in-phase collective dome.
- **Tubes** (`axisymmetric/ring_tube.py`, matches IRENE to 2e-4):
  - the force overshoots, then reaches the tether force 1.84 pN for R = 1 µm;
  - force maxima of 22 / 8.5 / 2.9 / 2.1 pN for R = 50 / 100 / 300 / 1000 nm.
- **Wrinkles** (`plate/study_wrinkles.py`):
  - a sliding membrane gives static stripes parallel to the flow (wavelength matches 2π√(2κ/|σ|));
  - strongly driven boxes go from one dome to crescent-shaped wrinkle trains near the PI (3 → 10 → 17 unstable modes).
- **Mobile curvature proteins** (`plate/analysis_mobile.py`):
  - a modulated phase appears at χ_c = εσ − 2C0√(εσ);
  - flow suppresses modulations along it, so the first instability is **static stripes parallel to the flow**.
- **Physical tables** (`plate/physical_tables.py`, single-PI part done): drag on the PI at threshold
  4 pN (σ = 1e-6 N/m) to about 40–55 pN (σ = 1e-4 N/m); velocities 0.1 to more than 1 mm/s.

## Left to do (in order)

0. (update) Done since the first version of this file: flutter map for all Γ (11 IRENE checks); lattices at three
   tensions; long-wave thresholds up to Lc = 80 (dilute limit: drag per PI → 1.0 κ/r0, the single-PI value); physical
   tables; INTERPRETATION sections 11, 12, 15. Still running when last updated: lattice_longwave for the friction /
   tension variants, and ring_force_table for R = 100.
1. **Finish the queued runs** (queue `/tmp/claude-0/runs/q4`, lost if the container is reclaimed; commands are in
   `jobs.txt` there, or rerun the scripts):
   - `plate/study_flutter_map.py --gammas 10` and `--gammas 50`;
   - IRENE checks `flow_critical.py` for Γ = 100 / 25 at ℓ_b = 14 / 3.5, 7, 14;
   - `axisymmetric/ring_force_table.py` (R = 30, 100 missing);
   - `plate/study_lattice.py --sigma0 0.025` and `--sigma0 0.00025` (cells 8, 14, 28);
   - `plate/lattice_longwave.py` on each lattice run (`--extra_cells 40,56,80`): exact long-wave thresholds, K_eff,
     drift speed, and the check that no finite-q mode comes first.
2. **Then:** `collect_round3.sh`, `plot_round3.py`, `plate/physical_tables.py` (lattice and ring tables),
   `make_gifs_round3.py`.
3. **Write-up:**
   - complete `INTERPRETATION.md` Part II: sections 11 (flutter map), 12 (lattice), 15 (tables), 16a–d (patterns,
     amplitude equations), reordered, with new take-home messages;
   - README: a new "Round 3" section with tables and figures;
   - PAPER_PLAN: tick the items.
4. **Open physics after this round:**
   - **post-snap state beyond Monge:** a parametric or axisymmetric description, or a force-free PI in the time
     stepping;
   - **nonlinear lattice amplitude equation:** the cubic coefficient of the long-wave slope equation (convective
     Cahn–Hilliard type: ζ H_t = σ_eff H_xx − K_eff H_xxxx − ζ c H_x + g (H_x³)_x), which needs a nonlinear periodic
     model;
   - flutter map at other domain sizes, and a WKB analysis far above threshold;
   - mobile proteins in IRENE (extra conserved field), and coupling to the flow around anchored PIs;
   - **discuss with M. Castellana:** the square_b force balance (now shown to change the PRE's Fig. 11B), the
     tangential boundary conditions, and the force definition.
