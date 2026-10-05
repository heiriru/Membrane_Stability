# Round 4 — status and what is left

Request:
1. follow the post-snap state beyond Monge, with a parametric shape or a force-free protein in the time stepping;
2. compute the nonlinear coefficient of the lattice's long-wave equation;
3. add mobile curvature proteins to IRENE, coupled to the flow around anchored proteins.

## Done

1. **Force-free PI in the post-snap time stepping** (`irene_addon/stability/flow/flow_snap_ff.py`; runs at 0.93 and
   0.97 v0_c, t = −0.3, mesh A).
   - The PI is lifted to h ≈ 7–8 r0 while the pit deepens to −16.5 r0.
   - The runs end at slope ≈ 12 (Monge breakdown, Newton failure).
   - INTERPRETATION section 17.
2. **Nonlinear lattice coefficient** (`plate/nonlinear_cell.py`; Lc = 6, 8, 10, 14, 20; drives 0, D_lw/2, D_lw).
   - g < 0 (subcritical) for φ ≤ 5 %, g > 0 (saturation at |u| ≈ 0.1) for φ = 9 %.
   - Validated at rest (energy = flux = Bloch).
   - INTERPRETATION section 18.
3. **Mobile curvature proteins in IRENE** (`irene_addon/stability/flow_phi/`: 8-field function space, protein forms
   in `flow_problem.py`, `phi_stability.py`).
   - Validated at rest against the infinite-membrane threshold (−0.0506 vs −0.0475), and at χ = 100 against the
     no-protein buckling threshold (27.45 vs 27.50).
   - χ_c(v0): −0.051, −0.083, −0.124 at SL = 0, 2, 5; at SL = 10 only an outflow boundary-layer mode is left; stable
     down to χ = −3.4 at SL = 20. Modes: stripes along the flow.
   - v0_c(χ): SL_c = 26.96, 25.38, 25.11, 24.96 for χ = 10, 1, 0, −0.045 (advection-limited softening).
   - INTERPRETATION section 19.

## Not done / next

- **Parametric (non-Monge) surface for the post-snap state.** It needs an ALE parametrization X(s, θ, t) with the
  in-plane flow, i.e. a new gauge in IRENE's geometry module, with remeshing or an arbitrary Lagrangian–Eulerian
  tangential velocity. An axisymmetric reduction is not possible: the pit is upstream of the PI, so the problem is not
  axisymmetric.
- **Lattice cell with in-plane readjustment and drift.** A dynamic cell would remove the small negative σ_eff of the
  static cell at the Bloch threshold. Also: the crossover density between 5 % and 9 %, and the quintic coefficient
  for the dilute lattices (hysteretic terraces).
- **Simulation of the convective Cahn–Hilliard slope equation** with the computed σ_eff, K_eff, c and g (terrace
  coarsening for dense lattices).
- **Proteins:**
  - nonlinear time stepping of IRENE + φ (protein patterns in the flow);
  - force-free PI with proteins;
  - protein sources at the PI (proteins recruited by the anchored protein);
  - an outflow condition for φ that lets proteins leave freely (removes the boundary-layer mode at large v0).
