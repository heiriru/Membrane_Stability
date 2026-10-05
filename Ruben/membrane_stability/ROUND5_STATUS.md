# Round 5 — status and what is left

Request:
1. a 3D GIF of a simulation with mobile proteins;
2. a parametric surface to follow the fold, simulations of the slope equation, nonlinear time stepping of IRENE with
   proteins, an outflow condition that lets proteins leave freely;
3. nonlinear flutter (does a protein emit steady waves?), the Péclet number (where protein demixing and buckling
   meet), terraces and the tricritical density, protein wakes, tubes beyond the fold.

## Done

- **3D GIFs** (`make_gifs_round5.py`):
  - gif8: protein demixing at rest;
  - gif9: proteins in the flow;
  - gif10: protein source at the PI (wake);
  - gif11: saturated flutter;
  - gif12: the fold turning over, from the parametric surface.
- **Nonlinear protein time stepping** (`flow_phi/phi_dynamics.py`):
  - consistent nonlinear forms (same linearization: thresholds reproduced to all digits), quartic saturation β;
  - checkpoint/resume;
  - the free energy decreases at rest.
- **Free outflow for the proteins** (`flow_phi/flow_problem.py`, phi_sponge = 2): an absorbing sponge.
  - The first version (χ raised in the sponge) creates a chemical-potential barrier with drift ∝ M, which affects
    strong flow and large M. It is kept as phi_sponge = 1 for reference.
- **Flutter** (`flow/flow_flutter.py`): a supercritical Hopf bifurcation.
  - Limit cycles with A = 3.9 / 6.5 / ≈ 11 r0 at 1.05 / 1.10 / 1.20 v0_c.
  - No hysteresis: started from the limit cycle at 0.97 v0_c, the oscillation decays.
  - Linear rates match the eigenvalues.
- **Slope equation** (`plate/slope_equation.py`):
  - terraces, hysteresis, collapse, depending on the density;
  - absolute threshold from the Briggs–Bers pinch point (C_a = 1.622): ε_a ≈ 0.94–1.18;
  - finite patches.
- **Parametric IRENE** (`stability/parametric/`):
  - the frame E = ∇X as a mixed variable; a quasi-Monge ALE gauge; consistent reaction force; checkpoints;
  - validated against the Monge run;
  - followed the fold past vertical into an overhang (n_z = −0.67) under the lifted PI.
- **Péclet sweep** (absorbing outflow, M = 1, 10, 100, 1000; `pa_M*`).
  - At M = 1 the flow suppresses demixing.
  - At M = 10 the flow destabilizes it, with travelling modes. The buckling threshold falls towards the
    equilibrium-softening value and becomes oscillatory at χ = 0.

## Not done / next

- **Tube growth beyond the overhang:** remeshing, or the equidistributing gauge (option gauge_eq), to follow the
  invagination further.
- **Weakly nonlinear (complex) Landau coefficient of the flutter** from adjoint modes, to compare with the
  time-stepping fit. The amplitudes computed (4–11 r0) are already beyond the cubic regime.
- **Weakly nonlinear analysis at the codimension-2 point** where protein demixing and buckling meet (M ≈ 10, χ ≈ 0).
- **Protein-source wake as an imperfection of the buckling** (snap → smooth?): needs the steady state with source
  and its continuation in v0.
- **2D slope equation** with an anisotropic J (the transverse coefficients of the cell problem).
