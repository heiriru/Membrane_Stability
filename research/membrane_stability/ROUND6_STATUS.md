# Round 6 — status and what is left

Request ("Do now"):
1. follow the overhang into a tube;
2. a weakly nonlinear analysis at the point where protein demixing and buckling meet;
3. whether a protein wake smooths the snap;
4. a 2D slope equation.

## Done

- **Protein wake vs snap** (`flow_phi/imperfection.py`, `figures/r6_wake.png`, INTERPRETATION §24).
  - Weakly nonlinear reduction with the protein fields: λ_v, the cubic coefficient a, and the source projection H_Q.
    The continuation of steady states with the source checks the predicted folds.
  - Answer: **no smoothing**. The fold moves earlier as Q^(2/3) (Koiter); mobile proteins make the snap sharper.
- **2D slope equation** (`plate/lattice_transverse.py`, `plate/nonlinear_cell.py --angle`, `plate/slope_equation_2d.py`,
  `figures/r6_slope2d.png`, §25).
  - Anisotropic Bloch coefficients and tilted cells.
  - Terraces are stripes across the flow, transversely stable.
  - Lateral pulled front: 0.089 measured, 0.084 predicted.
- **Codimension-two analysis** (`flow_phi/codim2.py` stages grid / locate / normal_form / bt / bt_locate,
  `bt_unfolding.py`, `figures/r6_codim2.png`, §26).
  - At M = 10 the static buckling merges into the travelling protein mode (degenerate Bogdanov–Takens point,
    χ ≈ 0.06–0.09, SL ≈ 18–20).
  - Static and Hopf branches are both subcritical; the static cubic coefficient vanishes at the merger.
  - Time stepping confirms runaway.
  - Bug fixed: strided NumPy views assigned to dolfin vectors.

## Not done / partly done

- **Tube** (§27). Tools are built: harmonic log-polar reparametrization, same-triangulation and new-mesh remeshing,
  frame transfer, weak rim option, solver reset, `tube_driver.sh`. The round-5 overhang cannot be continued: Newton
  diverges in any new parametrization.
  - Next: an ALE formulation with the tangential velocity as a field from a harmonic/Winslow energy (integrated by
    parts), or remeshing early with a C¹ surface transfer.
- **Codim-2:**
  - quintic terms at the degenerate Bogdanov–Takens point (the extent of the supercritical stretch);
  - the same analysis with the force-free PI (pi_height = 2) of the Péclet sweep.
- **Wake:** a source combined with a contact slope (sign of the combined imperfection); M ≥ 10.
