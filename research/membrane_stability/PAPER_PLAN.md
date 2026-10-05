# Paper plan: linear stability of lipid membranes with protein inclusions (IRENE + FE eigenvalue analysis)

Working title: *Stability of membrane deformations induced by protein inclusions in the large-deformation regime*

Starting point: Ferraro & Castellana, PRE 114, 014405 (2026) — non-monotonic force F(h) on a protein inclusion (PI),
"a maximum followed by a decrease may suggest the onset of instabilities"; "a full stability analysis of the
governing equations may reveal the existence of flow-driven instabilities at larger velocities" (future work).
IRENE (Wörthmüller, Ferraro, Sens, Castellana 2025) solves the steady states; it has no stability tool.

## 0. Coordination (before anything else)
- [ ] Agree scope and authorship with the supervisor and M. Castellana (this is listed as future work of their
      group, and it overlaps with the internship "instability thresholds leading to membrane buckling").
- [ ] Decide the target journal (PRE / Soft Matter / PRL-letter if the result is striking) and the scope of the paper
      (points 2 + 3 below are one paper; 4 could be a second one).

## 1. Method in IRENE (tool)
- [x] Generic eigenvalue solver from an FE residual (`linear_stability.py`), validated on the cylinder wake
      (Re_c = 46.3 vs 46.7, DNS cross-check).
- [ ] Port to IRENE's no-flow membrane problem: Jacobian of IRENE's residual (F_z, F_omega, F_mu + penalty terms),
      mass/metric form for the shape z, homogeneous BCs of the perturbations.
- [ ] Two eigenvalue problems, both needed in the paper:
      (a) energetic stability: second variation (Hessian) of the Helfrich energy, B = L2 metric on z
          (eigenvalue > 0 = stable), independent of the dynamics;
      (b) dynamic stability / relaxation rates with IRENE's dynamics (overdamped, rho -> 0; B on z only, or with
          viscosity when v, w are included).
- [ ] Verification tests: flat membrane (h = 0) spectrum vs the analytical Bessel-function eigenvalues of the
      linearized shape operator on the annulus; symmetry of the Hessian; mesh convergence of the leading
      eigenvalues; independence of the penalty parameter alpha.

## 2. Result 1: force-displacement stability, pinned ring (this repo, first)
- [x] Port done and verified (flat-membrane spectrum, error 2e-4); sweeps for R = 5, 10, 30 (tan alpha = 0.5) done.
- [x] Force checked against exact solutions (linear theory, catenoid): virtual-work force converges, line-force
      integral does not (check_force_exact.py).
- [x] Mesh convergence of F(h) and lambda (R = 5): <= 3e-4 / 2.5e-3.
- [x] Contact-angle dependence (R = 10, tan alpha = 0, 0.5, 1).
- [ ] **Force computation — discuss with M. Castellana first.** The virtual-work force -dE/dh is mesh-converged,
      the line-force integral (PRE Eqs. 17-18 / IRENE's dFdl_sigma_kappa_3d) is not, and it misses the boundary term
      2 kappa H n.grad(psi) (moment) that does work under a vertical translation (check_force_energy.py). Decide which
      force Fig. 5 used and re-derive the line force including the moment term (analytic check: small-deformation
      limit, where F is linear in h).
- [x] Reproduce Fig. 5 of the PRE (F(h) for R = 5, 10, 30, 100 r0) with IRENE, in pN / nm (figures/ring_force_physical_units.png).
- [x] Leading eigenvalues along F(h), per azimuthal number m = 0, 1, 2, ... (full 2D ring mesh).
- [x] (Schur complement + virtual-work force; an explicit force-control eigenproblem needs the vertical force balance on the rim, see README) Displacement control (height h imposed) vs force control (force f imposed, h free): the m = 0 stability must
      change sign where dF/dh = 0 under force control. Check this analytically/numerically -> interpretation as
      tubule nucleation / snap-through, hysteresis.
- [x] Look for non-axisymmetric (m >= 1) instabilities at large h or large contact angle alpha: none found up to
      tan alpha = 2 within the Monge range.
- [x] Stability diagram in the (h, alpha) plane for R = 5, 10, 30 (tan alpha 0 ... 2): displacement control stable
      everywhere (m = 0 least stable, m = 1 near the branch ends for R = 5, tan alpha >= 1.5); force-control boundary =
      force extrema.
- [x] Physical units: critical displacements/forces in nm and pN; comparison with tether-pulling forces
      (f ~ 2 pi sqrt(2 kappa sigma), Derenyi-Julicher-Prost 2002).
- [x] (round 3: axisymmetric tube solver) Limit of the Monge gauge: overhangs cannot be represented; report the onset only, or use an axisymmetric
      arc-length ODE check for the post-instability branch.

## 3. Result 2: flow-driven instability around a PI (square domain with flow)
- [x] Flow version of the tool (flow/flow_problem.py) verified at rest against the no-flow spectrum (1.5 %), and its
      overdamped version (rho = 0, normal friction zeta) against the inertial one (1e-9).
- [x] Found: IRENE's open tangential BCs make the linearized dynamics ill-posed (spurious growth at rest); free-slip /
      traction-free variant is well posed. **Discuss with M. Castellana.**
- [x] (superseded: IRENE PI condition) PRE setup (square_b, L = 100 r0, sigma0 = 0.0025, no slip on the PI, rho = 0): **flow-driven buckling** at
      SL_c ~ 16 for a PI with zero contact angle (drag on the PI -> tension drops upstream -> compression -> dome mode).
      Robust to penalty alpha and wall BC; meshes A/B/C.
- [x] (superseded) Threshold vs tension: SL_c = 12.1 + 0.174 Gamma (Gamma = sigma0 L^2/kappa = 0 ... 100).
- [x] (superseded: the bifurcation is subcritical, see below) Contact angle t = -0.3: imperfect bifurcation; the PRE's "flow-induced deformation above v*" is the unfolded
      buckling mode (sharp threshold recovered for t -> 0).
- [x] Cross-check with nonlinear time stepping of IRENE's equations (BDF2): growth/decay rates within 1.3e-4.
- [x] IRENE example (square_a, rho = 1): inertial divergence (bending-wave collision) + viscous mechanism; thresholds
      depend on the slip penalty alpha -> not quantitative (needs Nitsche slip BC). Small-v0 oscillatory growth is
      numerical (~h^3).
- [x] **IRENE's square_b PI has no vertical force condition** (z free on the rim, but the rim rows of F_w keep the
      boundary term of the integration by parts): the PI carries a spurious vertical force in the critical mode, which
      lowers the threshold (check_pi_force.py). With a clamped PI or a rigid force-free PI the full FE and an
      independent compressed-plate model agree to 0.2 %: SL_c = 27.5 (mesh A), not 16.5. **Discuss with M. Castellana**
      (also affects the PRE's deformations with flow). Studies below redone with the physical PI conditions.
- [x] Physical PI conditions (clamped / rigid force-free) and **compressed-plate model** (exact reduction at a flat
      base state; agrees with the full FE to 0.1–0.4 %): SL_c = 27.4 (Γ = 25, mesh-converged); mechanism = tension
      drop upstream, critical mode upstream of the PI.
- [x] Domain: SL_c depends on L/r0 only, logarithmically (Stokes paradox); strongly on the width and on how the outer
      boundary is held (12 … 31). In-plane friction (Brinkman) lowers the threshold and does not remove the box
      dependence (tension disturbance is harmonic); with friction a **flutter (Hopf) instability** appears (ℓ_b = 5–10).
- [x] Bifurcation diagram: **subcritical** pitchfork (v0/v0_c = 1 − 0.0018 A²), folds before v0_c for t ≠ 0 with
      Koiter's 2/3 law (fitted 0.63): snap-through and hysteresis instead of smooth growth.
- [x] Adjoint mode and sensitivity to steady forcing (checked to 2e-7).
- [x] (round 3: lattice of PIs, threshold vs density) Geometry: a formulation without the box dependence (e.g. periodic channel, or the full bulk fluid instead of a
      local friction); the threshold must otherwise be stated for a given geometry.
- [x] (round 3: collapse into a deep pit, Monge breaks down) Post-snap state: beyond the Monge gauge (overhangs), or time stepping from the fold to see where the membrane goes.
- [x] (round 3: map in (ell_b, Gamma), 11 IRENE checks) Flutter with friction: frequency and mode, map of divergence vs flutter in (ℓ_b, Γ).
- [ ] Nonlinear force-free PI (t ≠ 0) needs the vertical force balance on the rim (same issue as the line force).
- [ ] Nitsche slip condition on the PI (square_a) if the slip case is needed.

## 4. Possible extensions (second paper / internship)
- [x] (round 3: pairs, rows, cluster; total drag criterion) Two PIs: stability of pair configurations (same/opposite orientation).
- [ ] Membrane + bulk fluid (actin) coupling (internship): buckling thresholds via the same eigenvalue tool.

## 5. Numerical quality (referee-proof)
- [x] Mesh convergence of critical values (3 meshes: flow thresholds converge to 5e-4; plate model 1e-4), eigen-solver residuals recorded (critical_search.csv).
- [ ] Sensitivity to domain size R and to penalty parameters.
- [ ] Comparison with small-deformation analytics in the linear regime.
- [ ] Reproducibility: scripts + data in IRENE (pull request to Michele-s-team/irene), one command per figure.

## 5b. Towards pattern formation (see INTERPRETATION.md §6)
- [x] (round 3) Plate-model test: many modes at larger L / stronger compression (domes -> wrinkle trains?).
- [x] (round 3) Periodic cell with one PI + Bloch-Floquet analysis (band structure lambda(q)); also removes the box dependence.
- [x] (round 3) Local dispersion relation with friction (follower load) and convective/absolute criterion for the flutter.
- [x] (round 3) Amplitude equations from the direct/adjoint modes (Landau coefficient vs continuation -0.0018 A^2).
- [x] (round 4) Mobile curvature-coupled proteins advected by the flow (extra conserved field in IRENE): flow
      suppresses demixing (stripes along the flow); buckling threshold lowered by ≤ 9 %.
- [x] (round 4) Nonlinear coefficient of the lattice long-wave equation: g < 0 dilute (collapse), g > 0 at φ ≈ 9 %
      (terraces).
- [x] (round 4) Post-snap with a force-free PI: PI lifted, S-shaped fold, Monge breakdown at slope ≈ 12.
- [x] (round 5) Parametric surface beyond Monge: the fold turns over into an overhang under the lifted PI.
- [x] (round 5) Slope-equation simulations: terraces, hysteresis, convective threshold (ε_a ≈ 1); tricritical
      density φ* ≈ 0.075.
- [x] (round 5) Nonlinear time stepping of IRENE + proteins (consistent forms, absorbing outflow), 3D animations.
- [x] (round 5) Nonlinear flutter: supercritical Hopf, saturated wave emission, no hysteresis.
- [x] (round 5) Péclet sweep: proteins and buckling merge at low Péclet; travelling modes at M ≈ 10.
- [x] (round 6) Weakly nonlinear analysis where buckling and the travelling protein mode meet (M = 10): a
      degenerate Bogdanov–Takens merger; both primary instabilities subcritical (time stepping: runaway).
- [x] (round 6) Protein wake as an imperfection: the snap is advanced (Koiter Q^(2/3)), not smoothed.
- [x] (round 6) 2D slope equation with anisotropic coefficients: terraces across the flow, no zigzag, lateral
      pulled-front invasion.
- [ ] Tube growth beyond the overhang: remeshing tools built, but needs an ALE formulation with built-in mesh
      smoothing (or early remeshing with a C1 transfer); see INTERPRETATION §27.

## 6. Writing
- [ ] Introduction + state of the art (PI-induced deformations, Derenyi-Julicher-Prost tethers, flow-driven
      instabilities: Sahu et al. 2020, Tchoufag et al. 2022, Al-Izzi-Sens-Turner 2020; global stability methods).
- [ ] Methods: FE formulation (IRENE), linearization, eigenvalue problem, verification.
- [ ] Results 1 and 2, figures: F(h) coloured by stability; spectra vs h; eigenmodes (m = 0, 1, 2); stability
      diagram; flow case sigma(SL).
- [ ] Discussion: biological relevance (tubulation, protein sorting, trafficking), limits (Monge gauge, no bulk fluid).
- [ ] Supplementary: verification tests, convergence, code availability.
