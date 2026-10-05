# Scientific and editorial evidence review

Review date: 4 October 2026. Source snapshot:
`45d54f29909d4026ece901c28287b23104190b1d`.

## Sources and scope

The synthesis reads INTERPRETATION.md, PAPER_PLAN.md, the main README and
ROUND3–ROUND6_STATUS.md; archived numerical summaries and selected underlying
outputs; the generic eigensolver and vertical-force forms; ring, flow, plate,
nonlinear coefficient, parametric-fold, protein-transport, codimension-two,
and two-dimensional slope-equation implementations. The cylinder benchmark
README and visual outputs were also inspected. Relative research paths below
start at `Ruben/membrane_stability/` unless stated otherwise.

The report does not claim to rerun expensive numerical experiments. Independent
verification refers to experiments already archived in the repository.

## Checks that materially affected the text

| Topic | Evidence and interpretation used in the report |
| --- | --- |
| Dynamics sign | `irene_addon/modules/stability/linear_stability.py` uses A=-D R. `stability/no_flow/ring_problem.py` sets R=-F_IRENE and uses the half-normal-force mass. A bare Jacobian without the residual convention can reverse the sign. |
| Physical protein condition | Flow implementation and flow2 summaries give original 16.168, force-free 27.449, clamped 27.664 on mesh C. A free rim is not automatically a rigid force-free protein. Joint review with the original authors is explicitly required. |
| Ring force | Energy differentiation and `vertical_force.py` establish force orientation. Force control uses P'=E*''=-Fm'. Missing boundary moment and unresolved weak-rim derivatives are distinct errors. Height stability is limited to sampled branches. |
| Plate and friction | `plate/plate_lib.py` and flow plate reduction retain T:Hess(z). With div T+f=0, this is div(T grad z)+f.grad z. Omitting the follower term would incorrectly suggest self-adjoint flutter dynamics. |
| Drag scale | Round 3 finite-cluster calculations use summed drag; dilute periodic lattices use per-disc drag. Dense lattices and higher tension do not satisfy an exact universal 4 pN constant. |
| Weakly nonlinear snap | `flow_landau.py` and Round 3 coefficients give clamped vc=0.2773155, lambda_v=3.3126157e-4, N3=1.6504712e-7, Hs=-9.2488061e-5. The Koiter prediction has no fitted exponent or fold coefficient. It is asymptotic, not exact at finite slope. |
| Hysteresis | Cubic subcriticality establishes a nearby barrier and snap, not an independently resolved return branch. Single-protein post-snap flow hysteresis remains incomplete; ring/tube and intermediate-density reduced lattice loops are distinguished. |
| Overhang | Round 5 parametric fold has negative vertical normal but severe mesh stretch. Round 6 remeshing does not establish sustained flow-driven tube growth. No-flow axisymmetric tube branches are a separate calculation. |
| Flutter | Archived screened-flow spectra and BDF2 runs show saturation and decay below onset in the sampled case. This is not a complete Floquet or time-step/domain robustness study. The physical period depends on normal friction. |
| Mobile proteins | `flow_phi/flow_problem.py` supplies the consistent nonlinear free energy and mesh-frame transport. Phi is a signed deviation; a negative pit value cannot be called absolute enrichment without a convention/calibration. |
| Mobility versus merger | Mobility sweeps are force-free; codimension-two, nonlinear protein dynamics and rim-source continuation are clamped. Their thresholds are not pooled. No-flux and passive-sponge artifacts are identified. |
| Double-zero candidate | `flow_phi/codim2.py`, `bt_unfolding.py` and Round 6 summaries support a BT-type candidate, not an exactly located or fully classified point. Tiny cubic sign changes require quintic and mesh/parameter checks. |
| Wake sharpness | Normalized backward coefficient increases from 0.0017966 to 0.0029695 (~65%); raw cubic increases from 1.65047e-7 to 1.99021e-7 (~21%). Continued Q=.03 fold shift is .0814844. |
| Lattice fitting | Small-slope one-dimensional cubic changes sign near 7.5% area fraction. The broader two-dimensional fit has a negative cubic and large positive quintic at 8.7%; finite-amplitude terraces are demonstrated but do not independently certify the same onset criticality. |
| Finite patch | The reduced absolute threshold is about twice the periodic drive, but its intrinsic wavelength scale approaches/below the lattice spacing. A microscopic finite-patch confirmation is still needed. |
| Units | kappa/r0=4.14 pN, kappa/(eta r0)=414 micrometres/s and kappa/(eta L)=4.14 micrometres/s for the specified reference. SL=27.45 maps to 113.6 micrometres/s, not an established weak-flow cellular threshold. |

## Editorial standard and submission limits

The integrated report follows governing equations and verification with force
balance, static control ensembles, corrected flow onset, nonlinear snap,
collapse and overhang, flutter, mobile composition, and collective patterns.
Equations retain the audited force, curvature, residual, and normalization
conventions. Earlier boundary-condition diagnostics are identified explicitly
so their thresholds are not confused with corrected physical predictions.

The manuscript is written in paper style and the reviewed prose is coherent.
Publication readiness cannot be certified by editing alone: author agreement,
original-author review of corrections, and outstanding numerical robustness
remain conditions for submission. The numerical limitations remain explicit in the report; collaborative
publication decisions are recorded in the project checklist.

## Mathematical explanation expansion

The FEM explanation distinguishes the full mixed IRENE formulation from the
illustrative conservative flat-plate weak equations. New equations derive
existing interpretations: modal error measures, virtual reactions, relaxed
height stiffness, cylinder tether energy, rigid-rim constraints, compression
bands, parameter conversions, stress-integrated drag, the local amplitude
barrier, implicit force-free stepping, fold bottlenecks, parametric geometry,
Hopf diagnostics, dissipation and composition budgets, two-mode degeneracy,
source-induced folds, Bloch coefficients, slope dynamics, transverse
restoring stiffness, and convective/absolute growth. They introduce no new
simulation outputs or fitted physical constants. Local, quasistatic, and
homogenized derivations state their approximation and boundary assumptions.

All existing scientific content, 67 figure calls/captions, and 44 displayed
equation blocks were preserved. The report contains no account of
legacy teaching or solver work; the IRENE bibliography entry retains its
publication year.

## Boundary reduction and spectral-search derivation

Section 2.5 was checked against `linear_stability.py`, including homogeneous
Dirichlet row treatment, `tie_dofs`, the real-target SLEPc path, the complex
MUMPS block resolvent with SciPy/ARPACK, and the cylinder multi-shift scan.
The derivation distinguishes essential conditions from natural boundary work,
tied trial/test spaces from local free-rim conditions, and finite physical
rates from infinite generalized eigenvalues. It requires a regular pencil
and an invertible shifted matrix for the stated spectral transformation.

The real-block signs, eigenvalue transformation, and summed rigid-group row
were verified on small algebraic examples. These are formula checks, not new
membrane or cylinder simulations. Proposed overlap and operator-scaled error
formulas are identified as diagnostics; they do not assert that every archived
run already performs those checks. A finite shift scan is described as evidence
of coverage in its tested window, not a proof of complete spectral stability.
Only Section 2.5 changed in the scientific source during this revision.

## Cylinder threshold comparison table

The former cylinder classification graphic was replaced by a compact threshold
table. Coarse/medium/fine values are 46.4596, 46.3460, and 46.3239 from the
archived critical-Reynolds files already used in Section 3.3. The literature
comparison is approximately 46.7 from the existing wake reference. Relative
differences use the fine-mesh numerical value as their baseline and are
0.293%, 0.048%, 0.000%, and approximately 0.812%. DNS outcomes at Reynolds
numbers 40, 60, and 100 remain in the synthesis text; they do not locate an
independent DNS critical Reynolds number. The other 66 figure assets and
all displayed equations are unchanged.

## Ring control-ensemble comparison table

The ring classification graphic was replaced by a five-row table using
`results/summary.json`. It retains the figure's five configurations and adds
the contact slopes that its repeated radius labels did not identify. The table
shows the stored height windows, largest fixed-height growth rates, and ranges
of consecutively sampled force-unstable heights. Group endpoints and four-
significant-digit growth rates were independently checked against the JSON.
The force criterion is the virtual-work derivative F_m'(h)>0. Endpoints are
sampled heights spaced by 0.25 r0, not interpolated/refined neutral boundaries.
The empty R/r0=30 force-unstable list means no such sample was found in the
recorded window, not that every possible height is stable. Other scientific
text, displayed equations, and the 65 remaining figure calls are unchanged.

## Figure-size and layout revision

The 25 requested display changes alter figure sizes and arrangements only.
Four figures use lossless LaTeX panel viewports; two tightly packed field grids
use landscape pages. No plot was regenerated, resampled, or edited, and no new
numerical inference was added. All 65 image hashes match the audited manifest.
Scientific text, displayed equations, captions, labels, and references remain
unchanged; two layout-only landscape wrappers were added to the main source.

## Portrait field-grid revision

The two landscape figures were replaced with portrait layouts rendered from
`flow_pre_t0_meshA/mechanism_at_threshold.npz` and
`flow_pre_t-0.3_meshA/leading_modes.npz`. Rendering follows the archived plotting
script's triangulation, fields, limits, colour maps, and Gouraud shading. Axis
and colour-bar tick values were read after the original grid's aspect adjustment
and retained. Wrapped titles keep the original words. Arrays and research PNGs
are unchanged. Separate display assets and their data checksums are recorded in
the manifest, with the reproducible layout script included in the source ZIP.
No report prose, equation, caption, figure label, or reference was changed.
