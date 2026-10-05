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
start at `research/membrane_stability/` unless stated otherwise.

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

The main text is arranged around the physical mechanism, then the control
ensemble, geometric thresholds, snap, overhang, flutter, mobile proteins, and
collective patterns. Equations use consistent force, curvature, residual and
normalization conventions. The supplement separates derivation, validation,
limitations, and archival illustrations. Superseded diagnostic figures remain
in the archive with explanatory captions rather than being silently removed.

The manuscript is written in paper style and the reviewed prose is coherent.
Publication readiness cannot be certified by editing alone: author agreement,
original-author review of corrections, and outstanding numerical robustness
remain conditions for submission. These are recorded in the report itself.
