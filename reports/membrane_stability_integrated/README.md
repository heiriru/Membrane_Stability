# Membrane stability research report

This continuous research report describes Copy_Stage at source snapshot
`45d54f29909d4026ece901c28287b23104190b1d` (4 October 2026).

## Contents

The report contains 22 thematic sections and 65 static scientific figures.
The governing equations, methods, verification, physical mechanisms, nonlinear
results, and limitations are integrated in one document. The detailed account
of the stability method derives the generalized eigenproblem, explains the
mixed constraints and boundary reduction, and distinguishes stationary solver
convergence from physical stability. Section 2.2 explains basis functions, nodal
degrees of freedom, weak equations, element assembly, Newton iteration, and
the construction of the generalized eigenproblem. Formula-based explanations
are integrated across all 22 sections. Section 2.5 develops the boundary-row
and rigid-rim reductions, the shift-invert mapping, sparse resolvent solves,
Krylov searches, complex shifts in real arithmetic, spectral coverage, and
original-pencil verification as six explicit steps.

The scope covers cylinder and membrane verification, vertical force balance,
static ring and tube pulling, corrected flow-driven buckling, drag and domain
scales, snap theory, post-fold dynamics, overhangs, flutter, mobile composition,
mode merger, wake forcing, and collective lattice dynamics.

## Sources and build

`integrated.tex` is the canonical editable report. It uses `preamble.tex`,
`references.tex`, and the static images in `figures/`. Run `bash build.sh` to
compile. The script uses Tectonic or three passes of pdfLaTeX and writes
`build/integrated.pdf`. Set `REPORT_OUTPUT_DIR` for another output directory,
or `TECTONIC_BIN` for an explicit compiler path. Document compilation does not
require FEniCS or a numerical solver installation.

The delivered source package contains only the canonical document, build script,
static figures, manifest, and review records. Historical authoring inputs in the
workspace are excluded from that package. The report is edited directly; the
historical composition workflow is no longer used to regenerate it.

## Data provenance and review

`figure_manifest.json` records source paths, source-addition dates, image
dimensions, SHA-256 checksums, and the placement of each static figure.
Image bytes are unchanged from the audited research outputs.
Individual size limits and panel viewport layouts are defined in `preamble.tex`.
The 19 requested oversized figures were reduced individually. Figures 21, 22,
24, and 31 use larger panel arrangements. Figures 34 and 36 use portrait grids
rendered from the same archived field arrays; their original research PNGs are
retained. The manifest records source and display assets and data checksums.
`figure_layouts.py` reproduces these two display layouts from a repository
checkout with NumPy and Matplotlib installed. The delivered display PNGs are
already included, so compiling the report requires only LaTeX.
Captions, numbering, equations, and scientific text are unchanged.
`EVIDENCE_REVIEW.md` records scientific conventions and claim boundaries;
`REVIEW.md` records editorial and layout checks for this edition.

The previous integrated PDF and full source edition are preserved under
`output/archive/membrane_stability_integrated_before_portrait_revision*` in the
workspace. Earlier main-report and supplementary sources and outputs remain
unchanged. Numerical sweeps were not rerun during document preparation.
