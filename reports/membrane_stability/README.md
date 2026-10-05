# Membrane stability research report

Publication-style synthesis of Copy_Stage at commit
`45d54f29909d4026ece901c28287b23104190b1d` (reviewed 4 October 2026).

`main.tex` provides ten pages of main text and a separate reference page.
`supplement.tex` provides detailed derivations, verification, qualifications,
provenance, and the complete visual archive. Author names and affiliations are
intentionally absent: authorship and scope have not yet been agreed.

## Build

Run `bash build.sh` from any directory. The script uses Tectonic when available,
or three passes of pdfLaTeX. Outputs go to `build/`; set `REPORT_OUTPUT_DIR` to
choose another directory. Set `TECTONIC_BIN` to an executable path if needed.
A first Tectonic build may download its TeX bundle. Python 3 is needed only to
regenerate the figure atlas (`python3 build_archive.py`), not to compile the
included generated sources. No FEniCS installation or numerical simulation is
needed for document compilation.

## Figure archive

The manifest lists 84 tracked visual assets first added to Git during 2026:
67 static PNGs and 17 GIFs. Dates are first-addition dates, not checkout
modification times or simulation dates. Ten older assets are excluded.

Nine PNGs appear in the main text; 58 other PNGs and 17 four-frame storyboards
appear in the supplementary atlas. Larger supplementary copies of all nine
main figures make detailed panels easier to read. Original GIFs are included
under `figures/`, with stills under `animation_stills/`. Source paths, dates,
original dimensions, frame counts, placements, and SHA-256 hashes are recorded
in `figure_manifest.json`. Copies preserve original image bytes; storyboard
images are presentation derivatives. No numerical curve was redrawn or changed.

## Scientific status

This is a reviewed research report, not a submitted journal manuscript. It
separates original boundary-condition diagnostics from corrected predictions,
force-free from clamped protein setups, finite clusters from dilute lattices,
and computed reduced-model results from unresolved full-model claims.

Before submission: agree scope/authorship with the supervisor and M. Castellana;
review the force and boundary diagnostics jointly; complete the corrected-
condition penalty/domain robustness matrix and runtime provenance. Mobile-
protein endpoints, the precise double-zero/degenerate unfolding, and flow-driven
tube growth remain unresolved. See the supplement and `EVIDENCE_REVIEW.md`.

The archived results were inspected, not rerun for this writing task. References
use verified preprint identifiers where the repository's journal metadata could
not be independently confirmed. No journal submission, message, or repository
push was performed.
