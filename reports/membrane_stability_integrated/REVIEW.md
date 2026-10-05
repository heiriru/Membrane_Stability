# Editorial and technical review

Review date: 5 October 2026. Numerical source snapshot:
`45d54f29909d4026ece901c28287b23104190b1d`.

## Scope and retained explanations

The report contains 22 thematic sections, 65 static scientific figures,
eight tables, and 95 displayed equation blocks. The detailed FEM derivation
in Section 2.2, six-step spectral procedure in Section 2.5, and mathematical
explanations throughout the report are retained. Scientific assumptions,
force conventions, boundary conditions, and limitations remain explicit.
No numerical research simulations were rerun.

## Cylinder comparison table

The former Figure 19 classification graphic is replaced by a compact four-row
threshold table in Section 3.7. It compares coarse, medium, and fine meshes
with the existing literature value, using the fine mesh as a numerical
baseline for percentage differences. All differences were recalculated.
DNS decay at Reynolds number 40 and growth into shedding at 60 and 100 remain
in the synthesis text. DNS is not presented as a separately refined threshold.
Subsequent figures and tables are automatically renumbered, and the figure
manifest was updated to reflect all 66 current placements. The unused report
copy of the replaced graphic is excluded from the delivered source package;
the original research asset and prior editions are preserved.

Only the benchmark-synthesis subsection changed in the scientific source.
All other text, displayed equations, and remaining figure calls/captions were
verified byte-for-byte unchanged during that revision.

## Ring comparison table

The ring control-ensemble graphic is replaced by a five-row data table on
the ring-control subsection. Radii, contact slopes, height windows, largest fixed-height growth
rates, and force-unstable sample intervals match `results/summary.json`.
Sample interval endpoints and rounded growth rates were checked independently.
The caption distinguishes sampled unstable intervals from exact neutral
boundaries and limits the no-instability statement to the recorded window.
Only the requested figure block changed in the scientific source; all other
text and displayed equations are byte-for-byte unchanged against the preceding
source ZIP. Remaining figures and manifest placements were renumbered.

## Individual figure layouts

All 25 figures requested in the size review received individual layouts.
Figures 2, 3, 4, 10, 11, 18, 20, 25, 26, 27, 28, 29, 30, 33, 35, 37,
40, 52, and 60 use reduced width/height limits with preserved aspect ratios.
Figures 21 and 22 place five plots in two columns, with their fifth panel
centred; Figure 22 retains its shared legend. Figure 24 places four ring plots
in two columns with the shared legend. Figure 31 puts force panels (a,d) at
left and shape panels (b,c) at right, removing the unused blank image area.
Figures 34 and 36 now use portrait grids, replacing the preceding landscape
pages. Figure 34 arranges its four fields in two rows and two columns; Figure 36
has five rows, each pairing a steady shape with its slowest mode at the same SL.
Both were rendered directly from the archived NPZ arrays with the original
triangulation, shading, colour maps, field ranges, axis/colour-bar tick values,
and titles. Long titles wrap without changing their words. No simulations were
rerun. `figure_layouts.py` records the display-only transformation; original
research PNGs remain byte-for-byte unchanged. Figure 35 uses a slightly narrower
width to balance the adjoining portrait figures. Figure 34 and the tension plot share page 38; the paired shape/mode grid is on
page 39. Normal headers and centred footers are retained on every page, and no
rotated pages remain.

Apart from removing the two landscape wrappers and adding one page break,
`integrated.tex` is byte-for-byte
unchanged against the preceding source archive. Every scientific paragraph,
displayed equation, figure call, caption, label, and reference remains intact.
All remaining figures retain their preceding size settings.

## Verification

The final PDF contains 75 pages. All pages were rendered and visually inspected;
all six enlarged figures were additionally inspected at higher resolution.
Early viewport drafts that clipped labels were rejected. The final portrait grids
retain every axis label, colour bar, title, and panel. Captions remain attached
to the figures, with consecutive numbers 1 through 65 and eight tables.
Text extraction confirms valid reference markers and text within page margins.
Compilation used Tectonic 0.17 to produce XDV and xdvipdfmx for the final PDF.
There are no overfull boxes, undefined references, or missing-character warnings.
The two preceding underfull paragraph warnings remain unchanged.

All 65 static-image SHA-256 checksums match the manifest. The source package
contains nine document/build/review/layout files, 65 original static images, and
two portrait display images.
Archive integrity and file bytes were checked. The delivered PDF matches the
visually inspected proof. Checksums confirm that all 117 historical main-report
and supplementary source/output files remain unchanged. The preceding integrated
PDF and source ZIP were preserved under `output/archive/` before this edit.
