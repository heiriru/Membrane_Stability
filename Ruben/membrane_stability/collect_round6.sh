#!/bin/bash
# Copies the round-6 runs from /tmp/claude-0/runs to results/round6 (meshes' large files and checkpoints excluded).
R=/tmp/claude-0/runs
D=$(cd "$(dirname "$0")" && pwd)/results/round6
mkdir -p "$D/lattice_2d" "$D/tube"
# 2D slope equation inputs
cp $R/lat_force/longwave.json $R/lat_force/transverse.json "$D/lattice_2d/" 2>/dev/null
cp $R/lat_force/nonlinear_cell_L6_a*.json $R/lat_force/nonlinear_cell_L8_a*.json "$D/lattice_2d/" 2>/dev/null
# angle 0: the round-4 cells (all drives)
cp "$D/../round4/lattice_nonlinear/nonlinear_cell_L6.json" "$D/../round4/lattice_nonlinear/nonlinear_cell_L8.json" "$D/lattice_2d/"
# codim-2 point and wake imperfection
for name in imp_C0_wnl c2_M10 c2_M10_grid c2_M10_bt c2_M10_nf c2_hopf_chi0 c2_hopf_chi0.05 imp_M1 imp_M1_b imp_M10_chi0.1_wnl imp_M10_chi0.075_wnl c2dyn_static c2dyn_hopf; do
  [ -d "$R/r6/$name" ] && python3 -c "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True, ignore=shutil.ignore_patterns('checkpoint*', '*.xdmf', '*.h5'))" "$R/r6/$name" "$D/$name"
  [ -f "$R/r6/$name.log" ] && grep -v "JIT\|Solving\|Newton solver\|^\\\\int" "$R/r6/$name.log" > "$D/$name.log"
done
# tube: the remeshing tests and the continuation of the round-5 overhang (logs and diagnostics only)
for n in cont eq001 normal2 normal3 dbgsm2 dbgb02 dbgid dbgnr dbgA dbgA_0.1 dbgA_1000; do
  [ -f "$R/tube/$n.log" ] && grep -E "^step|ended|no conv|secant|static residual|Newton iteration|resum|start from|initial" "$R/tube/$n.log" > "$D/tube/$n.log"
  [ -f "$R/tube/$n/time_series.csv" ] && mkdir -p "$D/tube/$n" && cp "$R/tube/$n/time_series.csv" "$D/tube/$n/"
done
for n in meshsm2 meshb02 meshdbg; do
  [ -f "$R/tube/$n/remesh.json" ] && mkdir -p "$D/tube/$n" && cp "$R/tube/$n/remesh.json" "$D/tube/$n/"
done
echo "collected into $D"
