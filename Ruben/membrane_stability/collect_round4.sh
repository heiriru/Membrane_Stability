#!/bin/bash
# Copy the runs of the fourth study (force-free post-snap dynamics, nonlinear lattice cells, mobile proteins in IRENE)
# from /tmp/claude-0/runs to results/round4 (meshes excluded); filtered logs to results/logs/round4_*.log.
R=/tmp/claude-0/runs
D=$(dirname "$0")/results/round4
mkdir -p "$D/lattice_nonlinear" "$(dirname "$0")/results/logs"
for name in snapff_v0.93 snapff_v0.97 phi_chic2 phi_v0c phi_v0c_stiff phi_test; do
  [ -d "$R/$name" ] || continue
  python3 -c "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True, ignore=shutil.ignore_patterns('mesh*', '*.xdmf', '*.h5', 'test_integral_errors.csv'))" "$R/$name" "$D/$name"
  [ -f "$R/$name.log" ] && grep -v "Newton iteration\|Solving nonlinear\|Newton solver finished\|Solving linear\|JIT\|int f d\|Check t\|Reading param\|close\.\|Calling FFC" "$R/$name.log" > "$(dirname "$0")/results/logs/round4_$name.log"
done
cp $R/lat_force/nonlinear_cell_L*.json "$D/lattice_nonlinear/" 2>/dev/null
grep -v "Newton\|Solving\|Calling" $R/nonlinear_cell.log > "$(dirname "$0")/results/logs/round4_nonlinear_cell.log" 2>/dev/null
du -sh "$D"
