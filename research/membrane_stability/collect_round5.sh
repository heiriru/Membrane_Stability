#!/bin/bash
# Copy the runs of the fifth study (flutter, slope equation, protein dynamics, Peclet sweep, parametric fold, lattice
# cells near the tricritical density) from /tmp/claude-0/runs to results/round5 (meshes and checkpoints excluded);
# filtered logs to results/logs/round5_*.log.
R=/tmp/claude-0/runs
D=$(dirname "$0")/results/round5
mkdir -p "$D" "$(dirname "$0")/results/logs"
for name in flut_v1.05_k0.05 flut_v1.10_k0.05 flut_v1.20_k0.05 flut_v0.97_k2.0 flut_v0.97_from1.05 slope phidyn_rest phidyn_flow phidyn_source phidyn_flow2 phidyn_source2 phidyn_flow3 phidyn_source3 \
            pe_M1_chic pe_M1_v0c pe_M10_chic pe_M10_v0c pa_M1_chic pa_M1_v0c pa_M10_chic pa_M10_v0c pa_M100_chic pa_M100_v0c pa_M1000_chic pa_M1000_v0c test_M10_abs test_M10_nof test_M1_abs pa_M100_v0c_b pa_M1000_v0c_b \
            param_ff_v0.97r param_g05 param_eq lat_fine phidyn_rest_b; do
  [ -d "$R/$name" ] || continue
  python3 -c "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True, ignore=shutil.ignore_patterns('mesh*', '*.xdmf', '*.h5', 'test_integral_errors.csv', 'checkpoint*'))" "$R/$name" "$D/$name"
  [ -f "$R/$name.log" ] && grep -v "Newton iteration\|Solving nonlinear\|Newton solver finished\|Solving linear\|JIT\|int f d\|Check t\|Reading param\|close\.\|Calling FFC" "$R/$name.log" > "$(dirname "$0")/results/logs/round5_$name.log"
done
if [ -d "$R/lat_tc" ]; then
  mkdir -p "$D/lat_tc"
  cp $R/lat_tc/*.json "$D/lat_tc/" 2>/dev/null
fi
du -sh "$D"
