#!/bin/bash
# Follows the post-snap fold beyond the overhang (towards a tube): parametric time stepping (param_snap.py) in
# segments; when the parametrization degenerates (end = remesh) or Newton fails, the surface is remeshed
# (param_remesh.py: harmonic reparametrization + new gmsh mesh) and the next segment starts from the transfer data.
# Resumable: finished segments are skipped, a running segment resumes from its checkpoint.
#   bash tube_driver.sh [run root] [first mesh dir] [first run dir (with checkpoint.npz)] [n segments]
R=${1:-/tmp/claude-0/runs/tube}
prev_mesh=${2:-/tmp/claude-0/meshes/pre_A}
prev_run=${3:-/tmp/claude-0/runs/param_eq}
NSEG=${4:-12}
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
PY="/usr/bin/python3.12"
export PYTHONPATH=$ROOT/env_shims
export STABILITY_PARAMS=${STABILITY_PARAMS:-zeta=1,gauge_delta=0.5}
CRIT=$ROOT/results/flow2/crit_A_free_verify
cd "$HERE"
mkdir -p "$R"
end_of() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(d['end'], d['steps'])" "$1/param_snap.json"; }
for seg in $(seq 1 "$NSEG"); do
  mesh=$R/mesh$seg
  run=$R/seg$seg
  if [ -f "$run/param_snap.json" ]; then
    read -r end steps < <(end_of "$run")
    echo "segment $seg: done ($end after $steps steps)"
    if [ "$end" != "remesh" ] && [ "$end" != "newton_failed" ]; then break; fi
    if [ "$steps" -lt 1 ]; then echo "segment $seg made no step: stop"; break; fi
    prev_mesh=$mesh
    prev_run=$run
    continue
  fi
  if [ ! -f "$mesh/transfer.npz" ]; then
    echo "remeshing from $prev_run -> $mesh"
    $PY param_remesh.py square_b "$prev_mesh" "$mesh" --run "$prev_run" --max_vertices 5000 > "$R/remesh$seg.log" 2>&1 || { echo "remesh failed"; exit 1; }
    grep "reparametrization\|stretch (" "$R/remesh$seg.log"
  fi
  echo "segment $seg: time stepping"
  $PY param_snap.py square_b "$mesh" "$run" --transfer "$mesh/transfer.npz" --critical_dir "$CRIT" --v0_factor 0.97 \
      --max_steps 80 --dt0 10 --remesh_aniso 10 --remesh_stretch 6 > "$R/seg$seg.log" 2>&1
  if [ ! -f "$run/param_snap.json" ]; then echo "segment $seg did not finish (killed?)"; exit 1; fi
  prev_mesh=$mesh
  prev_run=$run
done
echo "driver finished"
