#!/bin/bash
# Copy the finished runs of the second flow study and the ring sweeps from the run directory (default
# /tmp/claude-0/runs) into results/: small files only (json, csv, selected npz), logs filtered.
RUNS=${1:-/tmp/claude-0/runs}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$HERE/results/flow2" "$HERE/results/logs"
for d in "$RUNS"/plate_* "$RUNS"/crit_* "$RUNS"/branch_* "$RUNS"/sens_*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    grep -q " $n exit 0" "$RUNS/queue/done.txt" 2>/dev/null || continue
    mkdir -p "$HERE/results/flow2/$n"
    cp "$d"/*.json "$d"/*.csv "$HERE/results/flow2/$n/" 2>/dev/null
    rm -f "$HERE/results/flow2/$n/test_integral_errors.csv"
    case $n in
        plate_A_*|plate_C_*) cp "$d"/plate_mode_*_G25.npz "$d"/plate_T1_*.npz "$HERE/results/flow2/$n/" 2>/dev/null ;;
    esac
    case $n in
        crit_A_*verify*|crit_C_free|crit_C_clamped) cp "$d"/critical_mode.npz "$HERE/results/flow2/$n/" 2>/dev/null ;;
        sens_*) cp "$d"/sensitivity.npz "$HERE/results/flow2/$n/" ;;
    esac
    grep -vE 'int f|FFC|Solving nonlinear|Newton solver finished|Check|Reading|close|sub_meshes' "$RUNS/$n.log" > "$HERE/results/logs/flow2_$n.log"
done
for d in "$RUNS"/ring_R*_tan*; do
    [ -d "$d" ] || continue
    n=$(basename "$d")
    grep -q " $n exit 0" "$RUNS/queue/done.txt" 2>/dev/null || continue
    mkdir -p "$HERE/results/$n"
    cp "$d"/*.json "$d"/*.csv "$d"/*.npz "$HERE/results/$n/" 2>/dev/null
    rm -f "$HERE/results/$n/test_integral_errors.csv"
    grep -vE 'int f|FFC|Solving nonlinear|Newton solver finished' "$RUNS/$n.log" > "$HERE/results/logs/fd_${n#ring_}.log"
done
