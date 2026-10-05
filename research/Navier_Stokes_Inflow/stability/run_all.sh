#!/bin/bash
# Full pipeline of the finite-element linear stability analysis of the flow past a cylinder.
# Run from this folder, inside the FEniCS container (the shared folder is mounted at /home/fenics/shared).
# Total time on 4 cores: a few hours (the fine-mesh sweep and the DNS dominate).
set -e
SHARED=/home/fenics/shared
MESHES=$SHARED/generate_mesh/2d/cylinder
export PYTHONPATH=$SHARED/modules:.
PY=${PY:-python3}

# 1) meshes (unbounded domain: coarse / medium / fine, and the DFG channel)
(cd $MESHES && for spec in "unbounded 1.4 solution_unbounded_coarse" "unbounded 1.0 solution_unbounded_medium" \
                           "unbounded 0.75 solution_unbounded_fine" "dfg 1.0 solution_dfg"; do
    set -- $spec; mkdir -p $3; $PY generate_cylinder_mesh.py $1 $2 $3; done)

# 2) intermediate checks
mkdir -p solution/checks solution/logs
$PY check_eigensolver.py solution/checks | tee solution/checks/check_eigensolver.log
$PY check_projector_equivalence.py square mesh solution/checks --Re 50 | tee solution/checks/check_projector_equivalence.log
$PY check_dfg_benchmark.py channel $MESHES/solution_dfg solution/checks | tee solution/checks/check_dfg_benchmark.log

# 3) stability analysis: sweep in Re and critical Reynolds number, on three meshes
$PY stability_analysis.py cylinder $MESHES/solution_unbounded_coarse solution/cylinder_coarse > solution/logs/sweep_coarse.log &
$PY stability_analysis.py cylinder $MESHES/solution_unbounded_medium solution/cylinder_medium > solution/logs/sweep_medium.log &
wait
$PY stability_analysis.py cylinder $MESHES/solution_unbounded_fine solution/cylinder_fine --Re 40,44,48,52,60 \
    --spectrum_Re 40,48,52,60 > solution/logs/sweep_fine.log &
# eigenvalue at Re = 100 on the coarse mesh, for the comparison with the DNS
$PY stability_analysis.py cylinder $MESHES/solution_unbounded_coarse solution/cylinder_coarse_extra --Re 100 \
    --spectrum_Re 100 > solution/logs/sweep_coarse_extra.log &

# 4) DNS on the coarse mesh: stable (40), weakly unstable (60) and strongly unstable (100)
for Re in 40 60 100; do
    $PY solve_dynamics.py cylinder $MESHES/solution_unbounded_coarse solution/dns_Re$Re --Re $Re --T 300 --dt 0.1 \
        > solution/logs/dns_Re$Re.log &
done
wait

# 5) figures
$PY plot_results.py solution figures
