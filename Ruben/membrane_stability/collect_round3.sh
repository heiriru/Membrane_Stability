#!/bin/bash
# Copy the finished runs of the third study (periodic cells, Bloch, multi-PI, flutter map, Landau, snap, force-free
# PI, tubes, local / mobile analyses) from /tmp/claude-0/runs to results/round3 (meshes excluded), filtered logs to
# results/logs/round3_*.log.
R=/tmp/claude-0/runs
D=$(dirname "$0")/results/round3
mkdir -p "$D" "$(dirname "$0")/results/logs"
for name in landau_A_clamped post_amp_t0 post_amp_t-0.3 snap_t-0.3_v0.92 snap_t-0.3_v0.97 ff_t-0.3 ff_t-0.1 ff_t-0.03 \
            fmap_plate_g0 fmap_plate_g10 fmap_plate_g25 fmap_plate_g50 fmap_plate_g100 lat_force lat_fric5 lat_fric20 lat_force_s1e-5 lat_force_s1e-7 multi wrinkles \
            local mobile tube ring_table plate_validate check_vf_R10 \
            fmap_G0_ell2.5 fmap_G0_ell3.5 fmap_G0_ell5 fmap_G0_ell7 fmap_G0_ell14 fmap_G100_ell3.5 fmap_G100_ell7 \
            fmap_G100_ell14 fmap_G25_ell3.5 fmap_G25_ell7 fmap_G25_ell14; do
  [ -d "$R/$name" ] || continue
  python3 -c "import shutil, sys; shutil.copytree(sys.argv[1], sys.argv[2], dirs_exist_ok=True, ignore=shutil.ignore_patterns('mesh*', '*.msh', '*.xdmf', '*.h5', 'R*_ell*', 'box', 'cell*', 'test_integral_errors.csv', 'scan_spectra.npz', 'leading_modes.npz'))" "$R/$name" "$D/$name"
  [ -f "$R/$name.log" ] && grep -v "Newton iteration\|Solving nonlinear\|Newton solver finished\|Solving linear\|JIT\|int f d\|Check t\|Reading param\|close\.\|Calling FFC" \
        "$R/$name.log" > "$(dirname "$0")/results/logs/round3_$name.log"
done
du -sh "$D"
