#!/usr/bin/env bash
set -euo pipefail
report_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
report_output="${REPORT_OUTPUT_DIR:-$report_dir/build}"
mkdir -p -- "$report_output"
report_output="$(cd -- "$report_output" && pwd)"
cd -- "$report_dir"
if [[ -n "${TECTONIC_BIN:-}" ]]; then
    "$TECTONIC_BIN" --keep-logs --outdir "$report_output" integrated.tex
elif command -v tectonic >/dev/null 2>&1; then
    tectonic --keep-logs --outdir "$report_output" integrated.tex
elif command -v pdflatex >/dev/null 2>&1; then
    for report_pass in 1 2 3; do
        pdflatex -interaction=nonstopmode -halt-on-error -output-directory "$report_output" integrated.tex
    done
else
    printf '%s\n' 'Install Tectonic or a LaTeX distribution with pdfLaTeX.' >&2
    exit 1
fi
printf 'Integrated PDF written to %s\n' "$report_output"
