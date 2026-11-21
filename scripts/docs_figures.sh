#!/usr/bin/env bash
# Draw the figures and animations of the tutorial (docs/public/tutorial/) from the example programs.
#
# The programs are the ones scripts/docs_examples.sh builds (build/docs-examples/bin, against the serial library);
# each tutorial program that has figures writes its data when run with `dump`, into build/docs-figures (git-ignored).
# docs/scripts/make_doc_figures.py then draws every figure on both grounds and encodes the animations (ffmpeg).
# The figures are committed; CI does not redraw them.
#
# Usage: bash scripts/docs_figures.sh [NAME ...]      (NAME: a figure or animation of make_doc_figures.py; default all)
#        PYTHON=/path/to/python3 for another interpreter (numpy, scipy, matplotlib)
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
bin=$root/build/docs-examples/bin
data=$root/build/docs-figures

[ -x "$bin/t01_box" ] || bash "$root/scripts/docs_examples.sh"
rm -rf -- "$data"
mkdir -p "$data"
for prog in t01_box t02_faces t03_isolated t04_amr t05_values t06_iterations t08_embers; do
  (cd "$data" && "$bin/$prog" dump > /dev/null)
done
only=()
[ $# -gt 0 ] && only=(--only "$@")
"${PYTHON:-python3}" -I "$root/docs/scripts/make_doc_figures.py" --data "$data" "${only[@]}"
