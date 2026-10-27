#!/usr/bin/env bash
# Three walls with the same stone layout: mortared (0%), 20% and 40% frontal porosity.
# Re-running picks up where it stopped: finished cases are skipped, an interrupted case resumes
# from its last checkpoint (saved every 5000 steps).
# Usage:  bash run_all.sh            # full run, H=40 (~8 GB VRAM)
#         H=32 bash run_all.sh       # lighter (~4 GB VRAM)
set -euo pipefail
cd "$(dirname "$0")"
H=${H:-40}
OUT=${OUT:-results_H${H}}
EXTRA=${EXTRA:-}
run() {  # name porosity
  if [ -f "$OUT/$1.npz" ]; then echo "[$1] already finished, skipping"; return; fi
  python lbm3d.py --case "$1" --porosity "$2" --H "$H" --out "$OUT" $EXTRA
}
run solid 0.0
run p20   0.2
run p40   0.4
python postprocess.py "$OUT/solid.npz" "$OUT/p20.npz" "$OUT/p40.npz" --out "$OUT" --no-gif
echo "done -> $OUT/stone_wall_3d.html"
# rotatable 3D scene (needs: pip install scikit-image)
python export3d.py "$OUT/solid.npz" "$OUT/p20.npz" "$OUT/p40.npz" --out "$OUT" \
  || echo "3D scene skipped: run 'pip install scikit-image' and then the export3d.py line above"
