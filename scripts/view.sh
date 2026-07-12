#!/usr/bin/env bash
# Launch CellTrack Studio (napari GUI) on one dataset's prediction.
# Usage: scripts/view.sh <stem> [downsample]     e.g. scripts/view.sh 6bba_09961292
# Predictions must already exist under predictions/<stem>.geff
# (build them with: .venv-track/bin/python scripts/submission_to_geff.py <csv> --out predictions)
set -euo pipefail
STEM="${1:?usage: view.sh <stem> [downsample]}"
DS="${2:-1,4,4}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUDIO="$HOME/celltrack-studio/.venv/bin/celltrack-studio"

[ -x "$STUDIO" ] || { echo "studio not installed at $STUDIO"; exit 1; }
[ -d "$ROOT/predictions/$STEM.geff" ] || { echo "no prediction: predictions/$STEM.geff — run submission_to_geff.py first"; exit 1; }
[ -d "$ROOT/data/train/$STEM.geff" ] || { echo "no GT/image for $STEM in data/train/"; exit 1; }

cd "$ROOT"
exec "$STUDIO" --name "$STEM" --data-dir data/train --pred predictions --downsample "$DS"
