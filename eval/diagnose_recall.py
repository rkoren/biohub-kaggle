"""Diagnose WHY detection recall collapses on dense volumes.

For each GT node, find the nearest detection in the same frame (physical µm) and
bucket it. This separates the failure modes the metric's aggregate recall hides:
  * < 7µm  → matched (counts toward recall)
  * 7–14µm → detected-but-mislocalized OR a neighbor is closest (NMS merge / drift)
  * > 14µm or none → genuine miss (threshold too high / cell not detected)

If most misses are 7–14µm, the fix is localization / NMS (cheap). If most are
"none nearby", the fix is detection sensitivity / peak separation.

Run:  .venv-track/bin/python eval/diagnose_recall.py 44b6_551a5dba
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root for `pipeline`

import numpy as np
import tracksdata as td

from pipeline.config import PipelineConfig
from pipeline.detect import detect_cells
from pipeline.io_zarr import VolumeSeries

SCALE = np.array([1.625, 0.40625, 0.40625])


def gt_nodes_by_frame(geff_path: Path) -> dict[int, list[tuple[float, float, float]]]:
    r = td.graph.IndexedRXGraph.from_geff(geff_path)
    g = r[0] if isinstance(r, tuple) else r
    na = g.node_attrs(attr_keys=["t", "z", "y", "x"])
    by_t: dict[int, list] = defaultdict(list)
    for row in na.iter_rows(named=True):
        by_t[int(row["t"])].append((row["z"], row["y"], row["x"]))
    return by_t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("--data-dir", default="data/train", type=Path)
    ap.add_argument("--menu", default="menu.yaml")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.menu)
    vs = VolumeSeries(args.data_dir / f"{args.dataset}.zarr")
    gt = gt_nodes_by_frame(args.data_dir / f"{args.dataset}.geff")

    nearest = []
    det_counts, gt_counts = [], []
    for t in range(vs.T):
        gt_pts = gt.get(t, [])
        if not gt_pts:
            continue
        coords, _ = detect_cells(vs.volume(t), cfg)
        det_counts.append(len(coords)); gt_counts.append(len(gt_pts))
        if len(coords) == 0:
            nearest.extend([np.inf] * len(gt_pts)); continue
        det_um = coords.astype(np.float64) * SCALE[None, :]
        for z, y, x in gt_pts:
            d = np.sqrt(((det_um - np.array([z, y, x]) * SCALE) ** 2).sum(1)).min()
            nearest.append(float(d))

    nearest = np.array(nearest)
    n = len(nearest)
    b_match = int((nearest < 7).sum())
    b_mid = int(((nearest >= 7) & (nearest < 14)).sum())
    b_far = int((nearest >= 14).sum())
    print(f"[{args.dataset}] GT nodes={n}  detections/frame≈{np.mean(det_counts):.0f} "
          f"vs GT/frame≈{np.mean(gt_counts):.0f}  (det/gt ratio={np.mean(det_counts)/max(np.mean(gt_counts),1):.1f}x)")
    print(f"  nearest-detection distance to each GT node:")
    print(f"    < 7µm  (MATCHED)               : {b_match:5d}  ({100*b_match/n:.1f}%)")
    print(f"    7–14µm (mislocalized/NMS-merge) : {b_mid:5d}  ({100*b_mid/n:.1f}%)")
    print(f"    ≥14µm / none (genuine miss)     : {b_far:5d}  ({100*b_far/n:.1f}%)")
    finite = nearest[np.isfinite(nearest)]
    print(f"  nearest-dist µm: median={np.median(finite):.2f} p75={np.percentile(finite,75):.2f} "
          f"p90={np.percentile(finite,90):.2f}")
    # z vs xy error contribution among the "close but missed" band (7-14µm)
    print(f"\n  Interpretation: {100*b_match/n:.0f}% within 7µm ceiling; "
          f"{'localization/NMS is the lever (many 7–14µm)' if b_mid >= b_far else 'detection sensitivity is the lever (many far/none)'}")


if __name__ == "__main__":
    main()
