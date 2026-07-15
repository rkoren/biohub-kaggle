"""Name the linking failure mode: what distinguishes the blend's error edges from its correct edges?

For one held-out video, isolates the linking-limited misses (FN: both cells detected, edge not made) and the
spurious links (FP: valid pred edge with no GT match), then compares them to the correct edges (TP) on the two
standard linking-failure axes — local crowding (ambiguous assignment) and displacement (fast motion). If errors
skew dense/fast, that names the mode more specifically than "linking" and points at a concrete lever.

    .venv-track/bin/python eval/error_anatomy.py <submission.csv> <video> --data-dir data/train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE / "_vendor")]

import numpy as np
import tracksdata as td

from failure_modes import load_pred, load_gt
from tracking_cellmot.metrics import evaluate as compute_metric

K = td.DEFAULT_ATTR_KEYS
SCALE = np.array([1.625, 0.40625, 0.40625])


def _pos_by_frame(pos_t):
    """{t: (ids[], coords_um[])} for fast per-frame neighbour queries."""
    frames = {}
    for nid, (t, z, y, x) in pos_t.items():
        frames.setdefault(t, [[], []])
        frames[t][0].append(nid)
        frames[t][1].append([z * SCALE[0], y * SCALE[1], x * SCALE[2]])
    return {t: (np.array(a), np.array(b)) for t, (a, b) in frames.items()}


def _crowding(frames, t, point_um, r=7.0):
    """# other cells within r µm of point at frame t (source-frame crowding)."""
    if t not in frames:
        return 0
    _, coords = frames[t]
    if len(coords) == 0:
        return 0
    d = np.linalg.norm(coords - point_um, axis=1)
    return int((d < r).sum()) - 1  # exclude self


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", type=Path)
    ap.add_argument("video")
    ap.add_argument("--data-dir", type=Path, default=Path("data/train"))
    args = ap.parse_args()

    pred = load_pred(args.submission, args.video)
    gt, _ = load_gt(args.data_dir / f"{args.video}.geff")
    p = pred.copy()
    compute_metric(p, gt, scale=tuple(SCALE), max_distance=7.0)

    # pred->gt node map, covered gt edges, valid FP pred edges
    na = p.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID]).to_dict(as_series=False)
    pred2gt = {int(i): int(m) for i, m in zip(na[K.NODE_ID], na[K.MATCHED_NODE_ID]) if m is not None and int(m) != -1}
    matched_gt = set(pred2gt.values())

    gna = gt.node_attrs(attr_keys=[K.NODE_ID, K.T, K.Z, K.Y, K.X]).to_dict(as_series=False)
    gt_pos = {int(i): (int(t), float(z), float(y), float(x))
              for i, t, z, y, x in zip(gna[K.NODE_ID], gna[K.T], gna[K.Z], gna[K.Y], gna[K.X])}
    gt_frames = _pos_by_frame(gt_pos)
    gea = gt.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]).to_dict(as_series=False)
    gt_edges = [(int(s), int(t)) for s, t in zip(gea[K.EDGE_SOURCE], gea[K.EDGE_TARGET])]

    covered = set()
    if p.num_edges():
        ea = p.edge_attrs(attr_keys=[K.MATCHED_EDGE_MASK, "source_id", "target_id"]).to_dict(as_series=False)
        for m, s, t in zip(ea[K.MATCHED_EDGE_MASK], ea["source_id"], ea["target_id"]):
            if m and int(s) in pred2gt and int(t) in pred2gt:
                covered.add(frozenset((pred2gt[int(s)], pred2gt[int(t)])))

    def feats(s, t):
        (ts, zs, ys, xs), (tt, zt, yt, xt) = gt_pos[s], gt_pos[t]
        ps = np.array([zs, ys, xs]) * SCALE
        disp = float(np.linalg.norm(ps - np.array([zt, yt, xt]) * SCALE))
        return disp, _crowding(gt_frames, ts, ps)

    tp = [feats(s, t) for s, t in gt_edges if frozenset((s, t)) in covered]
    fn_link = [feats(s, t) for s, t in gt_edges
               if frozenset((s, t)) not in covered and s in matched_gt and t in matched_gt]

    def summ(label, rows):
        if not rows:
            print(f"  {label:<20} (none)")
            return
        d = np.array([r[0] for r in rows]); c = np.array([r[1] for r in rows])
        print(f"  {label:<20} n={len(rows):>4}  displacement µm: med {np.median(d):.2f} p90 {np.percentile(d,90):.2f}"
              f"   crowding(<7µm): med {np.median(c):.1f} mean {c.mean():.2f}")

    print(f"=== {args.video}: what distinguishes linking errors from correct edges? ===")
    summ("correct (TP)", tp)
    summ("missed (FN-link)", fn_link)
    if tp and fn_link:
        dtp = np.array([r[0] for r in tp]); dfn = np.array([r[0] for r in fn_link])
        ctp = np.array([r[1] for r in tp]); cfn = np.array([r[1] for r in fn_link])
        print(f"\n  FN-link vs TP:  displacement {np.median(dfn):.2f} vs {np.median(dtp):.2f} µm "
              f"({np.median(dfn)/max(np.median(dtp),0.1):.1f}×)   "
              f"crowding {cfn.mean():.2f} vs {ctp.mean():.2f} ({cfn.mean()/max(ctp.mean(),0.1):.1f}×)")
        print("  => higher displacement ⇒ fast-motion mode; higher crowding ⇒ dense-ambiguity mode.")


if __name__ == "__main__":
    main()
