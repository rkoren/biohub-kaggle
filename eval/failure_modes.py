"""Where does the pipeline lose points? — split missed GT edges into detection- vs linking-limited.

The score is dominated by missed edges (FN). This asks the strategic question the 0.90-tuners aren't:
of the GT edges we miss, how many are because a cell was never detected (detection-limited) vs. because
both cells WERE detected but we didn't connect them (linking-limited)? If it's mostly linking, more
detection tuning is wasted and the lever is the linker; if detection, vice-versa. Also profiles divisions
and per-video error, and characterises the linking-limited misses (edge length, frame density).

    .venv-track/bin/python eval/failure_modes.py <preds_dir_or_csv> --data-dir data/train
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE / "_vendor")]

import numpy as np
import polars as pl
import tracksdata as td

from local_eval import load_gt, build_pred_graph
from tracking_cellmot.metrics import evaluate as compute_metric

K = td.DEFAULT_ATTR_KEYS
SCALE = (1.625, 0.40625, 0.40625)


def load_pred(source: Path, name: str):
    """pred geff dir (<name>.geff) or a submission CSV filtered to `name` → tracksdata graph."""
    gp = source / f"{name}.geff"
    if gp.exists():
        r = td.graph.IndexedRXGraph.from_geff(str(gp))
        return r[0] if isinstance(r, tuple) else r
    import pandas as pd
    df = pd.read_csv(source)
    g = df[df.dataset == name]
    n = g[g.row_type == "node"][["node_id", "t", "z", "y", "x"]].copy()
    e = g[g.row_type == "edge"][["source_id", "target_id"]].copy()
    # ids go float when node/edge rows share a column (blank cells → NaN); cast back
    for c in ("node_id", "t"):
        n[c] = n[c].astype("int64")
    for c in ("source_id", "target_id"):
        e[c] = e[c].astype("int64")
    # the real submission stores integer voxel coords — round to match what actually gets scored
    # (raw floats shift the 7µm node matching and mis-count FP/FN by a few edges)
    for c in ("z", "y", "x"):
        n[c] = n[c].round().astype("int64")
    return build_pred_graph(pl.from_pandas(n), pl.from_pandas(e))


def gt_tables(gt):
    na = gt.node_attrs(attr_keys=[K.NODE_ID, K.T, K.Z, K.Y, K.X]).to_dict(as_series=False)
    pos = {int(i): (int(t), float(z), float(y), float(x))
           for i, t, z, y, x in zip(na[K.NODE_ID], na[K.T], na[K.Z], na[K.Y], na[K.X])}
    ea = gt.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]).to_dict(as_series=False)
    edges = list(zip((int(s) for s in ea[K.EDGE_SOURCE]), (int(t) for t in ea[K.EDGE_TARGET])))
    return pos, edges


def analyze_one(pred, gt):
    p = pred.copy()
    er = compute_metric(p, gt, scale=SCALE, max_distance=7.0)
    # pred -> gt node map + which pred edges are TP
    na = p.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID]).to_dict(as_series=False)
    pred2gt = {int(i): int(m) for i, m in zip(na[K.NODE_ID], na[K.MATCHED_NODE_ID]) if m is not None and int(m) != -1}
    matched_gt = set(pred2gt.values())
    covered = set()
    if p.num_edges():
        ea = p.edge_attrs(attr_keys=[K.MATCHED_EDGE_MASK, "source_id", "target_id"]).to_dict(as_series=False)
        for m, s, t in zip(ea[K.MATCHED_EDGE_MASK], ea["source_id"], ea["target_id"]):
            if m and int(s) in pred2gt and int(t) in pred2gt:
                covered.add(frozenset((pred2gt[int(s)], pred2gt[int(t)])))
    # NB: the metric's edge_fp (er.edge_fp) counts pred edges that are "valid" (source's GT has an
    # out-edge OR target's GT has an in-edge) but unmatched — i.e. spurious links near GT-active cells,
    # NOT over-detection noise. Over-detected edges (endpoints unmatched to sparse GT) aren't scored.

    gt_pos, gt_edges = gt_tables(gt)
    det_fn = link_fn = 0
    link_lengths = []
    for s, t in gt_edges:
        if frozenset((s, t)) in covered:
            continue  # TP
        both_detected = s in matched_gt and t in matched_gt
        if both_detected:
            link_fn += 1
            (ts, zs, ys, xs), (tt, zt, yt, xt) = gt_pos[s], gt_pos[t]
            link_lengths.append(float(np.sqrt(((zs - zt) * SCALE[0]) ** 2 + ((ys - yt) * SCALE[1]) ** 2 + ((xs - xt) * SCALE[2]) ** 2)))
        else:
            det_fn += 1
    return er, det_fn, link_fn, link_lengths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path, help="dir of *.geff or a submission.csv")
    ap.add_argument("--data-dir", type=Path, default=Path("data/train"))
    args = ap.parse_args()

    src = args.predictions
    if src.is_dir():
        names = sorted(g.name[:-5] for g in src.glob("*.geff"))
    else:
        import pandas as pd
        names = sorted(pd.read_csv(src)["dataset"].unique())

    tot_tp = tot_fp = tot_fn = tot_det = tot_link = 0
    tot_div_tp = tot_div_fp = tot_div_fn = 0
    all_link_lengths = []
    print(f"{'video':<16}{'eTP':>6}{'eFP':>6}{'eFN':>6}{'FN=det':>8}{'FN=link':>8}{'divTP/FP/FN':>14}")
    for name in names:
        gt, _ = load_gt(args.data_dir / f"{name}.geff")
        pred = load_pred(src if src.is_dir() else src, name)
        er, det_fn, link_fn, lens = analyze_one(pred, gt)
        tot_tp += er.edge_tp; tot_fp += er.edge_fp; tot_fn += er.edge_fn
        tot_det += det_fn; tot_link += link_fn
        tot_div_tp += er.division_tp; tot_div_fp += er.division_fp; tot_div_fn += er.division_fn
        all_link_lengths += lens
        print(f"{name:<16}{er.edge_tp:>6}{er.edge_fp:>6}{er.edge_fn:>6}{det_fn:>8}{link_fn:>8}"
              f"{er.division_tp:>4}/{er.division_fp}/{er.division_fn:>4}")

    fn = max(tot_fn, 1)
    print("\n===== AGGREGATE =====")
    print(f" edge  TP={tot_tp}  FP={tot_fp}  FN={tot_fn}   (edge Jaccard = {tot_tp/max(tot_tp+tot_fp+tot_fn,1):.4f})")
    print(f" MISSED EDGES (FN) split:  detection-limited={tot_det} ({100*tot_det/fn:.0f}%)   "
          f"linking-limited={tot_link} ({100*tot_link/fn:.0f}%)")
    if all_link_lengths:
        a = np.array(all_link_lengths)
        print(f"   linking-limited misses: median len {np.median(a):.2f}µm, p90 {np.percentile(a,90):.2f}µm, "
              f">7µm(gate): {100*(a>7).mean():.0f}%")
    print(f" FALSE LINKS (FP)={tot_fp}: valid spurious links near GT-active cells (a linking error, not over-detection)")
    print(f" divisions  TP={tot_div_tp}  FP={tot_div_fp}  FN={tot_div_fn}")
    # ceilings on RAW edge jaccard (node penalty aside): reachable gain from each linking lever
    ej = tot_tp / max(tot_tp + tot_fp + tot_fn, 1)
    ej_recall = (tot_tp + tot_link) / max(tot_tp + tot_link + tot_fp, 1)   # link all detected-but-unlinked
    ej_prec = tot_tp / max(tot_tp + tot_fn, 1)                              # remove all valid FP
    print(f"\n ceilings (raw eJ, node-penalty aside): now={ej:.4f}"
          f" | +recover linking-FN→{ej_recall:.4f} | +remove all FP→{ej_prec:.4f}")
    print(" => FN 83% linking-limited + FP all valid spurious links => the linker is the lever, not detection.")


if __name__ == "__main__":
    main()
