"""Resolve the fork: does the ranker KNOW the correct far edge, or does it under-score it?

For every linking-limited miss (both cells detected, edge not made), look up the ranker's candidate
probability for the correct edge (src->correct) vs the distractor the linker actually took (src->stolen),
using the Phase-2 candidate-prob sidecars. Classifies each miss:
  - ranker-knows     : prob(correct) > prob(distractor)  -> selection/ILP lost it; fixable downstream
  - candidate>0.5    : prob(correct) cleared the 0.5 candidate gate -> it WAS a candidate the pipeline dropped
  - sub-threshold    : 0 < prob(correct) <= 0.5 -> ranker scored it but below the gate (rel penalty)
  - absent (<dump)   : prob(correct) below the dump floor -> ranker effectively blind to it
If correct is mostly sub-threshold / out-ranked by the distractor, the ranker is the bottleneck and the
fix needs a displacement-agnostic signal (appearance re-link / retrain), not a postproc knob.

    .venv-track/bin/python eval/candidate_probe.py <candidates_dir> <submission.csv> --data-dir data/train
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE / "_vendor")]

import numpy as np
import polars as pl
import tracksdata as td

from failure_modes import load_pred, load_gt
from tracking_cellmot.metrics import evaluate as compute_metric

K = td.DEFAULT_ATTR_KEYS
SCALE = (1.625, 0.40625, 0.40625)


def probe_video(cand_dir: Path, submission: Path, name: str, data_dir: Path):
    cpath = cand_dir / f"{name}.candidates.parquet"
    if not cpath.exists():
        return []
    cdf = pl.read_parquet(cpath)
    cand = {(int(s), int(t)): float(p)
            for s, t, p in zip(cdf["source_id"], cdf["target_id"], cdf["edge_prob"])}

    pred = load_pred(submission, name)
    gt, _ = load_gt(data_dir / f"{name}.geff")
    p = pred.copy()
    compute_metric(p, gt, scale=SCALE, max_distance=7.0)
    na = p.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID]).to_dict(as_series=False)
    p2g = {int(i): int(m) for i, m in zip(na[K.NODE_ID], na[K.MATCHED_NODE_ID]) if m is not None and int(m) != -1}
    g2p = {v: k for k, v in p2g.items()}
    mg = set(p2g.values())
    ea = p.edge_attrs(attr_keys=[K.MATCHED_EDGE_MASK, "source_id", "target_id"]).to_dict(as_series=False)
    out = defaultdict(list)
    cov = set()
    for m, s, t in zip(ea[K.MATCHED_EDGE_MASK], ea["source_id"], ea["target_id"]):
        out[int(s)].append(int(t))
        if m and int(s) in p2g and int(t) in p2g:
            cov.add(frozenset((p2g[int(s)], p2g[int(t)])))
    gea = gt.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]).to_dict(as_series=False)

    rows = []
    for gs, gt_ in zip(gea[K.EDGE_SOURCE], gea[K.EDGE_TARGET]):
        gs, gt_ = int(gs), int(gt_)
        if frozenset((gs, gt_)) in cov:
            continue
        if gs not in mg or gt_ not in mg:
            continue  # detection-limited miss
        ps, cpt = g2p[gs], g2p[gt_]
        stolen = [t for t in out.get(ps, []) if t != cpt]
        p_correct = cand.get((ps, cpt), 0.0)
        p_stolen = max((cand.get((ps, d), 0.0) for d in stolen), default=0.0)
        rows.append((p_correct, p_stolen, bool(stolen)))  # has_distractor
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", type=Path, help="dir of {name}.candidates.parquet sidecars")
    ap.add_argument("submission", type=Path, help="baseline submission csv (post-postproc)")
    ap.add_argument("--data-dir", type=Path, default=Path("data/train"))
    args = ap.parse_args()

    names = sorted(f.name[:-len(".candidates.parquet")] for f in args.candidates.glob("*.candidates.parquet"))
    if not names:
        sys.exit(f"no *.candidates.parquet in {args.candidates}")

    allrows = []
    print(f"{'video':<16}{'misses':>8}{'stolen':>8}{'cand>.5':>9}{'sub-thr':>9}{'absent':>8}")
    for name in names:
        rows = probe_video(args.candidates, args.submission, name, args.data_dir)
        stolen = sum(1 for _, _, hd in rows if hd)
        cand = sum(1 for pc, _, _ in rows if pc > 0.5)
        sub = sum(1 for pc, _, _ in rows if 0.0 < pc <= 0.5)
        absent = sum(1 for pc, _, _ in rows if pc == 0.0)
        print(f"{name:<16}{len(rows):>8}{stolen:>8}{cand:>9}{sub:>9}{absent:>8}")
        allrows += rows

    n = max(len(allrows), 1)
    cand = sum(1 for pc, _, _ in allrows if pc > 0.5)
    sub = sum(1 for pc, _, _ in allrows if 0.0 < pc <= 0.5)
    absent = sum(1 for pc, _, _ in allrows if pc == 0.0)
    stolen_rows = [(pc, ps) for pc, ps, hd in allrows if hd]   # only where a distractor was actually taken
    knows = sum(1 for pc, ps in stolen_rows if pc > ps)
    ns = max(len(stolen_rows), 1)
    pcs = np.array([pc for pc, _, _ in allrows])
    print("\n===== AGGREGATE (linking-limited misses) =====")
    print(f" total={len(allrows)}  ({len(stolen_rows)} stolen / {len(allrows)-len(stolen_rows)} unlinked-source)")
    print(f" prob(correct) cleared 0.5 candidate gate: {cand} ({100*cand/n:.0f}%)")
    print(f" prob(correct) sub-threshold (0,0.5]:      {sub} ({100*sub/n:.0f}%)")
    print(f" prob(correct) below dump floor (~0):      {absent} ({100*absent/n:.0f}%)")
    print(f" prob(correct): median {np.median(pcs):.3f}")
    print(f" among STOLEN misses: ranker ranks correct > taken distractor: {knows}/{len(stolen_rows)} ({100*knows/ns:.0f}%)")
    med_c = np.median([pc for pc, _ in stolen_rows]) if stolen_rows else float("nan")
    med_s = np.median([ps for _, ps in stolen_rows]) if stolen_rows else float("nan")
    print(f"   stolen: median prob(correct) {med_c:.3f}  vs  prob(distractor) {med_s:.3f}")
    print("\n=> cand>.5 high  -> correct WAS a viable candidate the selection/ILP dropped (fixable upstream, cheaper)")
    print("=> sub-threshold/absent high & ranker doesn't rank correct>distractor -> ranker under-scores large-rel")
    print("   edges -> the fix needs a displacement-agnostic signal (appearance re-link Rung 4 / retrain Rung 6)")


if __name__ == "__main__":
    main()
