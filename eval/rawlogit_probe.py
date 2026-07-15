"""Does the model RANK the true source above wrong ones (pre-softmax), even though softmax buries it?

candidate_probe found softmax prob(correct) ≈ 0 — but softmax-over-sources with ~50 candidates floors each prob
near 1/50 ≈ the dump floor, so that may be an artifact. This reads the RAW pre-softmax logits (top-K sources per
target from the instrumented run) and asks: for each fast-motion miss, where does the TRUE source rank among the
correct target's sources? If it's rank 1 for many misses, the model knows and a per-target argmax re-rank could
recover them WITHOUT retraining. If the true source ranks low/absent, the model is genuinely blind → retrain.

    .venv-track/bin/python eval/rawlogit_probe.py <sidecars_dir> <submission.csv> --data-dir data/train
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


def probe_video(sidecars: Path, submission: Path, name: str, data_dir: Path):
    rpath = sidecars / f"{name}.rawlogits.parquet"
    if not rpath.exists():
        return []
    rdf = pl.read_parquet(rpath)
    # per target: sorted list of (source, raw_logit) descending
    by_target: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for s, t, lg in zip(rdf["source_id"], rdf["target_id"], rdf["raw_logit"]):
        by_target[int(t)].append((int(s), float(lg)))
    for t in by_target:
        by_target[t].sort(key=lambda x: -x[1])

    pred = load_pred(submission, name)
    gt, _ = load_gt(data_dir / f"{name}.geff")
    p = pred.copy()
    compute_metric(p, gt, scale=SCALE, max_distance=7.0)
    na = p.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID]).to_dict(as_series=False)
    p2g = {int(i): int(m) for i, m in zip(na[K.NODE_ID], na[K.MATCHED_NODE_ID]) if m is not None and int(m) != -1}
    g2p = {v: k for k, v in p2g.items()}
    mg = set(p2g.values())
    ea = p.edge_attrs(attr_keys=[K.MATCHED_EDGE_MASK, "source_id", "target_id"]).to_dict(as_series=False)
    cov = set()
    for m, s, t in zip(ea[K.MATCHED_EDGE_MASK], ea["source_id"], ea["target_id"]):
        if m and int(s) in p2g and int(t) in p2g:
            cov.add(frozenset((p2g[int(s)], p2g[int(t)])))
    gea = gt.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]).to_dict(as_series=False)

    rows = []
    for gs, gt_ in zip(gea[K.EDGE_SOURCE], gea[K.EDGE_TARGET]):
        gs, gt_ = int(gs), int(gt_)
        if frozenset((gs, gt_)) in cov or gs not in mg or gt_ not in mg:
            continue
        ps, cpt = g2p[gs], g2p[gt_]
        srcs = by_target.get(cpt, [])
        rank = next((r for r, (s, _) in enumerate(srcs, 1) if s == ps), None)  # 1-indexed, None if absent
        rows.append(rank)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sidecars", type=Path)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--data-dir", type=Path, default=Path("data/train"))
    args = ap.parse_args()

    names = sorted(f.name[:-len(".rawlogits.parquet")] for f in args.sidecars.glob("*.rawlogits.parquet"))
    if not names:
        sys.exit(f"no *.rawlogits.parquet in {args.sidecars} — run the instrumented predict with the raw-logit dump")
    allranks = []
    print(f"{'video':<16}{'misses':>8}{'rank1':>7}{'top3':>7}{'top8':>7}{'absent':>8}")
    for name in names:
        ranks = probe_video(args.sidecars, args.submission, name, args.data_dir)
        r1 = sum(1 for r in ranks if r == 1)
        t3 = sum(1 for r in ranks if r is not None and r <= 3)
        t8 = sum(1 for r in ranks if r is not None)
        ab = sum(1 for r in ranks if r is None)
        print(f"{name:<16}{len(ranks):>8}{r1:>7}{t3:>7}{t8:>7}{ab:>8}")
        allranks += ranks

    n = max(len(allranks), 1)
    r1 = sum(1 for r in allranks if r == 1)
    t3 = sum(1 for r in allranks if r is not None and r <= 3)
    t8 = sum(1 for r in allranks if r is not None)
    print("\n===== AGGREGATE (true-source rank for the correct target, by RAW logit) =====")
    print(f" misses={len(allranks)}")
    print(f" true source RANK 1 for correct target: {r1} ({100*r1/n:.0f}%)  <- per-target argmax re-rank recovers these")
    print(f" true source in TOP 3:                  {t3} ({100*t3/n:.0f}%)")
    print(f" true source in TOP 8 (dumped):         {t8} ({100*t8/n:.0f}%)")
    print(f" true source ABSENT from top 8:         {n-t8} ({100*(n-t8)/n:.0f}%)  <- model genuinely doesn't rank it")
    print("\n=> rank1 high -> softmax/threshold buried a correct ranking -> CHEAP re-rank fix, no retrain")
    print("=> mostly absent -> the model truly can't identify the source -> retrain (Rung 6)")


if __name__ == "__main__":
    main()
