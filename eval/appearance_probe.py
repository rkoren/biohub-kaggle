"""Rung-4 viability: does the 32-ch appearance embedding separate the correct partner from the distractor?

The ranker gives fast-motion edges ~0 probability (candidate_probe). The only hope for a postproc recovery is
a displacement-agnostic signal. This tests the dumped per-node UNet appearance embeddings: for each fast-motion
miss, is cos(source, correct) > cos(source, distractor)? If yes for most, an appearance re-link (Rung 4) can
recover them; if appearance doesn't separate, only a retrain (Rung 6) will. Compares against the null (random
same-frame candidate) so we know the appearance signal beats chance.

    .venv-track/bin/python eval/appearance_probe.py <sidecars_dir> <submission.csv> --data-dir data/train
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE), str(_HERE / "_vendor")]

import numpy as np
import tracksdata as td

from failure_modes import load_pred, load_gt
from tracking_cellmot.metrics import evaluate as compute_metric

K = td.DEFAULT_ATTR_KEYS
SCALE = (1.625, 0.40625, 0.40625)


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def probe_video(sidecars: Path, submission: Path, name: str, data_dir: Path):
    npz = sidecars / f"{name}.embeddings.npz"
    if not npz.exists():
        return []
    z = np.load(npz)
    emb = {int(i): v for i, v in zip(z["node_ids"], z["embeddings"])}

    pred = load_pred(submission, name)
    gt, _ = load_gt(data_dir / f"{name}.geff")
    p = pred.copy()
    compute_metric(p, gt, scale=SCALE, max_distance=7.0)
    na = p.node_attrs(attr_keys=[K.NODE_ID, K.MATCHED_NODE_ID, K.T]).to_dict(as_series=False)
    p2g = {int(i): int(m) for i, m in zip(na[K.NODE_ID], na[K.MATCHED_NODE_ID]) if m is not None and int(m) != -1}
    node_t = {int(i): int(t) for i, t in zip(na[K.NODE_ID], na[K.T])}
    by_t = defaultdict(list)
    for nid, t in node_t.items():
        by_t[t].append(nid)
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

    rng = np.random.default_rng(0)
    rows = []
    for gs, gt_ in zip(gea[K.EDGE_SOURCE], gea[K.EDGE_TARGET]):
        gs, gt_ = int(gs), int(gt_)
        if frozenset((gs, gt_)) in cov or gs not in mg or gt_ not in mg:
            continue
        ps, cpt = g2p[gs], g2p[gt_]
        if ps not in emb or cpt not in emb:
            continue
        stolen = [t for t in out.get(ps, []) if t != cpt and t in emb]
        cos_correct = _cos(emb[ps], emb[cpt])
        cos_stolen = max((_cos(emb[ps], emb[d]) for d in stolen), default=None)
        # null: a random other detection in the correct target's frame
        pool = [n for n in by_t[node_t[cpt]] if n != cpt and n in emb]
        cos_rand = _cos(emb[ps], emb[int(rng.choice(pool))]) if pool else None
        rows.append((cos_correct, cos_stolen, cos_rand))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sidecars", type=Path)
    ap.add_argument("submission", type=Path)
    ap.add_argument("--data-dir", type=Path, default=Path("data/train"))
    args = ap.parse_args()

    names = sorted(f.name[:-len(".embeddings.npz")] for f in args.sidecars.glob("*.embeddings.npz"))
    allrows = []
    print(f"{'video':<16}{'misses':>8}{'cos_corr':>10}{'cos_dist':>10}{'cos_rand':>10}{'corr>dist':>10}")
    for name in names:
        rows = probe_video(args.sidecars, args.submission, name, args.data_dir)
        if not rows:
            continue
        cc = np.array([r[0] for r in rows])
        cd = np.array([r[1] for r in rows if r[1] is not None])
        cr = np.array([r[2] for r in rows if r[2] is not None])
        beats = [r[0] > r[1] for r in rows if r[1] is not None]
        print(f"{name:<16}{len(rows):>8}{cc.mean():>10.3f}"
              f"{(cd.mean() if len(cd) else float('nan')):>10.3f}{(cr.mean() if len(cr) else float('nan')):>10.3f}"
              f"{(100*np.mean(beats) if beats else float('nan')):>9.0f}%")
        allrows += rows

    cc = np.array([r[0] for r in allrows])
    cd = np.array([r[1] for r in allrows if r[1] is not None])
    cr = np.array([r[2] for r in allrows if r[2] is not None])
    beats = [r[0] > r[1] for r in allrows if r[1] is not None]
    beats_rand = [r[0] > r[2] for r in allrows if r[2] is not None]
    print("\n===== AGGREGATE =====")
    print(f" misses with embeddings: {len(allrows)}")
    print(f" mean cos(source, correct)   = {cc.mean():.3f}")
    print(f" mean cos(source, distractor)= {cd.mean():.3f}  (n={len(cd)})")
    print(f" mean cos(source, random)    = {cr.mean():.3f}  (null)")
    print(f" cos(correct) > cos(distractor): {100*np.mean(beats):.0f}%  (need >~65% for Rung-4 viability)")
    print(f" cos(correct) > cos(random):     {100*np.mean(beats_rand):.0f}%  (appearance beats chance?)")
    print("\n=> correct clearly > distractor AND > random -> appearance separates -> Rung 4 (appearance re-link) viable")
    print("=> correct ≈ distractor ≈ random -> raw appearance uninformative -> Rung 6 retrain (learned metric) needed")


if __name__ == "__main__":
    main()
