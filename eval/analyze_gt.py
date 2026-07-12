"""Characterize the sparse ground-truth annotation structure across all datasets.

Key questions for detector/linking strategy:
  * Is GT a few COMPLETE lineages (cells tracked across all frames) or scattered points?
  * Do annotated tracks span the full movie (continuous) or short fragments?
  * How spatially clustered are annotated cells (a labeled sub-population vs. random)?
  * Division structure.

Run:  .venv-track/bin/python eval/analyze_gt.py --data-dir data/train
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import tracksdata as td

SCALE = np.array([1.625, 0.40625, 0.40625])


def components(node_ids, edges):
    adj = {int(n): [] for n in node_ids}
    for s, t in edges:
        adj[s].append(t); adj[t].append(s)
    seen, comps = set(), []
    for n in node_ids:
        n = int(n)
        if n in seen:
            continue
        comp = []
        q = deque([n]); seen.add(n)
        while q:
            u = q.popleft(); comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); q.append(v)
        comps.append(comp)
    return comps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train", type=Path)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    geffs = sorted(args.data_dir.glob("*.geff"))
    if args.limit:
        geffs = geffs[:args.limit]

    n_comps, comp_sizes, tspan_cov, full_span, divs_per_ds, cluster_ratio = [], [], [], [], [], []
    for gp in geffs:
        r = td.graph.IndexedRXGraph.from_geff(gp)
        g = r[0] if isinstance(r, tuple) else r
        na = g.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", "z", "y", "x"])
        nid = na[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list()
        tof = {int(row[td.DEFAULT_ATTR_KEYS.NODE_ID]): int(row["t"]) for row in na.iter_rows(named=True)}
        pos = {int(row[td.DEFAULT_ATTR_KEYS.NODE_ID]): np.array([row["z"], row["y"], row["x"]]) for row in na.iter_rows(named=True)}
        ea = g.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
        edges = [(int(row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]), int(row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET])) for row in ea.iter_rows(named=True)]
        T = max(tof.values()) + 1 if tof else 1

        comps = components(nid, edges)
        n_comps.append(len(comps))
        for c in comps:
            comp_sizes.append(len(c))
            ts = [tof[n] for n in c]
            tspan_cov.append((max(ts) - min(ts) + 1) / T)          # fraction of movie the lineage spans
            full_span.append(1 if (min(ts) == 0 and max(ts) == T - 1) else 0)
        outdeg = defaultdict(int)
        for s, t in edges:
            outdeg[s] += 1
        divs_per_ds.append(sum(1 for v in outdeg.values() if v >= 2))
        # spatial clustering: mean pairwise dist of annotated cells (µm) vs volume extent
        if len(pos) > 2:
            P = np.array(list(pos.values())) * SCALE
            centroid = P.mean(0)
            cluster_ratio.append(float(np.linalg.norm(P - centroid, axis=1).mean()))

    def stats(a, label, pct=True):
        a = np.array(a, dtype=float)
        f = 100 if pct else 1
        print(f"  {label:32} min={a.min()*f:.1f} med={np.median(a)*f:.1f} mean={a.mean()*f:.1f} max={a.max()*f:.1f}")

    print(f"Analyzed {len(geffs)} datasets, {len(comp_sizes)} total lineages (components)\n")
    stats(n_comps, "lineages per dataset", pct=False)
    stats(comp_sizes, "nodes per lineage", pct=False)
    stats(tspan_cov, "lineage t-span (% of movie)")
    print(f"  lineages spanning the FULL movie (t=0..T-1): {100*np.mean(full_span):.1f}%")
    stats(divs_per_ds, "divisions per dataset", pct=False)
    stats(cluster_ratio, "mean dist to annotation centroid (µm)", pct=False)
    print(f"\n  → interpretation:")
    med_cov = np.median(tspan_cov)
    print(f"    lineages are {'CONTINUOUS full-movie tracks' if med_cov > 0.8 else 'FRAGMENTS' if med_cov < 0.4 else 'partial tracks'} "
          f"(median span {100*med_cov:.0f}% of movie)")
    print(f"    ~{np.median(n_comps):.0f} annotated lineages/dataset, ~{np.median(comp_sizes):.0f} nodes each")


if __name__ == "__main__":
    main()
