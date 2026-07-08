"""Perturbation tests for the local metric harness.

The GT-vs-GT self-test only proves matching works when pred == GT. These tests
exercise the mechanics that actually decide the LB score:
  (a) far extra nodes/edges  -> edge_jaccard unchanged, only adj moves (off-GT edges invisible)
  (b) centroid jitter        -> matching falls off a cliff at the 7µm tolerance
  (c) dropped GT nodes        -> edge_jaccard drops; adj gets its (bounded) under-prediction bump
  (d) N_pred sweep 0.5–5x     -> map the node-count penalty response surface

Run inside the eval env:
    .venv-track/bin/python eval/perturbation_test.py --data-dir data/train --dataset 44b6_0c582fdc
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import polars as pl
import tracksdata as td

from local_eval import SCALE, SPATIAL, _new_graph, _score_pair, load_gt

RNG = random.Random(42)


def _gt_rows(gt: td.graph.BaseGraph) -> tuple[list[dict], list[tuple[int, int]]]:
    na = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", *SPATIAL])
    nodes = [
        {"id": int(r[td.DEFAULT_ATTR_KEYS.NODE_ID]), "t": int(r["t"]),
         "z": float(r["z"]), "y": float(r["y"]), "x": float(r["x"])}
        for r in na.iter_rows(named=True)
    ]
    ea = gt.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
    edges = [(int(r[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]), int(r[td.DEFAULT_ATTR_KEYS.EDGE_TARGET]))
             for r in ea.iter_rows(named=True)]
    return nodes, edges


def _build(nodes: list[dict], edges: list[tuple[int, int]]) -> td.graph.InMemoryGraph:
    g = _new_graph()
    idmap: dict[int, int] = {}
    for n in nodes:
        idmap[n["id"]] = g.add_node({"t": n["t"], "z": n["z"], "y": n["y"], "x": n["x"]})
    for s, t in edges:
        if s in idmap and t in idmap:
            g.add_edge(idmap[s], idmap[t], {})
    return g


def _add_far_chains(nodes, edges, n_extra, tmax, offset=10_000):
    """Append n_extra extra nodes as short chains far from any GT centroid."""
    nodes = list(nodes); edges = list(edges)
    nid = max(n["id"] for n in nodes) + 1
    made = 0
    while made < n_extra:
        t0 = RNG.randint(0, max(tmax - 1, 0))
        base = {"z": offset + RNG.random() * 50, "y": offset + RNG.random() * 50,
                "x": offset + RNG.random() * 50}
        prev = None
        for dt in range(2):
            n = {"id": nid, "t": t0 + dt, **base}
            nodes.append(n)
            if prev is not None:
                edges.append((prev, nid))
            prev = nid; nid += 1; made += 1
            if made >= n_extra:
                break
    return nodes, edges


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train", type=Path)
    ap.add_argument("--dataset", required=True)
    args = ap.parse_args()

    gt, n_total = load_gt(args.data_dir / f"{args.dataset}.geff")
    nodes, edges = _gt_rows(gt)
    tmax = max(n["t"] for n in nodes) + 1
    n_gt = len(nodes)
    print(f"[{args.dataset}] GT nodes={n_gt} edges={len(edges)} T_true={n_total:.0f}")

    def score(ns, es):
        # rebuild GT fresh each call (evaluate mutates the pred graph, not gt, but be safe)
        gt2, _ = load_gt(args.data_dir / f"{args.dataset}.geff")
        return _score_pair(_build(ns, es), gt2, n_total)

    base = score(nodes, edges)
    print(f"\nbaseline:        edge_J={base['edge_jaccard']:.4f} adj={base['adj_edge_jaccard']:.4f} "
          f"N_pred={base['num_pred_nodes']} recall={base['node_recall']:.3f}")

    # (a) far extra nodes: J must stay, adj must fall
    print("\n(a) add far off-GT nodes  [expect edge_J unchanged, adj decreases]")
    for extra in (n_gt, 5 * n_gt, 50 * n_gt):
        ns, es = _add_far_chains(nodes, edges, extra, tmax)
        r = score(ns, es)
        print(f"    +{extra:>6} far nodes: edge_J={r['edge_jaccard']:.4f} adj={r['adj_edge_jaccard']:.4f} "
              f"N_pred={r['num_pred_nodes']} ratio={(r['num_pred_nodes']-n_total)/n_total:+.3f}")

    # (b) centroid jitter along x (µm -> voxels via SCALE[x]); cliff at 7µm
    print("\n(b) jitter all centroids  [expect match cliff near 7µm]")
    for dmicron in (0.0, 3.0, 6.0, 6.9, 7.1, 8.0):
        dv = dmicron / SCALE[2]  # x scale
        jn = [{**n, "x": n["x"] + dv} for n in nodes]
        r = score(jn, edges)
        print(f"    jitter {dmicron:>4.1f}µm: edge_J={r['edge_jaccard']:.4f} recall={r['node_recall']:.3f}")

    # (c) drop a fraction of GT nodes
    print("\n(c) drop GT nodes  [expect edge_J down, adj gets under-prediction bump]")
    for frac in (0.2, 0.5):
        keep = [n for n in nodes if RNG.random() > frac]
        keep_ids = {n["id"] for n in keep}
        es = [(s, t) for s, t in edges if s in keep_ids and t in keep_ids]
        r = score(keep, es)
        print(f"    drop {int(frac*100)}%: edge_J={r['edge_jaccard']:.4f} adj={r['adj_edge_jaccard']:.4f} "
              f"N_pred={r['num_pred_nodes']} recall={r['node_recall']:.3f}")

    # (d) N_pred sweep as a fraction of T_true (far nodes) -> penalty surface
    print("\n(d) N_pred sweep vs T_true  [map the adj penalty surface]")
    for mult in (0.5, 0.9, 1.0, 2.0, 5.0):
        target = int(mult * n_total)
        extra = max(target - n_gt, 0)
        ns, es = _add_far_chains(nodes, edges, extra, tmax)
        r = score(ns, es)
        print(f"    N_pred≈{mult:>3.1f}·T_true ({r['num_pred_nodes']:>6}): "
              f"edge_J={r['edge_jaccard']:.4f} adj={r['adj_edge_jaccard']:.4f}")


if __name__ == "__main__":
    main()
