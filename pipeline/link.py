"""Frame-to-frame linking, conservative division detection, and node pruning.

Primary links are one-to-one Hungarian assignments in physical space with a
distance gate. A second pass adds a division edge only when a parent already has
one daughter and an unmatched second daughter sits close to both parent and
sister — protecting division *precision* (the metric weights divisions only 0.1
and they are extremely sparse in GT).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .config import PipelineConfig

BIG = 1e6


def link_frames(prev_ids: list[int], prev_xyz: np.ndarray,
                curr_ids: list[int], curr_xyz: np.ndarray,
                cfg: PipelineConfig) -> list[tuple[int, int]]:
    """Links from frame t-1 to t. A source gets a 2nd edge only via the division pass."""
    if len(prev_ids) == 0 or len(curr_ids) == 0:
        return []
    scale = np.asarray(cfg.scale, dtype=np.float64)
    P = prev_xyz.astype(np.float64) * scale[None, :]
    C = curr_xyz.astype(np.float64) * scale[None, :]
    D = np.sqrt(((P[:, None, :] - C[None, :, :]) ** 2).sum(axis=2))

    cost = np.where(D <= cfg.max_link_dist_um, D, BIG)
    ri, ci = linear_sum_assignment(cost)

    edges: list[tuple[int, int]] = []
    parent_children: dict[int, list[int]] = defaultdict(list)
    matched_curr: set[int] = set()
    for r, c in zip(ri, ci):
        if cost[r, c] < BIG:
            edges.append((int(prev_ids[r]), int(curr_ids[c])))
            parent_children[int(r)].append(int(c))
            matched_curr.add(int(c))

    if cfg.detect_divisions and (len(curr_ids) - len(prev_ids) >= cfg.div_min_count_gain):
        for c in range(len(curr_ids)):
            if c in matched_curr:
                continue
            best = None
            for p in range(len(prev_ids)):
                if len(parent_children.get(p, [])) != 1:
                    continue
                if D[p, c] > cfg.div_parent_dist_um:
                    continue
                sister = parent_children[p][0]
                sister_dist = float(np.linalg.norm(C[c] - C[sister]))
                if sister_dist <= cfg.div_sister_dist_um:
                    score = float(D[p, c] + 0.25 * sister_dist)
                    if best is None or score < best[0]:
                        best = (score, p)
            if best is not None:
                _, p = best
                edges.append((int(prev_ids[p]), int(curr_ids[c])))
                parent_children[p].append(int(c))
                matched_curr.add(int(c))
    return edges


def connected_components(node_ids: Iterable[int], edges: list[tuple[int, int]]) -> dict[int, int]:
    """node_id → component_id for undirected components (diagnostics)."""
    node_ids = list(node_ids)
    adj: dict[int, list[int]] = {int(n): [] for n in node_ids}
    for s, t in edges:
        if s in adj and t in adj:
            adj[s].append(t)
            adj[t].append(s)
    comp: dict[int, int] = {}
    cid = 0
    for n in node_ids:
        if n in comp:
            continue
        cid += 1
        comp[n] = cid
        q = deque([n])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in comp:
                    comp[v] = cid
                    q.append(v)
    return comp


def prune_isolated(node_rows: list[dict], edge_rows: list[dict],
                   node_scores: dict[int, float], cfg: PipelineConfig) -> tuple[list[dict], list[dict], dict]:
    """Drop detections that never participate in an edge (they only add node penalty)."""
    if not cfg.prune_isolated_nodes or not node_rows:
        return node_rows, edge_rows, {"removed_isolated": 0,
                                      "kept_nodes": len(node_rows), "kept_edges": len(edge_rows)}
    all_ids = [int(r["node_id"]) for r in node_rows]
    keep: set[int] = set()
    for e in edge_rows:
        keep.add(int(e["source_id"]))
        keep.add(int(e["target_id"]))
    if cfg.keep_strong_isolated:
        scores = np.array([node_scores.get(n, 0.0) for n in all_ids], dtype=np.float32)
        if len(scores):
            floor = float(np.quantile(scores, cfg.strong_isolated_quantile))
            keep.update(n for n in all_ids if node_scores.get(n, 0.0) >= floor)
    kept_nodes = [r for r in node_rows if int(r["node_id"]) in keep]
    kept_ids = {int(r["node_id"]) for r in kept_nodes}
    kept_edges = [e for e in edge_rows
                  if int(e["source_id"]) in kept_ids and int(e["target_id"]) in kept_ids]
    return kept_nodes, kept_edges, {"removed_isolated": len(node_rows) - len(kept_nodes),
                                    "kept_nodes": len(kept_nodes), "kept_edges": len(kept_edges)}
