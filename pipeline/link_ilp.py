"""Global ILP linking (Ultrack-style) as an alternative to greedy Hungarian.

Builds a candidate-edge graph from our detections and lets `tracksdata`'s ILP
solver choose a globally consistent, flow-valid set of edges (with native ≤2-
daughter division + appear/disappear costs) — instead of locally-optimal greedy
frame-to-frame assignment. Detection is unchanged; only linking differs, so this
composes with any detector.

Offline note: the ILP uses the open-source CBC solver when no Gurobi license is
active (no internet needed). Chunking bounds solve time for large graphs.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import tracksdata as td

from .config import PipelineConfig


def ilp_link(node_rows: list[dict], cfg: PipelineConfig,
             n_neighbors: int = 6, delta_t: int = 1,
             appearance: float = 0.1, disappearance: float = 0.1,
             division: float = 1.0, timeout: float | None = 600.0) -> list[dict]:
    """Return edge rows (source_id/target_id over node_rows' node_id) chosen by ILP.

    node_rows: dicts with node_id, t, z, y, x (voxel). Distance is computed in
    physical µm (coords scaled by cfg.scale) with an 11µm-style gate = max_link_dist_um.
    """
    from .submission import edge_row

    if not node_rows:
        return []
    scale = np.asarray(cfg.scale, dtype=np.float64)

    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -1e9)
    # store PHYSICAL coords so DistanceEdges' Euclidean distance is in µm
    gids = g.bulk_add_nodes([
        {"t": int(r["t"]),
         "z": float(r["z"]) * scale[0],
         "y": float(r["y"]) * scale[1],
         "x": float(r["x"]) * scale[2]}
        for r in node_rows
    ])
    gid_to_nodeid = {int(gid): int(r["node_id"]) for gid, r in zip(gids, node_rows)}

    # candidate edges between consecutive frames within the physical gate
    td.edges.DistanceEdges(
        distance_threshold=cfg.max_link_dist_um,
        n_neighbors=n_neighbors,
        delta_t=delta_t,   # >1 adds gap-closing candidate edges (bridge missed detections)
    ).add_edges(g)

    if g.num_edges() == 0:
        return []

    # distance -> reward: prob in [0,1], 1 at dist 0 (mirrors the reference's -1*edge_prob)
    ea = g.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_ID, "distance"])
    g.add_edge_attr_key("edge_prob", pl.Float64, 0.0)
    prob = (1.0 - ea["distance"] / cfg.max_link_dist_um).clip(0.0, 1.0)
    g.update_edge_attrs(edge_ids=ea[td.DEFAULT_ATTR_KEYS.EDGE_ID].to_list(),
                        attrs={"edge_prob": prob.to_list()})

    solver = td.solvers.ILPSolver(
        edge_weight=-1.0 * td.EdgeAttr("edge_prob"),
        appearance_weight=appearance,
        disappearance_weight=disappearance,
        division_weight=division,
        timeout=timeout,   # bound solve time for the offline 12h budget
    )
    solved = solver.solve(g)
    if solved is None:
        # solution written in place on g under DEFAULT_ATTR_KEYS.SOLUTION
        solved = g
        sel = solved.edge_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
                       td.DEFAULT_ATTR_KEYS.SOLUTION]
        ).filter(pl.col(td.DEFAULT_ATTR_KEYS.SOLUTION))
    else:
        sel = solved.edge_attrs(
            attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
        )

    dataset = node_rows[0]["dataset"]
    out = []
    for row in sel.iter_rows(named=True):
        s = gid_to_nodeid.get(int(row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]))
        t = gid_to_nodeid.get(int(row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET]))
        if s is not None and t is not None:
            out.append(edge_row(dataset, s, t))
    return out
