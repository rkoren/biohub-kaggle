"""Local evaluation harness for the Biohub Cell Tracking competition.

Reproduces the official leaderboard score offline so we can iterate without
burning Kaggle submissions. Uses the organizers' vendored metric
(`eval/_vendor/tracking_cellmot`, BSD-3) + `tracksdata` for node matching.

Run inside the dedicated env:  `.venv-track/bin/python eval/local_eval.py ...`

Two entry points:
  * self-test:  score ground-truth-against-itself (edge Jaccard must be ~1.0)
  * score:      score a submission.csv against the paired .geff ground truth

The final `score` matches `tracking_cellmot.metrics.summarise`:
    score = adj_edge_jaccard(weighted) + 0.1 * division_jaccard(micro)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl
import tracksdata as td
from geff import GeffMetadata

# vendored official metric (torch-free subset)
sys.path.insert(0, str(Path(__file__).parent / "_vendor"))
from tracking_cellmot.metrics import (  # noqa: E402
    evaluate as compute_metric,
    node_recall,
    per_sample_metrics,
    summarise,
)

SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)  # microns per voxel (z, y, x)
SPATIAL = ("z", "y", "x")


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def _new_graph() -> td.graph.InMemoryGraph:
    g = td.graph.InMemoryGraph()
    for key in SPATIAL:
        g.add_node_attr_key(key, pl.Float64, 0.0)
    return g


def build_pred_graph(nodes: pl.DataFrame, edges: pl.DataFrame) -> td.graph.InMemoryGraph:
    """Build a predicted graph from submission-format node/edge rows.

    `nodes` needs columns: node_id, t, z, y, x.
    `edges` needs columns: source_id, target_id (referencing submission node_id).
    """
    g = _new_graph()
    id_map: dict[int, int] = {}
    for row in nodes.iter_rows(named=True):
        gid = g.add_node(
            {"t": int(row["t"]), "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
        )
        id_map[int(row["node_id"])] = gid
    for row in edges.iter_rows(named=True):
        s, t = int(row["source_id"]), int(row["target_id"])
        if s in id_map and t in id_map:
            g.add_edge(id_map[s], id_map[t], {})
    return g


def gt_to_pred_graph(gt: td.graph.BaseGraph) -> td.graph.InMemoryGraph:
    """Copy a GT graph into a fresh predicted graph (for the self-test)."""
    g = _new_graph()
    na = gt.node_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.NODE_ID, "t", *SPATIAL])
    id_map: dict[int, int] = {}
    for row in na.iter_rows(named=True):
        gid = g.add_node(
            {"t": int(row["t"]), "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])}
        )
        id_map[int(row[td.DEFAULT_ATTR_KEYS.NODE_ID])] = gid
    ea = gt.edge_attrs(attr_keys=[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE, td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
    for row in ea.iter_rows(named=True):
        s = int(row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE])
        t = int(row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET])
        g.add_edge(id_map[s], id_map[t], {})
    return g


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
def load_gt(geff_path: Path) -> tuple[td.graph.BaseGraph, float]:
    """Load a GT graph and its `estimated_number_of_nodes` (T_true)."""
    res = td.graph.IndexedRXGraph.from_geff(geff_path)
    gt = res[0] if isinstance(res, tuple) else res
    try:
        meta = GeffMetadata.read(geff_path)
        n_total = float((meta.extra or {}).get("estimated_number_of_nodes", float("nan")))
    except Exception:
        n_total = float("nan")
    return gt, n_total


def _score_pair(pred: td.graph.BaseGraph, gt: td.graph.BaseGraph, n_total: float) -> dict:
    er = compute_metric(pred, gt, scale=SCALE)
    recall = node_recall(pred, gt) if pred.num_edges() > 0 and pred.num_nodes() > 0 else 0.0
    return per_sample_metrics(er, n_total, recall)


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def selftest(data_dir: Path, datasets: list[str]) -> dict:
    rows = []
    for name in datasets:
        gt, n_total = load_gt(data_dir / f"{name}.geff")
        pred = gt_to_pred_graph(gt)
        row = _score_pair(pred, gt, n_total)
        rows.append(row)
        print(f"  [{name}] edge_jaccard={row['edge_jaccard']:.4f} "
              f"nodes={row['num_pred_nodes']} n_total={n_total:.0f}")
    return summarise(rows)


def score_submission(sub_csv: Path, data_dir: Path, datasets: list[str] | None) -> dict:
    df = pl.read_csv(sub_csv)
    if datasets is None:
        datasets = sorted(df.filter(pl.col("row_type") == "node")["dataset"].unique().to_list())
    rows = []
    for name in datasets:
        gt, n_total = load_gt(data_dir / f"{name}.geff")
        sub = df.filter(pl.col("dataset") == name)
        nodes = sub.filter(pl.col("row_type") == "node").select("node_id", "t", "z", "y", "x")
        edges = sub.filter(pl.col("row_type") == "edge").select("source_id", "target_id")
        pred = build_pred_graph(nodes, edges)
        row = _score_pair(pred, gt, n_total)
        rows.append(row)
        print(f"  [{name}] edge_J={row['edge_jaccard']:.4f} "
              f"adj_edge_J={row['adj_edge_jaccard']:.4f} "
              f"nodes={row['num_pred_nodes']} (T_true={n_total:.0f}) "
              f"recall={row['node_recall']:.3f}")
    return summarise(rows)


def _print_summary(s: dict) -> None:
    print(
        f"\nSCORE={s['score']:.4f}  edge_jaccard={s['edge_jaccard']:.4f}  "
        f"adj_edge_jaccard={s['adj_edge_jaccard']:.4f}  "
        f"division_jaccard={s['division_jaccard']:.4f}  "
        f"node_recall={s['node_recall']:.4f}  (n={s['n']})"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Local scorer for Biohub cell tracking.")
    ap.add_argument("submission", nargs="?", help="submission.csv (omit with --selftest)")
    ap.add_argument("--data-dir", default="data/train", type=Path)
    ap.add_argument("--datasets", nargs="*", default=None, help="dataset names; default = all in submission / dir")
    ap.add_argument("--selftest", action="store_true", help="score GT against itself (sanity check)")
    args = ap.parse_args()

    if args.selftest:
        names = args.datasets or sorted(p.stem for p in args.data_dir.glob("*.geff"))
        print(f"Self-test on {len(names)} dataset(s): expect edge_jaccard ~1.0")
        _print_summary(selftest(args.data_dir, names))
        return

    if not args.submission:
        ap.error("submission.csv required unless --selftest")
    _print_summary(score_submission(Path(args.submission), args.data_dir, args.datasets))


if __name__ == "__main__":
    main()
