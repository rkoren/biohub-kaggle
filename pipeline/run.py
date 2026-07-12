"""Orchestrate the classical pipeline over datasets → submission rows.

`run_dataset` streams a dataset's volumes, detects + links + prunes, and returns
node/edge rows. `run_datasets` loops and assembles a submission DataFrame.
CLI: produce a submission.csv for a directory of `.zarr` datasets.
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .detect import detect_cells
from .io_zarr import VolumeSeries, list_dataset_names
from .link import link_frames, prune_isolated
from . import submission as sub


def run_dataset(zarr_path: str | Path, dataset: str, cfg: PipelineConfig,
                verbose: bool = True) -> tuple[list[dict], dict]:
    """Detect + link + prune one dataset. Returns (rows, stats)."""
    series = VolumeSeries(zarr_path)
    T, Z, Y, X = series.shape
    t0 = time.time()

    node_rows: list[dict] = []
    edge_rows: list[dict] = []
    node_scores: dict[int, float] = {}
    prev_ids: list[int] = []
    prev_xyz = np.empty((0, 3), dtype=np.int32)
    prev_count: int | None = None
    next_id = 1
    frame_counts: list[int] = []
    division_est = 0

    for t in range(T):
        vol = series.volume(t)
        coords, scores = detect_cells(vol, cfg, prev_count=prev_count)
        del vol
        gc.collect()

        if len(coords):  # stable ordering for reproducibility
            order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
            coords, scores = coords[order], scores[order]

        curr_ids = list(range(next_id, next_id + len(coords)))
        next_id += len(coords)
        for nid, zyx, sc in zip(curr_ids, coords, scores):
            node_rows.append(sub.node_row(dataset, nid, t, zyx))
            node_scores[int(nid)] = float(sc)

        if t > 0 and cfg.link_method == "greedy":
            links = link_frames(prev_ids, prev_xyz, curr_ids, coords, cfg)
            for s, u in links:
                edge_rows.append(sub.edge_row(dataset, s, u))
            src_counts = Counter(s for s, _ in links)
            division_est += sum(1 for c in src_counts.values() if c >= 2)

        prev_ids, prev_xyz, prev_count = curr_ids, coords, len(coords)
        frame_counts.append(len(coords))
        if verbose and ((t + 1) % 20 == 0 or t == T - 1):
            print(f"    frame {t+1:>3}/{T}: nodes={len(coords):>4} edges={len(edge_rows):>5}")

    if cfg.link_method == "ilp":
        from .link_ilp import ilp_link
        edge_rows = ilp_link(node_rows, cfg, n_neighbors=cfg.link_n_neighbors,
                             delta_t=cfg.link_delta_t,
                             appearance=cfg.ilp_appearance, disappearance=cfg.ilp_disappearance,
                             division=cfg.ilp_division)

    before = (len(node_rows), len(edge_rows))
    node_rows, edge_rows, pstats = prune_isolated(node_rows, edge_rows, node_scores, cfg)

    stats = {
        "dataset": dataset, "shape": series.shape,
        "nodes_before": before[0], "edges_before": before[1],
        "nodes": len(node_rows), "edges": len(edge_rows),
        "removed_isolated": pstats["removed_isolated"],
        "division_edges_est": division_est,
        "count_min": int(min(frame_counts)) if frame_counts else 0,
        "count_max": int(max(frame_counts)) if frame_counts else 0,
        "count_mean": round(float(np.mean(frame_counts)), 1) if frame_counts else 0.0,
        "seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(f"  [{dataset}] {stats['seconds']/60:.1f} min | nodes {before[0]}->{len(node_rows)} "
              f"edges {before[1]}->{len(edge_rows)} isolated_removed={pstats['removed_isolated']}")
    return node_rows + edge_rows, stats


def run_datasets(data_dir: str | Path, names: list[str], cfg: PipelineConfig,
                 verbose: bool = True):
    """Run every dataset in ``names`` under ``data_dir``; returns (submission_df, stats_list)."""
    data_dir = Path(data_dir)
    all_rows: list[dict] = []
    all_stats: list[dict] = []
    for i, name in enumerate(names, 1):
        if verbose:
            print(f"[{i}/{len(names)}] {name}")
        rows, stats = run_dataset(data_dir / f"{name}.zarr", name, cfg, verbose=verbose)
        all_rows.extend(rows)
        all_stats.append(stats)
    return sub.assemble(all_rows), all_stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the classical cell-tracking pipeline.")
    ap.add_argument("--data-dir", default="data/test", type=Path)
    ap.add_argument("--menu", default="menu.yaml")
    ap.add_argument("--out", default="submissions/submission.csv", type=Path)
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="dataset names; default = all .zarr in --data-dir")
    ap.add_argument("--subset-json", default=None,
                    help="JSON file with a 'dev_subset' list of dataset names")
    args = ap.parse_args()

    cfg = PipelineConfig.load(args.menu)
    if args.subset_json:
        names = json.loads(Path(args.subset_json).read_text())["dev_subset"]
    else:
        names = args.datasets or list_dataset_names(args.data_dir)

    print(f"Running {len(names)} dataset(s) from {args.data_dir}")
    submission_df, stats = run_datasets(args.data_dir, names, cfg)
    sub.validate(submission_df, expected_datasets=set(names))
    out = sub.save(submission_df, args.out)
    n_nodes = int((submission_df.row_type == "node").sum())
    n_edges = int((submission_df.row_type == "edge").sum())
    print(f"\nWrote {out}: {len(submission_df):,} rows ({n_nodes:,} nodes, {n_edges:,} edges)")


if __name__ == "__main__":
    main()
