"""Bridge: competition submission CSV -> prediction ``.geff`` for CellTrack Studio.

CellTrack Studio (an external viewer, ~/celltrack-studio) loads predictions as a
GEFF directory whose graph uses tracksdata's DEFAULT_ATTR_KEYS and *voxel* coords
(matching how the GT ``.geff`` stores them). Our pipeline emits a competition CSV
(node/edge rows). This turns one dataset's rows in that CSV into ``<name>.geff``.

    python scripts/submission_to_geff.py submissions/dev_best.csv \
        --out predictions --datasets 6bba_09961292            # one, or omit for all

Then launch the studio against our repo's train data (which carries GT + image):

    ~/celltrack-studio/.venv/bin/celltrack-studio \
        --name 6bba_09961292 --data-dir data/train --pred predictions --downsample 1,4,4
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import polars as pl
import tracksdata as td

K = td.DEFAULT_ATTR_KEYS


def build_graph(nodes: pd.DataFrame, edges: pd.DataFrame) -> td.graph.IndexedRXGraph:
    """One dataset's node/edge rows -> IndexedRXGraph with default keys, voxel coords."""
    g = td.graph.IndexedRXGraph()  # only `t` is a built-in key; declare the spatial ones
    for key in (K.Z, K.Y, K.X):
        g.add_node_attr_key(key, pl.Int64, 0)

    # Preserve the CSV's node_id so edge source/target refs stay valid.
    g.bulk_add_nodes(
        [{K.T: int(r.t), K.Z: int(r.z), K.Y: int(r.y), K.X: int(r.x)} for r in nodes.itertuples()],
        indices=[int(n) for n in nodes.node_id],
    )
    if len(edges):
        g.bulk_add_edges([
            {K.EDGE_SOURCE: int(r.source_id), K.EDGE_TARGET: int(r.target_id)}
            for r in edges.itertuples()
        ])
    return g


def export(csv_path: Path, out_dir: Path, names: list[str] | None) -> list[str]:
    sub = pd.read_csv(csv_path)
    all_names = sorted(sub["dataset"].unique())
    names = names or all_names
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[str] = []
    for name in names:
        if name not in all_names:
            print(f"  skip {name}: not in {csv_path.name}")
            continue
        grp = sub[sub.dataset == name]
        nodes = grp[grp.row_type == "node"]
        edges = grp[grp.row_type == "edge"]
        g = build_graph(nodes, edges)
        dst = out_dir / f"{name}.geff"
        if dst.exists():
            shutil.rmtree(dst)
        g.to_geff(str(dst), overwrite=True)
        print(f"  wrote {dst}  ({g.num_nodes()} nodes / {g.num_edges()} edges)")
        written.append(name)
    return written


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", type=Path, help="submission CSV (e.g. submissions/dev_best.csv)")
    ap.add_argument("--out", type=Path, default=Path("predictions"), help="output dir for <name>.geff")
    ap.add_argument("--datasets", nargs="*", default=None, help="dataset names (default: all in CSV)")
    args = ap.parse_args()

    written = export(args.csv, args.out, args.datasets)
    print(f"\nWrote {len(written)} prediction geff(s) to {args.out}/")


if __name__ == "__main__":
    main()
