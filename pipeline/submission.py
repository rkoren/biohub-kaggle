"""Submission assembly and validation (node/edge rows → competition CSV).

Required columns: id,dataset,row_type,node_id,t,z,y,x,source_id,target_id
`id` is a throwaway consecutive index. node rows fill t/z/y/x (integer voxels)
with source/target = -1; edge rows fill source_id/target_id with the rest = -1.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

COLUMNS = ["dataset", "row_type", "node_id", "t", "z", "y", "x", "source_id", "target_id"]


def node_row(dataset: str, node_id: int, t: int, zyx: Sequence[int]) -> dict:
    z, y, x = (int(v) for v in zyx)
    return {"dataset": dataset, "row_type": "node", "node_id": int(node_id),
            "t": int(t), "z": z, "y": y, "x": x, "source_id": -1, "target_id": -1}


def edge_row(dataset: str, source_id: int, target_id: int) -> dict:
    return {"dataset": dataset, "row_type": "edge", "node_id": -1,
            "t": -1, "z": -1, "y": -1, "x": -1,
            "source_id": int(source_id), "target_id": int(target_id)}


def assemble(rows: list[dict]) -> pd.DataFrame:
    """Rows → DataFrame with the exact competition columns + explicit `id` index."""
    sub = pd.DataFrame(rows)[COLUMNS]
    sub.insert(0, "id", range(len(sub)))
    return sub


def save(sub: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, index=False)
    return path


def validate(sub: pd.DataFrame, expected_datasets: set[str] | None = None) -> None:
    """Catch the common Kaggle submission failures before committing."""
    cols = [c for c in sub.columns if c != "id"]
    assert cols == COLUMNS, f"Wrong columns: {list(sub.columns)}"
    assert len(sub) > 0, "Empty submission"
    assert set(sub["row_type"].unique()).issubset({"node", "edge"}), "Invalid row_type"

    nodes = sub[sub.row_type == "node"]
    edges = sub[sub.row_type == "edge"]
    assert (nodes[["node_id", "t", "z", "y", "x"]] >= 0).all().all(), "Node fields must be >= 0"
    assert (nodes[["source_id", "target_id"]] == -1).all().all(), "Node source/target must be -1"
    assert (edges[["node_id", "t", "z", "y", "x"]] == -1).all().all(), "Edge node/coords must be -1"
    assert (edges[["source_id", "target_id"]] >= 0).all().all(), "Edge refs must be >= 0"

    if expected_datasets is not None:
        missing = expected_datasets - set(sub["dataset"].unique())
        assert not missing, f"Missing datasets in submission: {sorted(missing)}"

    for ds, grp in sub.groupby("dataset"):
        node_ids = set(grp.loc[grp.row_type == "node", "node_id"].astype(int))
        e = grp[grp.row_type == "edge"]
        assert node_ids, f"{ds}: no node rows"
        assert e["source_id"].astype(int).isin(node_ids).all(), f"{ds}: dangling source_id"
        assert e["target_id"].astype(int).isin(node_ids).all(), f"{ds}: dangling target_id"
