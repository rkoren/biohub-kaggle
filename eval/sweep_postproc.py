"""Sweep Stage-B post-processing knobs against ground truth, offline.

Reads predictions (a directory of ``*.geff`` — e.g. the learned model's output, which carries
``edge_prob`` — OR a submission CSV), applies ``pipeline/postprocess.py`` with each candidate
config, and scores every config with the official metric via ``eval/local_eval.py``. Lets us tune
the graph surgery on the LEARNED predictions without burning Kaggle submissions.

    .venv-track/bin/python eval/sweep_postproc.py <preds_dir_or_csv> --data-dir data/train
    .venv-track/bin/python eval/sweep_postproc.py preds/ --configs my_grid.json   # custom override list

Geff predictions preserve ``edge_prob`` so motion-relink's learned bonus engages (our classical CSV
has none → relink degrades to motion+distance, which is still what we tuned).
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "pipeline"))
sys.path.insert(0, str(_HERE))

import polars as pl
import tracksdata as td

import postprocess as P
from local_eval import load_gt, build_pred_graph, _score_pair, summarise

K = td.DEFAULT_ATTR_KEYS


# --------------------------------------------------------------------------- loading
def load_geff(geff_path: Path):
    """One prediction .geff -> (nodes_by_id, raw_edges) in postprocess's dict format (keeps edge_prob)."""
    res = td.graph.IndexedRXGraph.from_geff(str(geff_path))
    g = res[0] if isinstance(res, tuple) else res
    na = g.node_attrs(attr_keys=[K.NODE_ID, K.T, K.Z, K.Y, K.X]).to_dict(as_series=False)
    nodes_by_id = {int(i): {"node_id": int(i), "t": int(t), "z": float(z), "y": float(y), "x": float(x)}
                   for i, t, z, y, x in zip(na[K.NODE_ID], na[K.T], na[K.Z], na[K.Y], na[K.X])}
    raw_edges = []
    if g.num_edges():
        keys = list(g.edge_attr_keys())
        want = [K.EDGE_SOURCE, K.EDGE_TARGET] + (["edge_prob"] if "edge_prob" in keys else [])
        ea = g.edge_attrs(attr_keys=want).to_dict(as_series=False)
        has_prob = "edge_prob" in ea
        for k in range(len(ea[K.EDGE_SOURCE])):
            raw_edges.append({"source_id": int(ea[K.EDGE_SOURCE][k]), "target_id": int(ea[K.EDGE_TARGET][k]),
                              "edge_prob": float(ea["edge_prob"][k]) if has_prob else None})
    return nodes_by_id, raw_edges


def load_predictions(source: Path) -> dict:
    """{dataset: (nodes_by_id, raw_edges)} from a geff dir or a submission CSV."""
    source = Path(source)
    if source.is_dir():
        geffs = sorted(source.glob("*.geff"))
        if not geffs:
            raise FileNotFoundError(f"No .geff files in {source}")
        return {gp.name[:-5] if gp.name.endswith(".geff") else gp.stem: load_geff(gp) for gp in geffs}
    import pandas as pd
    df = pd.read_csv(source)
    return {ds: P.submission_to_graph(grp) for ds, grp in df.groupby("dataset", sort=True)}


# --------------------------------------------------------------------------- scoring
def score_config(preds: dict, gt_dir: Path, overrides: dict | None) -> dict:
    """Apply postprocess (with knob overrides) to every dataset and return the summarised score."""
    g = vars(P)
    saved = {}
    for key, val in (overrides or {}).items():
        if key not in P._KNOBS:
            raise KeyError(f"unknown postprocess knob: {key!r}")
        saved[key] = g[key]; g[key] = val
    try:
        rows = []
        for ds, (nodes_by_id, raw_edges) in preds.items():
            nbi, edges, _ = P.filter_output_graph(copy.deepcopy(nodes_by_id), copy.deepcopy(raw_edges), dataset=ds)
            nodes_df = pl.DataFrame(  # round coords to match the int submission (graph_to_rows), not float
                [{"node_id": n["node_id"], "t": n["t"], "z": max(0, round(float(n["z"]))),
                  "y": max(0, round(float(n["y"]))), "x": max(0, round(float(n["x"])))} for n in nbi.values()],
                schema={"node_id": pl.Int64, "t": pl.Int64, "z": pl.Float64, "y": pl.Float64, "x": pl.Float64})
            edges_df = pl.DataFrame(
                [{"source_id": e["source_id"], "target_id": e["target_id"]} for e in edges],
                schema={"source_id": pl.Int64, "target_id": pl.Int64})
            gt, n_total = load_gt(Path(gt_dir) / f"{ds}.geff")
            rows.append(_score_pair(build_pred_graph(nodes_df, edges_df), gt, n_total))
    finally:
        for key, val in saved.items():
            g[key] = val
    return summarise(rows)


# --------------------------------------------------------------------------- default search
def default_configs() -> list[tuple[str, dict]]:
    """(label, overrides) list: reference points + single-knob variations + a coarse grid."""
    all_off = {k: False for k in P._KNOBS if k.startswith("OUTPUT_") and isinstance(getattr(P, k), bool)}
    cfgs: list[tuple[str, dict]] = [
        ("defaults (current)", {}),
        ("all-postproc-off", all_off),
        ("no-motion-relink", {"OUTPUT_MOTION_RELINK": False}),
        ("no-gap2", {"OUTPUT_GAP2_RECOVERY": False}),
        ("no-safe-div", {"OUTPUT_SAFE_DIVISIONS": False}),
        ("no-short-track-filter", {"OUTPUT_FILTER_SHORT_TRACKS": False}),
        ("no-linefit", {"OUTPUT_LINEFIT_SMOOTH": False}),
        ("div-geometry-filter-on", {"OUTPUT_DIVISION_GEOMETRY_FILTER": True}),
    ]
    for v in (0.25, 0.5, 0.75):
        cfgs.append((f"motion_vel={v}", {"MOTION_RELINK_VELOCITY_WEIGHT": v}))
    for v in (5.0, 6.0, 7.0):
        cfgs.append((f"gap_close_um={v}", {"GAP_CLOSE_UM": v}))
    for v in (4, 6, 8):
        cfgs.append((f"min_track_len={v}", {"OUTPUT_MIN_TRACK_LEN": v}))
    for v in (0.6, 0.8, 1.0):
        cfgs.append((f"linefit_weight={v}", {"OUTPUT_LINEFIT_WEIGHT": v}))
    return cfgs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", help="dir of *.geff OR a submission.csv")
    ap.add_argument("--data-dir", default="data/train", type=Path, help="dir with GT <dataset>.geff")
    ap.add_argument("--configs", default=None, help="JSON: list of {label, overrides} (default: built-in sweep)")
    ap.add_argument("--out", default=None, help="write ranked results JSON here")
    args = ap.parse_args()

    print(f"Loading predictions from {args.predictions} ...", flush=True)
    preds = load_predictions(Path(args.predictions))
    print(f"  {len(preds)} datasets: {sorted(preds)[:6]}{'...' if len(preds) > 6 else ''}", flush=True)

    if args.configs:
        spec = json.loads(Path(args.configs).read_text())
        configs = [(c["label"], c.get("overrides", {})) for c in spec]
    else:
        configs = default_configs()

    results = []
    for label, overrides in configs:
        s = score_config(preds, args.data_dir, overrides)
        results.append({"label": label, "score": s["score"], "edge_jaccard": s["edge_jaccard"],
                        "adj_edge_jaccard": s["adj_edge_jaccard"], "division_jaccard": s["division_jaccard"],
                        "node_recall": s["node_recall"], "overrides": overrides})
        print(f"  {s['score']:.4f}  edge_J={s['edge_jaccard']:.4f} div_J={s['division_jaccard']:.4f}  {label}", flush=True)

    results.sort(key=lambda r: r["score"], reverse=True)
    base = next((r["score"] for r in results if r["label"] == "defaults (current)"), None)
    print("\n===== RANKED =====")
    for r in results:
        delta = f"{r['score'] - base:+.4f}" if base is not None else "   —  "
        print(f"  {r['score']:.4f}  ({delta} vs defaults)  {r['label']}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
