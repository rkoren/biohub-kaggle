"""Run the learned model locally on held-out TRAIN videos → geffs, score raw and post-processed.

Establishes the offline learned-CV loop (no Kaggle, no submissions). Predicts with the 50ep detector
(the same weights the ~0.90 blend uses), saves per-video geffs (with edge_prob), and scores each both
raw and with our ported Stage-B graph surgery (`pipeline/postprocess.py`). The saved geffs are then a
sandbox for `eval/sweep_postproc.py` to tune knobs on the LEARNED predictions.

    .venv-track/bin/python eval/learned_cv_local.py            # default: 5 fold-0 held-out videos
Env: BIOHUB_PACK (pack dir), BIOHUB_DEVICE (cpu|mps), BIOHUB_MAX_FRAMES (debug).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

PACK = Path(os.environ.get("BIOHUB_PACK", str(Path.home() / "biohub-tracking-support-pack-50ep-v1")))
sys.path[:0] = [str(PACK / "repo" / "src"), str(PACK / "repo" / "scripts"),
                str(Path(__file__).parent), str(Path(__file__).parent.parent / "pipeline")]

import torch
import tracksdata as td
import polars as pl

import predict_unet_transformer as P
from biohub_tracking.io import open_dataset, save_graph  # noqa: F401
from local_eval import load_gt, build_pred_graph, _score_pair, summarise
import sweep_postproc as SW

K = td.DEFAULT_ATTR_KEYS
HELDOUT = ["44b6_0b24845f", "44b6_1574802b", "44b6_267148e4", "44b6_3bb3690f", "44b6_40c45f5a"]
DATA_DIR = Path("data/train")
OUT_DIR = Path("gpu_preds_local"); OUT_DIR.mkdir(exist_ok=True)


def graph_to_dfs(g):
    na = g.node_attrs(attr_keys=[K.NODE_ID, "t", "z", "y", "x"]).to_dict(as_series=False)
    nodes = pl.DataFrame({"node_id": na[K.NODE_ID], "t": na["t"], "z": na["z"], "y": na["y"], "x": na["x"]})
    if g.num_edges():
        ea = g.edge_attrs(attr_keys=[K.EDGE_SOURCE, K.EDGE_TARGET]).to_dict(as_series=False)
        edges = pl.DataFrame({"source_id": ea[K.EDGE_SOURCE], "target_id": ea[K.EDGE_TARGET]})
    else:
        edges = pl.DataFrame({"source_id": [], "target_id": []}, schema={"source_id": pl.Int64, "target_id": pl.Int64})
    return nodes, edges


def main() -> None:
    dev = torch.device(os.environ.get("BIOHUB_DEVICE", "cpu"))
    max_frames = int(os.environ["BIOHUB_MAX_FRAMES"]) if os.environ.get("BIOHUB_MAX_FRAMES") else None
    model, ws, ds = P.load_model(PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth", dev)
    cfg = P.PredictConfig()
    print(f"device={dev} window={ws} downsample={ds} | held-out={len(HELDOUT)} videos", flush=True)

    raw_rows = []
    for name in HELDOUT:
        t0 = time.time()
        coords, edges = P.predict_video(model, DATA_DIR / name, dev, cfg=cfg,
                                        window_size=ws, max_frames=max_frames, downsample=ds)
        g = P.build_graph(coords, edges)
        if getattr(cfg, "use_ilp", True) and g.num_edges() > 0:
            g = td.solvers.ILPSolver(
                edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                appearance_weight=cfg.ilp_appearance_weight,
                disappearance_weight=cfg.ilp_disappearance_weight,
                division_weight=cfg.ilp_division_weight).solve(g)
        save_graph(g, OUT_DIR / f"{name}.geff")
        nodes, edf = graph_to_dfs(g)
        gt, n_tot = load_gt(DATA_DIR / f"{name}.geff")
        row = _score_pair(build_pred_graph(nodes, edf), gt, n_tot)
        raw_rows.append(row)
        print(f"  [{name}] {time.time()-t0:.0f}s nodes={g.num_nodes()} edges={g.num_edges()} "
              f"edge_J={row['edge_jaccard']:.4f} adj={row['adj_edge_jaccard']:.4f} recall={row['node_recall']:.3f}",
              flush=True)

    raw = summarise(raw_rows)
    print(f"\nRAW learned (50ep+ILP, no postproc): score={raw['score']:.4f} adj={raw['adj_edge_jaccard']:.4f} "
          f"div={raw['division_jaccard']:.4f}", flush=True)

    # + our ported Stage-B graph surgery on the saved learned geffs
    preds = SW.load_predictions(OUT_DIR)
    pp = SW.score_config(preds, DATA_DIR, overrides=None)
    print(f"+ our postprocess.py defaults:      score={pp['score']:.4f} adj={pp['adj_edge_jaccard']:.4f} "
          f"div={pp['division_jaccard']:.4f}", flush=True)
    print(f"\nSaved {len(HELDOUT)} learned geffs to {OUT_DIR}/ — tune with:")
    print(f"  .venv-track/bin/python eval/sweep_postproc.py {OUT_DIR}/ --data-dir data/train "
          f"--configs eval/grids/learned_postproc.json", flush=True)


if __name__ == "__main__":
    main()
