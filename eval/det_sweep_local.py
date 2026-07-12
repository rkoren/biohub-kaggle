"""Sweep det_threshold via re-inference on held-out videos — find the detection-recall knee offline.

det_threshold gates NMS during prediction, so testing it needs re-inference (not postproc). Now that
local inference works (~minutes/video), we can find the best threshold without burning submissions.
For each threshold: predict → ILP → score each held-out video; report per-threshold mean + node_ratio.
Saves geffs per threshold to gpu_preds_local_det<thr>/ (a sandbox for postproc/veto sweeps).

    BIOHUB_DET_LIST=0.5,0.90,0.95,0.97,0.99 .venv-track/bin/python eval/det_sweep_local.py
Env: BIOHUB_PACK, BIOHUB_DEVICE (mps|cpu; default mps w/ CPU fallback), BIOHUB_MAX_FRAMES.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
PACK = Path(os.environ.get("BIOHUB_PACK", str(Path.home() / "biohub-tracking-support-pack-50ep-v1")))
sys.path[:0] = [str(PACK / "repo" / "src"), str(PACK / "repo" / "scripts"),
                str(Path(__file__).parent), str(Path(__file__).parent.parent / "pipeline")]

import torch
import tracksdata as td
import polars as pl

import predict_unet_transformer as P
from biohub_tracking.io import save_graph
from local_eval import load_gt, build_pred_graph, _score_pair, summarise

K = td.DEFAULT_ATTR_KEYS
HELDOUT = ["44b6_0b24845f", "44b6_1574802b", "44b6_267148e4", "44b6_3bb3690f", "44b6_40c45f5a"]
DATA = Path(os.environ.get("BIOHUB_DATA_DIR", "data/train"))
THRS = [float(x) for x in os.environ.get("BIOHUB_DET_LIST", "0.5,0.90,0.95,0.97,0.99").split(",")]


def pick_device() -> torch.device:
    want = os.environ.get("BIOHUB_DEVICE")
    if want:
        return torch.device(want)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
    dev = pick_device()
    max_frames = int(os.environ["BIOHUB_MAX_FRAMES"]) if os.environ.get("BIOHUB_MAX_FRAMES") else None
    model, ws, ds = P.load_model(PACK / "weights/unet_transformer/split_0/edge_predictor_best.pth", dev)
    print(f"device={dev} thresholds={THRS}", flush=True)

    summary = []
    for thr in THRS:
        cfg = P.PredictConfig()
        cfg.det_threshold = thr
        outdir = Path(f"gpu_preds_local_det{thr}"); outdir.mkdir(exist_ok=True)
        rows = []
        t_thr = time.time()
        for name in HELDOUT:
            coords, edges = P.predict_video(model, DATA / name, dev, cfg=cfg,
                                            window_size=ws, max_frames=max_frames, downsample=ds)
            g = P.build_graph(coords, edges)
            if getattr(cfg, "use_ilp", True) and g.num_edges() > 0:
                g = td.solvers.ILPSolver(
                    edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
                    appearance_weight=cfg.ilp_appearance_weight,
                    disappearance_weight=cfg.ilp_disappearance_weight,
                    division_weight=cfg.ilp_division_weight).solve(g)
            save_graph(g, outdir / f"{name}.geff")
            nodes, edf = graph_to_dfs(g)
            gt, n_tot = load_gt(DATA / f"{name}.geff")
            rows.append(_score_pair(build_pred_graph(nodes, edf), gt, n_tot))
        s = summarise(rows)
        nr = sum(r["total_node_ratio"] for r in rows) / len(rows)
        summary.append((thr, s))
        print(f"  det={thr:<5} score={s['score']:.4f} adj={s['adj_edge_jaccard']:.4f} "
              f"recall={s['node_recall']:.3f} node_ratio={nr:+.3f} div={s['division_jaccard']:.4f} "
              f"({time.time()-t_thr:.0f}s)", flush=True)

    best = max(summary, key=lambda kv: kv[1]["score"])
    print("\n===== det sweep (raw learned, no postproc) =====", flush=True)
    for thr, s in sorted(summary, key=lambda kv: kv[1]["score"], reverse=True):
        print(f"  {s['score']:.4f}  det={thr}", flush=True)
    print(f"\nbest det={best[0]} → {best[1]['score']:.4f}. Blend uses 0.97. "
          f"geffs saved per threshold for postproc/veto sweeps.", flush=True)


if __name__ == "__main__":
    main()
