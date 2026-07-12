"""Sweep the DeepCenter veto locally on learned geffs — does gating repairs by the 2nd detector help?

Runs our Stage-B postproc on local learned geffs (`gpu_preds_local/`) with the DeepCenter veto ON at
several thresholds vs OFF, scoring each with the official metric. Lets us decide whether the veto (and
which thresholds) helps BEFORE spending Kaggle submissions.

    .venv-track/bin/python eval/veto_harness.py

Caveat: our local geffs come from the simple predict path (no D4 TTA, no association ranker), so this
tests the veto MECHANISM (reject bad gap-close/safe-div repairs), not the exact blend numbers. Trust the
sign/relative effect, not the absolute.
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parent / "pipeline"), str(_HERE)]

import postprocess as P
import deepcenter as DC
import sweep_postproc as SW

import os
PREDS = Path(os.environ.get("BIOHUB_PREDS_DIR", str(_HERE.parent / "gpu_preds_local")))
DATA = Path(os.environ.get("BIOHUB_DATA_DIR", str(_HERE.parent / "data" / "train")))
DC_CKPT = Path(os.environ.get("BIOHUB_DEEPCENTER_CKPT",
                              str(Path.home() / "biohub-deepcenter-unet3d-center-prior-v1" / "weights" / "full_frame_center" / "best.pt")))
DC_DEVICE = os.environ.get("BIOHUB_DEVICE", "cpu")

# (label, gap_threshold, div_threshold) — None thresholds => veto OFF
CONFIGS = [
    ("veto OFF (baseline)", None, None),
    ("veto gap=0.10 div=0.12 (blend default)", 0.10, 0.12),
    ("veto loose gap=0.05 div=0.06", 0.05, 0.06),
    ("veto strict gap=0.20 div=0.24", 0.20, 0.24),
    ("veto gap=0.15 div=0.15", 0.15, 0.15),
]


def main() -> None:
    assert DC_CKPT.exists(), f"DeepCenter checkpoint not found: {DC_CKPT}"
    preds = SW.load_predictions(PREDS)
    print(f"loaded {len(preds)} learned geffs; loading DeepCenter model ...", flush=True)
    bundle = DC.load(DC_CKPT, device=DC_DEVICE)

    base = None
    for label, gap_thr, div_thr in CONFIGS:
        if gap_thr is None:
            P.VETO_FN = None
        else:
            scorer = DC.PointScorer(bundle, data_dir=DATA, gap_threshold=gap_thr, div_threshold=div_thr)
            P.VETO_FN = scorer.accept
        s = SW.score_config(preds, DATA, overrides=None)
        P.VETO_FN = None
        if base is None:
            base = s["score"]
        d = f"{s['score']-base:+.4f}"
        print(f"  {s['score']:.4f}  ({d})  edge_J={s['edge_jaccard']:.4f} div_J={s['division_jaccard']:.4f}  {label}",
              flush=True)


if __name__ == "__main__":
    main()
