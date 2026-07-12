"""Generate the GPU-start notebook: run the HOST learned baseline (TemporalUNet3D +
SimpleNodeTransformer) on Kaggle GPU and score it honestly with the official metric.

Milestone 1 ONLY: reproduce the host baseline's learned prediction on a held-out TRAIN
video (which has GT) and print the official score. No training, no test-clip scoring, no
submission plumbing — those are later milestones, left as a markdown checklist so we decide
each one with a real number in hand.

Grounded in the actual host inference API (read from the artifacts dataset):
  load_model(weights, device) -> (model, window_size, downsample)
  predict_video(model, ds_path, device, cfg, window_size, unet_batch_size, downsample)
      -> coords[N,4]=(t,z,y,x) full-res, edges=[(src_idx, tgt_idx, prob, dist), ...]
  build_graph(coords, edges) -> tracksdata graph;  ILPSolver(edge_weight=w*EdgeAttr("edge_prob"))
  metric: biohub_tracking.metrics.evaluate(pred, gt, scale, max_distance=7.0) -> per_sample_metrics

Run:  .venv-track/bin/python notebooks/gpu-start/build_gpu_start_nb.py
"""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]

cells: list = []
def md(text: str): cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))
def code(src: str): cells.append(nbf.v4.new_code_cell(src.strip("\n")))


# ---------------------------------------------------------------- header
md(r"""
# GPU-start · pilkwang 50-epoch learned model (TemporalUNet3D + SimpleNodeTransformer)

**Milestone 1 — honest CV of the pilkwang 50ep model (the backbone of the 0.897 pipeline).**
Load the pretrained 50-epoch model, run its detection→linking→ILP inference on the **split-0
held-out TRAIN videos** (which have ground truth), and print the official score. This is the raw
learned-model number *before* the Stage-B graph surgery (`pipeline/postprocess.py`) that lifts it
toward 0.897 — a floor to compare against our classical (dev 0.648 → 0.688 with postproc).

> **Holdout caveat:** these 5 videos are the KT-documented split-0 *test* set. That's a clean holdout
> only if pilkwang's 50ep run used the same split-0 partition (same repo + `split_0` convention → likely,
> but unconfirmed). If they trained on all 199, treat the number as optimistic.

### Attribution & license
Model weights + inference code are from **`pilkwang/biohub-tracking-support-pack-50ep-v1`**
(`repo/` source + 50-epoch `weights/…/edge_predictor_best.pth`), **license CC0-1.0 (public domain)**
— free to build on; crediting pilkwang is courteous, not required. This notebook scores honestly on
held-out TRAIN videos; it does not itself submit.

### Honesty guardrail
Score only on **held-out TRAIN videos** (they ship with GT). Do **not** score against the 4 test
clips — they are byte-identical train copies (a disclosed leak); using their GT to tune or select
is out of bounds. This notebook never touches them.

### Attach to this Kaggle notebook (Settings → Add data)
1. `pilkwang/biohub-tracking-support-pack-50ep-v1` (CC0 — 50ep code + weights + bundled offline wheels)
2. the competition dataset (train `.zarr` + `.geff`)
3. *(later, for offline submission only)* `rkoren/biohub-celltrack-ilp-wheels`

**Requires a GPU accelerator.** Dev with Internet ON is fine here; the offline-wheel install is a
later milestone.
""")

# ---------------------------------------------------------------- paths / env
md("### 1 · Resolve paths & device")
code(r"""
import os, sys
from pathlib import Path
import torch

def _first_dir(cands):
    for c in cands:
        if Path(c).exists():
            return Path(c)
    return None

# Host artifacts (code + weights). Adjust if your mount name differs.
ARTIFACTS = _first_dir([
    "/kaggle/input/biohub-tracking-support-pack-50ep-v1",   # pilkwang 50ep pack (CC0)
    str(Path.home() / "biohub-tracking-support-pack-50ep-v1"),
])
assert ARTIFACTS, "Attach the pilkwang/biohub-tracking-support-pack-50ep-v1 dataset."
REPO     = ARTIFACTS / "repo"
SRC      = REPO / "src"
SCRIPTS  = REPO / "scripts"
WEIGHTS  = ARTIFACTS / "weights" / "unet_transformer" / "split_0" / "edge_predictor_best.pth"
assert WEIGHTS.exists(), f"Missing weights at {WEIGHTS}"

# Competition TRAIN data (has GT). dataspec.py also auto-detects the Kaggle mount.
DATA_DIR = _first_dir([
    "/kaggle/input/competitions/biohub-cell-tracking-during-development/train",
    "/kaggle/input/biohub-cell-tracking-during-development/train",
    str(Path.cwd() / "data" / "train"),
])
assert DATA_DIR, "Point DATA_DIR at the competition train dir (.zarr + .geff)."

# Host scripts import each other by bare name → both dirs must be importable.
for p in (str(SRC), str(SCRIPTS)):
    if p not in sys.path:
        sys.path.insert(0, p)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("artifacts:", ARTIFACTS)
print("data_dir :", DATA_DIR)
print("device   :", DEVICE, "| cuda:", torch.cuda.is_available())
if DEVICE.type != "cuda":
    print("WARNING: no GPU — inference will be very slow. Enable a GPU accelerator.")
""")

# ---------------------------------------------------------------- deps
md("### 2 · Dependencies (dev = Internet ON)\n"
   "The host inference chain needs `tracksdata`, `pyscipopt` (SCIP/ILP), `geff` on top of the "
   "pre-installed `torch/zarr/polars/tqdm`. With Internet ON we pip-install them; the **offline** "
   "swap (attach the wheels dataset, `--no-index --find-links`) is a later milestone for submission.")
code(r"""
import importlib, subprocess, sys
def _need(mod):
    try: importlib.import_module(mod); return False
    except Exception: return True
missing = [m for m in ("tracksdata", "pyscipopt", "geff") if _need(m)]
if missing:
    print("installing:", missing)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
else:
    print("deps present")
""")

# ---------------------------------------------------------------- import host API
md("### 3 · Import the host inference API\n"
   "Import chain verified offline: `predict_unet_transformer` → `train_unet_transformer` → "
   "`augmentations`/`dataspec`/`evaluate` — all resolve from `repo/scripts` with the deps above "
   "(no `monai` needed at import; that's training-only).")
code(r"""
import predict_unet_transformer as P            # scripts/predict_unet_transformer.py
from biohub_tracking.io import open_dataset, save_graph
from biohub_tracking import metrics
import tracksdata as td
import numpy as np, zarr, json

load_model, predict_video, build_graph = P.load_model, P.predict_video, P.build_graph
PredictConfig = P.PredictConfig
print("host inference API imported OK")
""")

# ---------------------------------------------------------------- load model
md("### 4 · Load the pretrained model")
code(r"""
model, window_size, downsample = load_model(WEIGHTS, DEVICE)
cfg = PredictConfig()   # host defaults (pool_kernel_um=5.0, ILP weights edge -1.0·P / app 0.1 / dis 0.1 / div 1.0)
print(f"window_size={window_size} downsample={downsample}")
print("PredictConfig:", cfg)
""")

# ---------------------------------------------------------------- score helper
md("### 5 · Scoring helper (official metric)\n"
   "Same metric family we cross-checked against our `eval/local_eval.py`. `final = adj_edge_jaccard "
   "+ 0.1·division_jaccard`; `n_estimated` = `estimated_number_of_nodes` from the GT geff metadata.")
code(r"""
def estimated_nodes(geff_dir):
    try: attrs = dict(zarr.open_group(str(geff_dir), mode="r").attrs)
    except Exception: return float("nan")
    def search(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "estimated_number_of_nodes":
                    try: return float(v)
                    except Exception: return float("nan")
                r = search(v)
                if r == r: return r
        elif isinstance(o, str):
            try: return search(json.loads(o))
            except Exception: return float("nan")
        return float("nan")
    return search(attrs)

def score_graph(graph, ds_path):
    gt_ds = open_dataset(ds_path, normalize=False, require_tracks=True, load_image=False)
    n_est = estimated_nodes(str(ds_path) + ".geff")
    p = graph.copy()                                  # evaluate() mutates its input
    er = metrics.evaluate(p, gt_ds.tracks, scale=tuple(gt_ds.scale), max_distance=7.0)
    try: nr = metrics.node_recall(p, gt_ds.tracks)
    except Exception: nr = float("nan")
    row = metrics.per_sample_metrics(er, n_est, nr)
    denom = er.division_tp + er.division_fp + er.division_fn
    dj = er.division_tp / denom if denom > 0 else 0.0
    adj = row["adj_edge_jaccard"]; adj = adj if adj == adj else 0.0
    return dict(n_est=n_est, pred_nodes=er.num_pred_nodes, node_ratio=row["total_node_ratio"],
                node_recall=nr, etp=er.edge_tp, efp=er.edge_fp, efn=er.edge_fn,
                edge_j=row["edge_jaccard"], adj=adj, div_j=dj,
                final=adj + metrics.SCORE_DIVISION_WEIGHT * dj)
""")

# ---------------------------------------------------------------- held-out sweep
md("### 6 · Inference + score over the split-0 HELD-OUT set → honest mean\n"
   "The checkpoint is `split_0`, so these 5 videos (the KT's fold-0 *test* set) are genuinely unseen "
   "by it — the correct apples-to-apples number vs the KT's reported **0.6637** and our classical dev "
   "**0.648**. Each: `predict_video → build_graph → probability-weighted ILP → save → score`.")
code(r"""
# split_0 held-out (fold-0 test) videos, per the KT dataset-EDA doc. All present in DATA_DIR.
HELDOUT = ["44b6_0b24845f", "44b6_1574802b", "44b6_267148e4", "44b6_3bb3690f", "44b6_40c45f5a"]
OUT_DIR = (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()) / "gpu_preds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []
for name in HELDOUT:
    ds_path = DATA_DIR / name                         # open_dataset appends .zarr / .geff
    coords, edges = predict_video(model, ds_path, DEVICE, cfg=cfg,
                                  window_size=window_size, downsample=downsample)
    graph = build_graph(coords, edges)
    if getattr(cfg, "use_ilp", True) and graph.num_edges() > 0:
        graph = td.solvers.ILPSolver(
            edge_weight=cfg.ilp_edge_weight * td.EdgeAttr("edge_prob"),
            appearance_weight=cfg.ilp_appearance_weight,
            disappearance_weight=cfg.ilp_disappearance_weight,
            division_weight=cfg.ilp_division_weight,
        ).solve(graph)
    save_graph(graph, OUT_DIR / f"{name}.geff")
    r = score_graph(graph, ds_path); r["video"] = name; rows.append(r)
    print(f"{name}: recall={r['node_recall']:.3f} node_ratio={r['node_ratio']:+.3f} "
          f"edge_J={r['edge_j']:.4f} adj={r['adj']:.4f} div_J={r['div_j']:.3f} FINAL={r['final']:.4f}")

mean_final = sum(r["final"] for r in rows) / len(rows)
mean_adj   = sum(r["adj"]   for r in rows) / len(rows)
print("\n" + "=" * 60)
print(f"HELD-OUT MEAN over {len(rows)} videos:  FINAL={mean_final:.4f}  adj_edge_J={mean_adj:.4f}")
print(f"compare → KT host fold-0 = 0.6637 | our classical dev = 0.648")
print("=" * 60)
""")

# ---------------------------------------------------------------- next milestones (NOT implemented)
md(r"""
### Next milestones — decide each with a real number in hand (NOT implemented here)

- **M2 · Sweep a few TRAIN videos** — run cell 5–6 over ~5 held-out videos, mean the score. That's
  our honest learned-baseline number to compare against classical dev (0.648) and the KT's fold-0
  (0.6637). Adopt the KT's exact fold-0 set for a directly comparable yardstick.
- **M3 · Detection threshold / loss** — the KT shows our-checkpoint (Gaussian-CenterNet) wants a
  different `det_threshold` than the host single-voxel one. Cheap knob sweep, no retraining.
- **M4 · Fine-tune / retrain** — only if M2 says the ceiling is worth it. Needs `monai` + the training
  scripts; budget against the 12h limit.
- **M5 · Offline submission** — swap deps to the wheel dataset (`--no-index`), run over the 4 **test**
  videos, and convert `.geff` → `submission.csv`. **Open question for us (see chat):** score-locally
  (fast, no LB) vs build the geff→CSV submission path (real LB, more plumbing) first.
""")

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "gpu-start" / "01_host_baseline_infer.ipynb"
nbf.write(nb, out)
print("wrote", out)
