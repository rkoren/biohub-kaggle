# Reference: pilkwang 0.897 learned pipeline (forked notebook + extracted source)

**External reference material — copied here to build from. Verified legitimate (no leak exploit).**

- **Notebook**: `fork-full-visual-pipeline-and-animation.ipynb` — our fork (`rkoren/...`) of the
  "0.897 Baseline With 4D Interactive Visual Walkthrough". Original config = the "0.897 baseline";
  visualization adaptation by Kaggle `boristown`.
- **Support pack**: `pilkwang/biohub-tracking-support-pack-50ep-v1` — **license CC0-1.0** (public
  domain; credit courteous, not required). Bundles `repo/` (UNet+transformer+ILP source, package
  `biohub_tracking`), `weights/unet_transformer/split_0/` (`checkpoint_last.pth` 25MB = 50-epoch
  model, `edge_predictor_best.pth`, `config.json`), **`wheels/`** (offline deps — supersedes our
  hand-built `rkoren/biohub-celltrack-ilp-wheels`), and `source_scripts/train_full_frame_center_detector.py`.
- **Legitimacy check**: internet OFF, GPU T4, runs a trained checkpoint over the test `.zarr`; all
  `.geff` reads are the notebook's OWN predictions or the `estimated_number_of_nodes` metric field.
  No test-answer lookup. Safe to build on.
- Extracted code cells are in `cells/` for study (the animation outputs are stripped).

## Method (how 0.897 is reached)

**Stage A — learned detect + link (`scripts/predict_unet_transformer.py`, run as subprocess):**
retrained 50-epoch TemporalUNet3D detection heatmap → high-threshold NMS → SimpleNodeTransformer
edge probabilities → global ILP (`edge_weight = w·edge_prob`, appearance/disappearance 0.1, division 1.0).
Same architecture as the host baseline (`docs/kt/Host Baseline…`), just trained longer + high threshold.

**Stage B — classical graph surgery on the predicted `.geff` (cell 16, ~1036 lines):** the score-lifter
from the ~0.66 raw baseline toward 0.897. Ordered repairs, each a `cell_16.py` function:
| step | fn | tuned knob (env-overridable → sweepable) |
|---|---|---|
| high-threshold detection | (predict cfg) | `DET_THRESHOLD = 0.99` |
| motion relinking | `motion_relink_edges` | `VELOCITY_WEIGHT=0.5`, `LEARNED_BONUS=0.75` (cost = motion + 0.05·raw − 0.75·prob) |
| 1-frame gap closing | `close_single_frame_gaps` | `GAP_CLOSE_MAX_GAP=1`, `GAP_CLOSE_UM=6.0` |
| 2-frame gap recovery | `recover_strict_gap2` | `GAP2_MAX_LINKS_FRAC=0.0045`, `_ABS=180` |
| safe division recovery | `add_safe_divisions_postlink` | `DIV_PARENT_MAX_UM=10.5`, `DIV_SISTER_MAX_UM=8.0`, `SAFE_DIV_SISTER=7.2` |
| short-track filtering | `filter_short_track_components` | `OUTPUT_MIN_TRACK_LEN=6` (keep if ≥6 frames OR has division) |
| line-fit smoothing | `linefit_smooth_output_graph` | `OUTPUT_LINEFIT_WEIGHT=0.8`, `WINDOW=2` |

**Auxiliary (optional):** a separately-trained full-frame center detector as a conservative
node-rescue prior (`weights/full_frame_center/` + `gate_*` calibration files) — present only if the
pack ships it (this "50ep-v1" may not).

## How to build from this
1. **Bank the score**: run the fork as-is (attach the CC0 support pack + competition data, GPU,
   internet OFF) → ~0.897 submission. Fastest leapfrog over our classical (~0.58–0.70) and the
   host-default GPU-start (~0.66).
2. **Port Stage B into our pipeline** — the graph-surgery steps are classical and transfer onto BOTH
   the learned geffs and our own classical predictions (`pipeline/`). This is the reusable IP.
3. **Repoint our GPU-start notebook** at the 50ep pack instead of the host-default weights.
