# Reference: pilkwang ~0.90 solution ("blend preprocessings" / learned-edge weighted motion assignment)

External reference, copied to build from. **Verified legitimate** (learned multi-model pipeline, internet
off, no leak). Notebook: `biohub-cell-tracking-blend-preprocessings.ipynb` (title in-cell = "Learned-Edge
Weighted Motion Assignment"). User reports it scores **~0.90** on the LB — our next baseline, up from 0.889.

## What makes it 0.90 (vs our 0.889 50ep pipeline) — the deltas

1. **Stronger models (the main lift), all CC0:**
   - a 300–400 epoch UNet detector anchor (preset `unet400_learned_edge_weighted_motion_assignment`)
   - **local association ranker** — `pilkwang/biohub-local-association-ranker-unet300-v1` (CC0): a learned
     edge-scoring model over 22 features (edge_prob, in/out-degree, local 7µm density, motion_dist/gain,
     candidate rank, dz/dy/dx, t_norm…), internal best_score 0.978. Improves the *linking* decision.
2. **`det_threshold = 0.97`** — confirms our own finding (0.99→0.95 gave +0.002 on the LB; they anchor 0.97).
3. **D4 detection TTA** (cell 9) — monkey-patches `predict_unet_transformer.py` to average detection heatmaps
   over all 8 square symmetries (identity + 3 flips + 2 rotations + transpose + anti-transpose). **This is
   exactly the TTA we were about to build — already done, so building it from scratch is redundant.**
4. **Tuned graph surgery** (cell 11, same family as our `pipeline/postprocess.py`, extended):
   `GAP_CLOSE_MAX_GAP=2` (2-frame gaps in the main closer → `OUTPUT_GAP2_RECOVERY=0`),
   `MOTION_RELINK_LEARNED_BONUS=0.90` (more learned-edge influence), tuned safe-div gates.
5. **DeepCenter veto detector** — a *second* detector that validates candidate repair points (gap-close
   midpoints, safe divisions) so only real cells get added. **Fully coded but OFF in this run**
   (`USE_DEEPCENTER_VETO=0`) — this is a probe of the learned-edge axis. **→ untapped lever: try it ON.**

Datasets it attaches (all CC0): `biohub-local-association-ranker-unet300-v1`, `biohub-tracking-support-pack-50ep-v1`,
`pilkwang-public-dataset-for-notebooks-figures`. GPU, internet off.

## How to build from this
1. **Adopt as baseline:** fork this notebook (already lean — viz gated off by `RUN_VISUAL_EDA=0`), attach the
   3 CC0 datasets + competition data, GPU on / internet off, run → confirm ~0.90 for us.
2. **Experiments** (edit cell 4, the env-var config surface):
   - **Enable DeepCenter veto** (`USE_DEEPCENTER_VETO=1`, `DEEPCENTER_GAP_VETO=1`, `DEEPCENTER_SAFE_DIV_VETO=1`)
     — the biggest untapped lever. **Caveat:** confirm the DeepCenter/veto weights are in an attached dataset
     first, else it no-ops.
   - `MOTION_RELINK_LEARNED_BONUS` axis (they anchor 0.90), `det_threshold` fine-tune, `GAP_CLOSE_MAX_GAP`.
3. Our `pipeline/postprocess.py` + `eval/sweep_postproc.py` still apply — tune the graph params locally on
   this model's held-out geffs (via `gpu-start/01` repointed at the unet300/400 weights). See [[reference-pilkwang-0897]].
