# Learned Detector Plan (DETECT-LEARNED)

Goal: push past the classical baseline (local **0.534** / LB **≈0.581**) by lifting the thing that
caps it — **detection recall + centroid localization on dense volumes** (overall node_recall 0.62,
collapsing to 0.23–0.29 on the hardest datasets). The 7 µm match tolerance makes localization
near-binary, and z is the binding axis (1.625 µm/vox ⇒ 7 µm ≈ 4.3 z-voxels).

*Revised after a Fable-5 plan review (2026-07-08) — the review flagged a false-negative-prone gate,
a missing cheap classical step, and several leakage/compute/ops risks. All folded in below.*

## Why this is high-leverage
- **Pretrained models + external data are allowed offline** (attach as a **private** Kaggle dataset —
  don't reveal the approach pre-deadline).
- The **organizers ship a full learned baseline**: `TemporalUNet3D` detector + `SimpleNodeTransformer`
  linker, trained jointly on THIS data with **sparse supervision**, 5-fold `dataset_splits.json`, plus a
  public checkpoint (only `--epochs 3`, undertrained) + inference notebook.
- Our **local metric harness is trustworthy**, so every step is measurable offline before we submit.

## Metric guardrails (verified against the vendored code)
- **Over-prediction is NOT free** — it's doubly penalized: FP edges sit in the edge-Jaccard denominator
  AND `J_adj = J·(1 − 0.1·(N_pred − T_true)/T_true)` penalizes exceeding the budget. So the dense-set
  fix is **reallocation within budget + better localization**, NOT flooding dense regions.
- **Off-GT predictions are invisible** to edge-Jaccard (only the node-count scalar sees them).
- Node budget ≈ 0.9–1.0·T_true. **Divisions negligible** (0.1 weight; 151 events / 199 sets) — ignore.
- **No leakage:** train/val must respect embryo + movie integrity (see Leakage below).

## Compute reality
- torch + 3D conv training needs a GPU; Mac (MPS/CPU) is too slow → **train on Kaggle GPU notebooks**
  (T4/P100, ~30h/wk, 12h/session; training notebooks may use internet). Local `.venv-track` (CPU) stays
  the **eval/orchestration** brain.
- The baseline uses **gradient checkpointing** ⇒ already memory-bound, trading compute for memory (slow).
  "epochs 50 on 199 volumes" is likely **weeks** of wall-clock under the quota. **Do not target a fixed
  epoch count** — measure **sec/epoch in Phase 0**, extrapolate against 30h/wk, then set a budget-derived
  target (likely 10–15, and/or patch/crop sampling, and/or a training subset).
- Needs `tracking-cellmot` (torch, git-only, royerlab). **It must also be VENDORED for offline inference**
  (the internet-off submission notebook can't pip-install it) — same pattern as `eval/_vendor/`. Plan this
  or it's a submission-day blocker.

## Leakage (must resolve before trusting any val score)
- **Only 2 embryos:** `44b6` (71 sets) + `6bba` (128 sets). Test is `44b6`×2 + `6bba`×2.
- **Confirm from the competition data description whether the 4 test volumes are a held-out embryo or
  the same two.** If a novel embryo → cross-fold val (both embryos always in train) *overestimates*;
  add a **leave-one-embryo-out** run as a pessimistic bound.
- **Verify the 5-fold split respects movie integrity** — if sub-volumes are spatial/temporal crops of the
  same movie split across train/val, cells leak within-fold even in a "proper" fold.
- The public checkpoint's training set is **unknown** → its "held-out" score is likely contaminated.
  Treat its aggregate as **uninformative** (see Phase 0).

## ★ STRATEGY REVISION (2026-07-08) — reorder: LINKING first, precision not recall
Corroborated by BOTH a methods-research pass AND our own count-controlled sweep. This supersedes the
phase ordering below.

**Empirical (dev-subset count sweep):** detection tuning alone can't beat the 0.534 baseline. At finer
`xy_ds=2`, node_recall RISES (0.62→0.72) but `edge_jaccard` FALLS (0.506→0.485) and score drops —
because the **greedy per-frame Hungarian linker makes more edge errors as detections get denser.**
⇒ **Linking is the current bottleneck, not detection.**

**Research (organizers' method = Ultrack):** Ultrack (Bragantini et al., Nature Methods 2025, royerlab)
is the reference tracker for THIS data (it built the Zebrahub lineages). It's **segmentation-hypothesis
+ global ILP** (jointly selects segments AND links, native ≤2-daughter division + appear/disappear
flow). Offline-runnable: ILP falls back to the open-source **CBC solver** when no Gurobi license (Gurobi
activation needs internet; CBC does not) — chunk in time/space to bound solve time; our 4 small volumes
are well within scale. `tracksdata.solvers.ILPSolver` (already installed) is the same engine.

**Corrected metric read:** node penalty uses `graph.num_nodes()` = ALL predicted nodes GLOBALLY (not
masked to GT). Over-detection anywhere shrinks score. ⇒ target `T_pred ≈ T_true` (dense count),
well-localized — **precision/count-calibration, NOT max recall.** (My earlier "recall-friendly loss" was
wrong; verified by metric code + perturbation test + the sweep.)

**Revised order (do in this sequence):**
1. **Global ILP linking on our EXISTING classical detections** (`LINK-BETTER`) — highest leverage, no
   GPU, no new model; directly lifts edge_J + the (currently 0) division term. Validate CBC solve-time
   within the 12h offline budget. *This is the immediate next action.*
2. **Precise, count-calibrated pretrained detector → Ultrack/ILP:** StarDist-3D (crispest centroids for
   the 7µm tolerance; tune `prob_thresh` so total nodes ≈ T_true) or Cellpose-SAM (better dim-nucleus
   recall in dense regions; robust to z-anisotropy). Optionally Cellpose3 denoise first; slight z-upscaling
   toward isotropy. Consider **full Ultrack end-to-end** (foreground+contour → watershed hierarchy → ILP).
3. **Explicit global node-budget calibration** to ~T_true (cheap, model-independent score lever).
4. **Only if (1)–(3) plateau:** retrain the TemporalUNet3D / linajea-style cell-indicator head with a
   **precision/count-calibrated sparse-masked Gaussian-regression** loss (Malin-Mayor et al. 2022 recipe),
   NOT recall-friendly.
5. **Divisions last** (0.1 weight; lean on ILP flow + ±1-timepoint tolerance).

**Community frontier:** public notebooks are already moving to *learned graph linking with gap-recovery*
(handling missed detections across frames) — reinforces that linking is where the points are.

**Open gap — GATES investment level:** the live LB score *distribution* / top-team threshold is unknown
(Kaggle LB is JS-rendered; browser extension not connected). Is 0.58 bottom-quartile or already
respectable? Reilly to check manually, or reconnect the Chrome extension and I'll read it. **Also note the
integrity leak makes the *public* LB unreliable regardless** — weight private-test generalization.

## ⚠ DECISIVE FINDING (2026-07-08) — the organizers' detector is recall-capped BY DESIGN
Verified in `scripts/train_unet_transformer.py::compute_detection_loss`: the detection head is trained
with `target = zeros`, GT node voxels → 1, **all other voxels → 0 (negatives, `neg_weight=0.1`)**.
Because GT is sparse, the real *unannotated* cells (the majority) are trained as background. Their own
inference default is `--det-threshold 0.99` with the comment *"the detector is poorly calibrated because
the ground truth is sparse... a high threshold keeps precision up."*
- ⇒ **The public checkpoint's detection recall is structurally capped** — a naive A/B against our classical
  detector (recall 0.62, no negative bias) would likely LOSE, a false-negative for the architecture.
- ⇒ **The real EV is a recall-friendly detection loss**: treat unannotated voxels as *ignore/uncertain*,
  not negative (positive-unlabeled learning, or only penalize negatives far from any GT peak). This is
  the key modification we own; it directly attacks the recall ceiling the classical baseline can't cross.
- ⇒ Sequencing: Phase −1 (cheap classical) first; then **retrain the detection head** rather than trusting
  the stock checkpoint's detector.

## Phased plan (each phase has a decision gate)

### Phase −1 — Cheap classical re-baseline (IN PROGRESS; big early win)  ← NEW
**Diagnosis** (`eval/diagnose_recall.py` — nearest-detection distance per GT node): the dense-set
collapse has TWO modes — (a) *localization*: detections land just past 7µm (e.g. 44b6_551a5dba: 64% of
GT in the 7–14µm band, median 8.3µm); (b) *sensitivity*: true cells with no detection nearby
(6bba_78a7bd97: 46% ≥14µm/none). The aggregate 0.62 recall hid both.

**Confirmed root cause of (a): XY block-downsampling (`xy_ds=4`) throws away localization precision.**
On the dense set, matched-within-7µm goes **30.7% → 66.7% → 86.0%** at `xy_ds = 4 → 2 → 1` (median
nearest 8.28 → 5.53 → 3.83µm).

**BUT the aggregate score got WORSE, not better** (dev-subset SCORE 0.534 → 0.529 → 0.466 at xy_ds
4/2/1), even though node_recall rose 0.62 → 0.75 → 0.81. Reason: finer detection also fires *more*
detections everywhere (nodes 95k → 177k → 235k), inflating FP edges (edge_J fell) and overshooting the
node budget — the "over-prediction doubly penalized" mechanic. **Empirical lesson: raising recall via
more detections backfires.**

**Refined lever: improve localization at ~CONSTANT node count.** Concrete next experiments:
- Detect at xy_ds=2 (localization gain) BUT raise `thresh_rel` / `nms_radius_um` to pull node count back
  to ~baseline (~T_true) — sweep to find the count-neutral sweet spot; check if score > 0.534.
- Keep xy_ds=4 detection (count control) but WIDEN centroid refinement (`refine_centroid` window is only
  ±5 xy-voxels ≈ 2µm — too small to fix an 8µm block-grid offset) so existing peaks localize better
  without adding nodes.
- Sub-voxel z refinement; adaptive NMS by local density for mode (b) sensitivity.
- **Gate:** best count-controlled classical dev-subset score becomes the real bar for DETECT-LEARNED.

### Phase 0 — Offline inference smoke test (no training)
Run the organizers' public checkpoint (or a quick `--epochs 3` reproduce) offline on a held-out fold.
- **Gate is FEASIBILITY + SIGNAL, not aggregate-beats-classical** (the checkpoint is undertrained AND
  possibly trained on our fold):
  - (a) loads offline, fits **≤12h on TEST-scale volumes** (not just a val fold), no OOM;
  - (b) lifts **per-dataset node_recall on the dense 0.23–0.29 sets**, even if aggregate < 0.534;
  - (c) record **sec/epoch** and inference sec/volume for the compute extrapolation.
- Do NOT calibrate the proxy on this score.

### Phase 1 — Detector-only A/B (isolate the lever) — with two prerequisites
Swap the learned **detector's centroids** into our classical Hungarian linker; score per-dataset.
- **PREREQUISITE — verify the sparse-loss masking scheme in the organizers' repo.** If unannotated cells
  are treated as **negatives**, the model is told real dense cells are background → structurally caps the
  recall we're targeting. If **ignored** (loss only at GT), generalization annotated→unannotated is the
  intended mechanism (fine). This single fact gates whether Phase 2 can even reach the ceiling.
- **Caveat:** detector-only discards `TemporalUNet3D`'s temporal attention (cross-frame consistency).
  A weak A/B is a false-negative for the *architecture*, not the *detection head* — also evaluate the
  full learned **detect+link** end-to-end separately before concluding.
- **Gate:** does learned detection lift dense-set recall vs classical (and vs Phase −1)?

### Phase 1b (fallback / parallel) — Off-the-shelf 3D nucleus detector
Cellpose-3D / StarDist-3D pretrained → centroids → our linker. No training; domain gap + 12h throughput
unknown. A/B under the same harness.

### Phase 2 — Train to a budget-derived target
Train UNet (+ transformer) on Kaggle GPU, **selecting on held-out-fold local score** (not loss), epochs
set by the Phase-0 sec/epoch extrapolation. Tune detection threshold → ~T_true. Pin **seed/determinism**
and **checkpoint-resume + logging** across sessions.
- **Gate:** best held-out (and leave-one-embryo-out) local score meaningfully > the Phase −1 bar.

### Phase 3 — Integrate + submit
Attach trained weights + vendored model code as a **private** Kaggle dataset. Extend the offline
submission notebook (`build_submission_nb.py` pattern) to load weights + run learned detect (+optionally
learned link) — **validate ≤12h on test-scale with tiling if needed** (checkpointing is train-only; full
3D+temporal inference can OOM a T4). Submit → LB; compare vs 0.581 and vs local prediction (proxy point).

## Decisions (locked with Reilly, 2026-07-08)
1. **Compute:** ✅ Kaggle GPU notebooks OK. **Surface any notebook for Reilly to review/edit the write-up
   before he runs it** — don't assume auto-run of notebooks he'll commit/submit.
2. **Scope:** ✅ Start on the organizers' reference (TemporalUNet3D) — but see DECISIVE FINDING: we'll
   retrain the detection head with a recall-friendly loss, not trust the stock checkpoint's detector.
3. **`tracking-cellmot`:** ✅ feasible offline — INSTALL it (torch) in the online training notebook;
   **VENDOR the model code** (like `eval/_vendor/`) into the offline submission notebook. "Do whatever it
   takes to build a competitive notebook that runs offline."
4. **Ambition:** Reilly researching to help decide; my judgment for now = **detector-first** (biggest lever
   is recall), swap into the classical linker, then consider full learned detect+link once detection lifts.

## Concrete first step
**Phase −1 now** (cheap classical z-localization + within-budget reallocation on the harness — no GPU, no
new deps, sets the true bar), in parallel with **confirming the test-embryo identity** from the data page
and **checking the sparse-loss masking scheme** in the organizers' repo. Only then commit to Phase 0/GPU.
