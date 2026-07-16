# Results log — configs, held-out-train CV, and leaderboard

The single source of truth for what we've tried. **Held-out-train CV** is from `blend09_tuning.ipynb`
(5 fold-0 videos, official metric); **LB** is the real leaderboard (4 test videos). The gap between them
is the calibration we're building — fill LB as scores land.

## Blend pipeline (~0.90 base)

| config | train-CV | LB | Δtrain vs base | Δ LB vs base | notes |
|---|---:|---:|---:|---:|---|
| blend baseline | 0.8560 | **0.900** | — | — | pilkwang defaults; T0 anchor |
| + full veto (gap+div) | — | **0.893** | — | −0.007 | veto rejected ~2000/2004 repairs; DON'T |
| **C1** = div-veto-only + min_track=8 | 0.8631 | **0.893** | +0.0071 | **−0.007** | ⚠ ANTI-transferred; div-veto hurts LB |
| **C2** = C1 + det=0.95 + gap_close_um=5 | 0.8719 | NOT SUBMITTED | +0.0159 | — | includes div-veto → likely ~0.89; HOLD |
| **competitor / gap-confirm** (=V6) = gap-confirm min_span7 + gap_thr0.2 + div-veto-off (linefit/bonus inert) | 0.8652 | **0.901** | +0.0091 | **+0.001** | ✅ NEW BEST; first veto-family member to transfer, but +0.0091 train→+0.001 LB = heavily attenuated (precision-only cap). Champion baseline for candidate #2. |

### Component isolation (v8, 2026-07-13) — CORRECTS the prior "linefit/bonus" hypothesis
Ran each competitor knob alone on held-out CV (predict-once sweep, same cached geffs):

| knob (vs baseline 0.8561) | train-CV | Δ |
|---|---:|---:|
| +linefit=0.77 | 0.8561 | **+0.0000 (inert)** |
| +learned_bonus=0.75 | 0.8561 | **+0.0000 (inert)** |
| +linefit+bonus together | 0.8561 | **+0.0000 (inert)** |
| **+gap-confirm only** (gap-veto min_span7 thr0.2, div-veto off) | **0.8652** | **+0.0091** |
| full competitor combo | 0.8652 | +0.0091 |

**The entire competitor gain is the gap-confirm gate; linefit=0.77 and learned_bonus=0.75 do nothing on held-out
CV.** So V6 (submitted, pending) is a *pure gap-confirm test*. gap-confirm is a **linking-side** fix (selectively
vetoes bad gap-repairs). ⚠ But the gap-veto family has anti-transferred twice (blanket −0.007 LB, C1 div-veto
+0.007→−0.007) — this +0.0091 is the same train-CV shape that burned us, so **V6's LB is the verdict; do NOT
stack the inert linefit/bonus knobs onto a candidate.**

## Calibration (train-CV → LB)

- Baseline: train-CV **0.856** vs LB **0.900** → **LB sits ~+0.044 above train-CV** (fold-0 train harder than
  the 4 test clips). Same pipeline, so **relative deltas should transfer; absolutes don't**.
- **Open question the pending LBs answer:** do the +0.007 / +0.016 train-CV deltas convert 1:1 to the LB?
  - C1's LB vs 0.900 → does the div-veto+mt8 delta transfer?
  - C2's LB vs C1's LB → isolates exactly what `det=0.95 + gap_close_um=5` buys (C2 is a strict superset of C1).
- **Noise floor:** on 5 videos, min_track 8/10/12 swung ±0.008 — so trust deltas ≥ ~0.005 and reproduced-across-runs
  (gap_close_um=5 reproduced 0.8719 twice) over single-run nominal bests.

## Validated levers (held-out-train CV)

- **div-veto-only** (+0.0047), **min_track=8** (+0.0022), **det=0.95** (+0.0039, stacks), **gap_close_um=5**
  (+0.0049, reproduced). Neutral/negative: full veto (−LB), gap veto, max_gap=1, learned_bonus, gap4, min_track=10
  (w/ gap5), edge_max_um=12, linefit_win=3, gap2.

## Failure-mode analysis (where the blend loses points)

Tool: `eval/failure_modes.py` — splits every missed GT edge (FN) into **detection-limited** (a cell was
never found, so the edge is impossible) vs **linking-limited** (both cells found, edge just not made). This
is the strategic question the 0.90-tuners don't ask: is the wall detection or the linker?

- **Division census (decisive, local, GT-only):** held-out fold-0 has **1 GT division across all 5 videos**
  (1086 nodes). Full train: 151 divisions / 133k nodes = **0.113% of nodes divide** → ~0.76 div/video →
  ~3 divisions on the 4 test clips. The 0.1× division term is a **thin, high-variance dial, not a frontier**;
  with GT divisions this rare, `div_jaccard`'s denominator is dominated by *false* divisions (greedy path's
  33 FP divisions → div_jaccard 0.029). So the game is the **edge term** (0.9 of the weight), and on divisions
  the lever is *not creating* false ones, not recovering true ones. (Confirms C1 div-veto anti-transfer risk.)
- **Greedy-proxy FN split (code-validation only, NOT the blend):** on the simple det=0.5 path, 98% of missed
  edges are linking-limited, median miss 1.9µm (short adjacent-frame jumps), only 8% span >7µm. This is
  precisely the class the blend's ranker+ILP+surgery exists to fix — so it says nothing about the blend yet.
- **BLEND baseline profile (v10, held-out fold-0, DECISIVE):** raw eJ 0.849 — TP=1002, **FP=107, FN=71**.
  - FN split: **83% linking-limited** (59: both cells detected, edge not made), 17% detection-limited (12).
  - Linking-limited misses are SHORT: median 3.3µm, p90 7.4µm, only 17% span >7µm (adjacent-frame drops).
  - **FP=107 are all valid spurious links** near GT-active cells (metric's `pred_valid`) — a linking error,
    NOT over-detection noise. (Pred detects ~113k cells densely = correct vs sparse GT's ~1086 scored nodes;
    those ~112k unscored edges are the GT-sparsity, node-penalty ≈ 0.)
  - **⇒ 166/178 scored errors (93%) are LINKING decisions among correctly-detected cells; detection is only
    7%.** The 0.90-crowd's detection-threshold tuning polishes the wrong 7%. Divisions negligible (0/6/1).
  - **The linker OVER-links** (FP 107 > FN 71) → precision ceiling (remove FP → 0.934) > recall ceiling
    (recover FN → 0.908). This is *why* gap-confirm (a precision veto) is the only knob that moved CV.
  - Errors concentrate: 267148e4 (50FP+30FN, worst), 3bb3690f (34+20); 1574802b is the lone detection case
    (8 of 12 FN detection-limited); 0b24845f perfect.
  - Local artifact saved: `blend_heldout_baseline.csv` (full baseline submission) → iterate with
    `eval/failure_modes.py` without re-predicting. (NB: submission stores integer voxel coords; `load_pred`
    now rounds to match — raw floats mis-count FP/FN by ~3 edges/video via the 7µm node matching.)

### Failure mode NAMED (v10 anatomy, `eval/error_anatomy.py`): FAST LATERAL MOTION
Compared error edges to correct edges on the two linking-failure axes (crowding vs displacement):
- **Crowding is ~0 everywhere** (sparse GT → no cell has a neighbour within 7µm) → dense-ambiguity is NOT
  the mode. Rules out the "cluster confusion" hypothesis.
- **Displacement cleanly separates errors from correct edges.** Worst video 267148e4: missed edges move
  **3.0× farther** (median 6.56µm vs 2.19µm; p90 8.07µm — past the 7µm match radius). Signal same direction
  in all videos (FN displacement > TP displacement).
- **z-decomposition:** the excess is LATERAL, not depth — 267148e4 FN-link has same z (1.62µm) as TP but
  **3.6× the xy** (6.13 vs 1.72µm). The linker drops cells that jump fast in xy while their endpoints stay
  detected.
- **⇒ Reframe:** the frontier is the LINKER's fast-lateral-motion handling, not detection/density/divisions.
  Two independent levers on the 178 scored errors: **precision** (remove the 107 spurious FP — gap-confirm /
  veto family, being tested by pending V6) and **recall** (recover the 59 fast-motion FN — motion-relink /
  association radius, UNTESTED and orthogonal to the veto play). ⚠ n=28 FN in one video; transfer to the 4
  test clips unknown — same 5-video caveat as every CV delta.
- **FP and FN are ONE coupled failure (not two levers):** none of the 107 FP link two GT cells (FP-wrong-link=0
  in every video) — every FP links a real GT cell to an over-detected *extra* cell. And **48/60 (80%) of the
  fast-motion misses are "stolen":** the linker DID give the source cell an outgoing edge, just to the wrong
  (nearer, spurious) target instead of the correct far one. Mechanism: fast lateral motion → a nearer
  over-detection exists → linker greedily grabs it (FP) and abandons the correct far partner (FN). One bad
  decision, two scorecard errors.
- **⇒ The lever is motion-gated re-association, and it beats the veto.** The veto family (V6/gap-confirm) can
  only *remove* the FP, at a TP cost (its 3:9 trade); it cannot recover the abandoned correct link. A
  velocity-consistent association (prefer the motion-predicted target over the nearest one) fixes BOTH the FP
  and the FN on the same 48 events. Ceiling if fully fixed: raw eJ 0.849 → ~0.93. `edge_max_um=12` alone was
  neutral because widening the radius without motion-gating just adds more near-neighbour candidates to steal.
- **Conditional next step:** V6's LB still tells us whether the veto/precision play transfers, but the motion
  lever is orthogonal and higher-ceiling regardless. Scope it as the primary GPU experiment: modify the blend's
  association / motion-relink to score candidates by velocity-consistency, not proximity. n=60 across 5 videos;
  test-transfer unknown.

## Phase 1 linking sweep (v11, 2026-07-14) — cheap postproc CANNOT fix the steal
- **B/C triage: 100% category C (60/60).** Every fast-motion miss has the correct edge ABSENT from the ILP
  solution → `motion_relink` sees prob 0.0 → no gate/bonus/fallback knob can recover it. Empirically confirms
  the postproc kernel is structurally blind to the steal (plan `staged-jumping-sifakis.md` F2).
- **single-pass union gate (12µm): CATASTROPHIC** — CV 0.856→0.62, FP 110→367. Widening the Hungarian gate
  floods false links graph-wide. Dead end (don't retry gate-widening without candidate-prob gating).
- **keep-ILP-fallback alone: 0.8653** (≈ gap-confirm 0.8652; +0.0092 vs baseline) — recovers a few non-fast-motion
  FN and sheds a little FP, but within 5-video noise and NOT the steal population. Marginal; not candidate #2.
- **⇒ Phase 2 (instrumented re-predict to dump candidate probs + embeddings) is mandatory.** The correct edge
  is killed at candidate-gen or ILP selection, upstream of all postproc — the ranker under-scores large-`rel`
  edges. Fix needs a displacement-agnostic signal (appearance re-link Rung 4, or retrain Rung 6), not knobs.

## Phase 2 fork resolution (v12, 2026-07-14) — the ranker is BLIND to fast-motion edges
Instrumented re-predict dumped candidate probs + appearance embeddings (`pipeline/predict_instrumented.py`,
`eval/candidate_probe.py`). For the 60 category-C misses:
- **prob(correct) ≈ 0: 98% below the 0.02 dump floor, 0% cleared the 0.5 gate, median 0.000.** The ranker
  scores the correct far edge near-zero (large `rel` → out-of-distribution).
- The taken distractor edge is ALSO ~0 prob — it exists only because `motion_relink` built it on geometry
  (nearest-within-gate), not because the ranker liked it.
- **⇒ No upstream selection/ILP/bonus fix is possible — the ranker signal is zero.** Recovery needs a
  displacement-agnostic signal: appearance re-link (Rung 4, embeddings dumped) or retrain (Rung 6).
- **Appearance re-link (Rung 4) is DEAD too** (`eval/appearance_probe.py`): raw 32-ch UNet embeddings do NOT
  separate the correct partner. Control on 113,910 easy adjacent links: cos(source,linked) 0.930 =
  cos(source,random) 0.930 → **linked>random exactly 50% (chance)**. Embedding std 0.28 (real variance, not
  degenerate) but it tracks local image context, not cell identity — they're the detector's features, not a
  re-ID fingerprint. For the 60 misses, cos(correct) even < cos(random). No appearance re-link can recover them.

## rel-ZERO ranker probe (v13) — the ranker's LEARNED space is blind too, not just the rel penalty
Re-ran inference with the ranker's displacement term (`rel`) zeroed (env-gated patch to
`simple_node_transformer.py:193`), dumped rel-zeroed candidate probs. `candidate_probe` on the 60 misses:
**identical to rel-ON — prob(correct) median 0.000, 98% below floor, 1/48 ranks correct>distractor.** So the
learned appearance-attention itself has no signal for fast-motion partners; the `rel` term wasn't the culprit.
The last cheaper-than-retrain avenue (re-rank on the ranker's learned metric) is now closed too.

## raw-logit rank probe (v14) — NOT a softmax artifact: the model is genuinely blind
Dumped top-8 sources per target by RAW pre-softmax logit (`eval/rawlogit_probe.py`). For the 60 misses, the
true source is **ABSENT from the correct target's top-8 in 97%** (rank-1 in only 1/60). The model ranks 7+ wrong
sources above the true parent — so `prob(correct)≈0` was real signal absence, not softmax-floor dilution. Rules
out any cheap re-rank/threshold/selection fix. **Retrain (Rung 6) confirmed as the only path**, gated by a
5-epoch fine-tune probe (does prob(correct)/true-source-rank move off the floor? yes→full retrain, no→window>2).

## ⇒ CHEAP POSTPROC ON THIS CHECKPOINT is capped (~0.901-0.902). Retrain is the leading path to 0.91.
Every downstream signal to recover the fast-motion steal *using the 50ep checkpoint's outputs* is exhausted:
softmax prob ≈ 0, constant-velocity 1/28, track-continuation 1/48, raw appearance cosine = chance. So no postproc
knob (gap-confirm, motion-margin, local-spacing, keep-fallback, gate tuning) passes ~0.901 — matching the
"dozens of 0.90 tuners, no public 0.91" landscape. **Caveats (advisor):** (a) prob≈0 is softmax-over-sources
output; with ~50 candidate sources per target, per-source prob floors near 1/50 ≈ 0.02 = our dump floor, so this
may be a floor artifact — the RAW pre-softmax logit ranking is the real test (pending). (b) We tested the
*trained* checkpoint's features (raw `_index_features`), not the architecture's capacity when trained on hard
positives. So "retrain" is the leading path (large-displacement positives / window>2 / loss reweighting), to be
gated by a cheap 5-epoch fine-tune probe, NOT a proven necessity. Postproc bank (gap-confirm+competitor) ~0.902.

## Dead ends (don't retry)
- Raw-path local loop (det=0.5, no TTA/ranker) — mis-called det AND veto direction; superseded by `blend09_tuning`.
- Classical ILP pipeline capped ~0.72 LB.
