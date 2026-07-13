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

## Dead ends (don't retry)
- Raw-path local loop (det=0.5, no TTA/ranker) — mis-called det AND veto direction; superseded by `blend09_tuning`.
- Classical ILP pipeline capped ~0.72 LB.
