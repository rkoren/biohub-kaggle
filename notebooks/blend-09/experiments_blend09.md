# Experiments from the ~0.90 blend baseline

Working notebook: `blend09_codeonly.ipynb`. Each shot goes in the **experiment override cell** (the one
between the baseline-config cell and the config-reader cell). One variable at a time; jot each LB score.

**Baseline (no overrides):** reproduces pilkwang's ~0.90 run — DeepCenter veto OFF. **Run this first** to
confirm we land at ~0.90 before experimenting.

**Attach:** `biohub-local-association-ranker-unet300-v1`, `biohub-tracking-support-pack-50ep-v1` (+
`biohub-deepcenter-unet3d-center-prior-v1` for Shots 1–2) + competition data. GPU on, Internet off.

---

### Shot 0 — Baseline confirm  *(no overrides)*
Leave the override cell as-is. Target: ~0.90. Establishes our number on this pipeline.

### Shot 1 — Enable the DeepCenter veto  *(the big untapped lever)*  ⭐ my pick
A 2nd detector (`full_frame_center/best.pt`) vetoes candidate repair points, so only real cells get added at
gap-closes and safe-divisions → higher-precision repairs. pilkwang built it but left it OFF in the 0.90 run.
**Attach `pilkwang/biohub-deepcenter-unet3d-center-prior-v1` first**, else it no-ops.
```python
os.environ["BIOHUB_USE_DEEPCENTER_VETO"]      = "1"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"]      = "1"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "1"
```

### Shot 2 — Veto on gap-close only  *(isolate which veto helps)*
If Shot 1 moves the score, split it: veto the gap-close repairs but not divisions (or vice-versa).
```python
os.environ["BIOHUB_USE_DEEPCENTER_VETO"]      = "1"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"]      = "1"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
```

### Shot 3 — Push learned-edge influence  *(the axis pilkwang was probing)*
The notebook's stated score-axis is "increase learned-edge influence inside motion assignment"; they anchor
`0.90`. Nudge above it — the association ranker may deserve more weight.
```python
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.10"
```

### Shot 4 — Detection recall  *(re-inference)*
On the 50ep model, 0.99→0.95 gained +0.002 on the LB. The 0.90 blend anchors 0.97; test whether this
stronger model still has recall headroom.
```python
os.environ["BIOHUB_DET_THRESHOLD"] = "0.95"
```

## Tuning-notebook config ladder (run in `blend09_tuning.ipynb`, held-out-train CV, FREE)

After the veto miss (raw-path loop mis-called it), test EVERYTHING here first. Rule: a config earns a
submission only if it **beats the baseline anchor's held-out-train CV**. Paste into the experiment cell,
Run All, read the CV at the bottom. Order by thesis strength:

- **T0 · Baseline anchor** — no overrides. Establishes the blend's held-out-train CV (the reference).
  Also a calibration check: does it land near the 0.90 test LB, or lower?
- **T1 · det=0.95** *(re-inference)* — the fork (WITH postproc) wanted lower det; blend anchors 0.97.
  `os.environ["BIOHUB_DET_THRESHOLD"]="0.95"`
- **T2 · Tighter gaps: max_gap=1** — the veto carnage (~2000 bad gap midpoints) hints the blend's dt=2
  gap-closes are noisy; dt=1 may be cleaner without needing a veto.
  `os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"]="1"`
- **T3 · Veto SAFE-DIV only** — the *gap* veto was the disaster (2000 rejects); vetoing only the rare
  divisions is targeted and low-volume. Attach the deepcenter dataset.
  `os.environ["BIOHUB_USE_DEEPCENTER_VETO"]="1"; os.environ["BIOHUB_DEEPCENTER_GAP_VETO"]="0"; os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"]="1"`
- **T4 · Learned-edge bonus 1.10** — the blend's stated probing axis (anchors 0.90); nudge up.
  `os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"]="1.10"`
- **T5 · min_track_len=8** — dense learned predictions may want more short-track filtering (blend=6).
  `os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"]="8"`

```
T0 baseline anchor:   train-CV = ____   (vs LB baseline 0.900)
T1 det=0.95:          train-CV = ____
T2 max_gap=1:         train-CV = ____
T3 veto safe-div:     train-CV = ____
T4 learned_bonus 1.1: train-CV = ____
T5 min_track=8:       train-CV = ____
```

---

### Shot 5 — Aggressive short-track filter  *(local-sweep hint; postproc)*
Local sweep on learned geffs: dense learned predictions like *more* filtering (min_track_len 6→8 gained;
4 hurt — opposite of our sparse classical output).
```python
os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "8"
```

### Shot 6 — Tighter gap-close radius  *(local-sweep hint; postproc)*
Local sweep's single best knob: gap-close radius 6→5 µm (+0.011 on our module). Learned links are cleaner,
so a tighter gate adds fewer bad bridges.
```python
os.environ["BIOHUB_GAP_CLOSE_UM"] = "5.0"
```
> Shots 5–6 are hints from tuning *our* postproc on local learned geffs — the blend's cell-11 surgery is
> already tuned by pilkwang, so treat as marginal probes, well below the veto (Shot 1) in priority.

---
```
shot 0 (baseline):         LB = 0.900  ✓ (16th, tied ~100)
shot 1 (veto full):        LB = 0.893  ✗ (−0.007; veto HURTS the blend)
shot 2 (veto gap-only):    LB = ____   (diagnostic only — expect ≈0.893; skippable, see verdict)
shot 3 (learned_bonus 1.1):LB = ____   ← NEXT (highest-EV, veto-independent)
shot 6 (gap_close 5.0µm):  LB = ____   ← then this
shot 4 (det 0.95):         LB = ____
```

### VERDICT on the veto (2026-07-12) — retire it
Shot 1 = **0.893 (−0.007)**. The veto is a precision gate, and on the strong blend (D4 TTA + association
ranker) the base repairs are already high-precision, so the gate only rejects *good* gap-close bridges —
gap-close is the main edge-recall lever. Corroborating evidence: (a) **pilkwang built the veto, wired it,
and shipped it OFF** = a designer's confirmed-neutral/negative lever, not "untapped headroom" as we'd
framed it; (b) our local `veto_harness` said +0.0046 but the LB says −0.007 → **the harness sign-flipped**,
so don't trust more local veto sweeps; (c) divisions are ~0.1× and 56% of videos have none, so the
safe-div half of the veto can't move the score — the −0.007 is the **gap-veto** rejecting real bridges.
- **Don't** run div-only (gap off, veto on) — by the same logic it just restates ≈0.900.
- **Shot 2 (gap-only)** is *diagnostic only*: expect ≈0.893 (confirms gap-veto is the culprit). Skip it
  unless a blend-log check shows the safe-div veto vetoed a *meaningful* number of candidates in the 0.893
  run (if it did, div contribution isn't ~0 and Shot 2 becomes informative). Otherwise go straight to Shot 3.

### Next bets (veto-independent)
- **Shot 3** — `MOTION_RELINK_LEARNED_BONUS=1.10`: the ladder's real axis (more weight to the learned ranker
  inside motion assignment). Highest EV.
- **Shot 6** — `GAP_CLOSE_UM=5.0`: also a precision gate on gap-close, but filters on *geometry* (cleaner
  learned links tolerate a tighter gate), NOT a 2nd model's heatmap → not the same failed lever as the veto.
