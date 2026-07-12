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
shot 1 (veto full):        LB = ____   ← the differentiator
shot 1 (veto full):        LB = ____
shot 2 (veto gap-only):    LB = ____
shot 3 (learned_bonus 1.1):LB = ____
shot 4 (det 0.95):         LB = ____
```
Read the pattern: if the veto helps, precision-of-repairs is the lever → tune its thresholds
(`BIOHUB_DEEPCENTER_GAP_THRESHOLD` / `_SAFE_DIV_THRESHOLD`). If det 0.95 helps, sweep it finer.
