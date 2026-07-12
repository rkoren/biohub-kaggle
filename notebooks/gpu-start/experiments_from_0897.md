# Experiments from the 0.889 pilkwang baseline

Each shot = a config cell pasted as the **FIRST cell** of the fork (`rkoren/fork-full-visual-pipeline-and-animation`),
so it sets env vars before the fork's config cell reads them. Then Run All → Submit. One variable at a
time so we learn which lever moves the score; combine winners afterward.

**Baseline:** pilkwang defaults = **0.889**. Dominant error is missed edges (FN ≫ FP in the metric),
so most shots push edge **recall**; the division term (0.1×) is the other untapped lever.

Rationale reminder: pilkwang's knobs are already tuned for the dense learned predictions, so these are
genuine experiments, not sure things. The clean de-risk is `gpu-start/01` → `sweep_postproc.py` first,
but with submission budget to spend, this ladder is the fast path.

---

### Shot 1 — Enable dt=2 gap recovery  *(postproc only; safest, no re-inference)*
Bridges 2-frame detection gaps → recovers FN edges. Tightly capped (≤180 links, geometric + motion-context
gates), so downside is bounded. pilkwang left it OFF; our sweep showed +0.004 on classical.
```python
import os
os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"
```

### Shot 2 — Lower detection threshold 0.99 → 0.95  *(re-inference; biggest model-side lever)*
det=0.99 is very high-precision; the metric rewards detecting toward the true cell count. If the 50ep model
under-detects at 0.99, this recovers recall. Higher variance (re-runs inference; high-precision may be
deliberate) — a modest step, not aggressive.
```python
import os
os.environ["BIOHUB_DET_THRESHOLD"] = "0.95"
```

### Shot 3 — Recall swing: det 0.92 + gap2 on  *(bigger push)*
If Shots 1–2 both nudge up, this combines them and pushes detection recall further.
```python
import os
os.environ["BIOHUB_DET_THRESHOLD"] = "0.92"
os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"
```

### Shot 4 (optional) — Division push  *(targets the 0.1× term)*
Widen safe-division gates + turn on the geometry filter to recover more mitosis events (we only reach
div_J≈0.03 today; each recovered division is worth ~+0.003 final).
```python
import os
os.environ["BIOHUB_OUTPUT_SAFE_DIVISIONS"] = "1"
os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "8.5"
os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "5.5"
os.environ["BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER"] = "1"
```

---

**Reading results:** note each submission's score next to its shot. If Shot 2 helps, det_threshold is a live
lever → sweep it finer (0.90/0.93/0.97) next. If it hurts, high-precision is confirmed deliberate → drop it
and focus on postproc + divisions. Fold winners together for a final combined submission.
```
shot 1 (gap2):            LB = ____
shot 2 (det 0.95):        LB = ____
shot 3 (det0.92+gap2):    LB = ____
shot 4 (division push):   LB = ____
```
