# Submission workflow (Kaggle code competition)

This is a **code competition**: you submit a *notebook* that runs offline (internet OFF, ≤12h) and
writes `submission.csv`. You (Reilly) run + submit on Kaggle; the notebooks are generated from the
verified `pipeline/` package so they never drift from what we validated locally.

## The notebooks

| File | What | Deps | Local dev-subset |
|---|---|---|---|
| `02_submission_baseline.ipynb` | greedy linking, **zero extra deps** (numpy/scipy/skimage/blosc2 — all pre-installed on Kaggle) | none | 0.534 |
| `03_submission_ilp.ipynb` | **best config**: finer detection (`xy_ds=2`) + global **ILP** linking | tracksdata + pyscipopt (via wheel dataset) | **0.648** |
| local `wheels/` (built via Docker) | offline ILP deps → uploaded as a Kaggle dataset for #3 | — | — |

`02` is the zero-dependency safety net; `03` is the real submission (our +0.11 gain). Local scores map
roughly to a higher LB (greedy 0.534 dev ↔ 0.618 LB), so `03` should land meaningfully above 0.618.

## ⚠ Confirm Kaggle's Python version first
The offline ILP deps (`pyscipopt`, `rustworkx`) are **native wheels tied to a Python version**. We build
them for **Python 3.11** (Kaggle's current default). **Confirm** by running `!python --version` in any
Kaggle notebook — if it's not 3.11, tell me and I rebuild the wheels for the right version.

## Submitting the ILP notebook (`03`) — first time

1. **Wheels are prebuilt locally** (no internet-on Kaggle notebook needed): they live in `./wheels/`,
   built in a Docker `python:3.11` container so they match Kaggle's Linux platform. Rebuild anytime with
   `bash scripts/build_kaggle_wheels.sh`.
2. **Upload the wheels as a private Kaggle dataset** (one-time), e.g.:
   ```bash
   kaggle datasets create -p wheels -m "biohub cell-tracking offline ILP wheels (tracksdata+pyscipopt, linux py3.11)"
   ```
   (add a `wheels/dataset-metadata.json`, or use the Kaggle UI "New Dataset" → upload the folder).
3. **Run the submission:** open `03_submission_ilp.ipynb`, attach (a) the competition data and
   (b) the wheels dataset. **Internet: Off.** Run All.
   - The install cell `pip install --no-index`-s tracksdata+pyscipopt from the wheels (no internet).
   - It detects the test dir, runs detect → ILP → writes `/kaggle/working/submission.csv`, validates columns.
   - ILP uses **SCIP** offline (auto-fallback from Gurobi); a per-dataset `ilp_timeout` (600s) guards the budget.
4. **Submit.** Record the LB score in `BACKLOG.md` (our next real calibration point).

Later submissions: reuse the same wheels dataset; just re-run `03`. Rebuild wheels only if deps change.

## Regenerating after a pipeline change

The submission notebooks are GENERATED — never hand-edit the inlined code. After changing anything in
`pipeline/`, regenerate so the notebook matches the verified package:

```bash
.venv-track/bin/python notebooks/build_submission_ilp_nb.py     # -> 03_submission_ilp.ipynb
.venv-track/bin/python notebooks/build_submission_nb.py          # -> 02_submission_baseline.ipynb
```

Both are validated locally by executing them on `data/test` (the install/deps cells no-op locally, where
tracksdata is already present in `.venv-track`).

## Notes / gotchas

- **Config lives in the notebook's run cell** (`PipelineConfig(xy_ds=2, thresh_rel=0.30, link_method="ilp", ...)`)
  — it mirrors `menu.yaml`. Change it there (and in `menu.yaml`) to sweep, then regenerate.
- **Wheels are platform-specific** → always build them on Kaggle (`04`), not locally (macOS wheels won't load on Kaggle Linux).
- **ILP solve time grows with node count** (~36s / 19k nodes). Fine for 4 test volumes; the timeout is the backstop.
- Do NOT add a `competition:` upload path — submission is notebook-only.
