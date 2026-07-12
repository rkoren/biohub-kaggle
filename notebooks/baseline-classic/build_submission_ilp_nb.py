"""Generate the offline ILP submission notebook from the verified pipeline/ package.

Like build_submission_nb.py but uses global ILP linking (our best: xy_ds=2, thr0.30,
link=ilp -> 0.648 dev-subset). ILP needs tracksdata + pyscipopt, which aren't pre-installed
on Kaggle — so the notebook opens with a graceful offline-install cell that pip-installs them
from an attached WHEEL DATASET (built by notebooks/build_wheels_nb.py) with --no-index (no
internet). Locally, where the deps already exist, that cell no-ops.

Run:  .venv-track/bin/python notebooks/build_submission_ilp_nb.py
"""
from pathlib import Path
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "pipeline"


def strip_source(path: Path) -> str:
    out = []
    for line in path.read_text().splitlines():
        if line.strip().startswith("from __future__"):
            continue
        if re.match(r"^\s*from \.", line):   # relative imports (top-level OR indented)
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


CONFIG = strip_source(PKG / "config.py")
DETECT = strip_source(PKG / "detect.py")
SUBMISSION = strip_source(PKG / "submission.py")
LINK_ILP = strip_source(PKG / "link_ilp.py")
POSTPROCESS = strip_source(PKG / "postprocess.py")

INSTALL = '''
# --- Offline install of ILP deps (tracksdata + pyscipopt) -------------------
# Attach the wheels dataset (kaggle.com/datasets/rkoren/biohub-celltrack-ilp-wheels)
# to this notebook; with Internet OFF this installs from it via --no-index (no network).
# Locally (deps already present), this no-ops. Robust to wheels being extracted OR zipped.
import subprocess, sys, glob, os, zipfile, shutil

FLAT = "/tmp/ilp_wheels"

def _gather_wheels() -> int:
    """Collect EVERY .whl found under /kaggle/input into one flat dir (Kaggle may
    mount/extract the dataset at a nested path, and wheels can be split across dirs)."""
    os.makedirs(FLAT, exist_ok=True)
    hits = glob.glob("/kaggle/input/**/*.whl", recursive=True)
    if not hits:  # wheels delivered inside a .zip → extract first
        for z in glob.glob("/kaggle/input/**/*.zip", recursive=True):
            try:
                zipfile.ZipFile(z).extractall("/tmp/ilp_zip")
            except Exception:
                pass
        hits = glob.glob("/tmp/ilp_zip/**/*.whl", recursive=True)
    for w in hits:
        dst = os.path.join(FLAT, os.path.basename(w))
        if not os.path.exists(dst):
            shutil.copy(w, dst)
    return len(glob.glob(f"{FLAT}/*.whl"))

# Install our wheel ONLY for packages Kaggle doesn't already have (or has too old for tracksdata).
# Overriding Kaggle's numpy/numba/etc. with our differently-compiled builds ABI-breaks its stack, so
# we DELETE any wheel whose package is already installed — EXCEPT this KEEP set, which tracksdata needs
# at versions newer/older than Kaggle ships (polars>=1.36 vs 1.35; numcodecs<0.16; the git-only packages).
_KEEP = {"tracksdata", "pyscipopt", "ilpy", "geff", "geff_spec", "bidict", "donfig",
         "rustworkx", "polars", "polars_runtime_32", "numcodecs", "zarr", "imagecodecs"}

def _norm(whl: str) -> str:
    return whl.split("-")[0].lower().replace("-", "_")

def _installed(name: str) -> bool:
    from importlib.metadata import version, PackageNotFoundError
    for cand in (name, name.replace("_", "-")):
        try:
            version(cand); return True
        except PackageNotFoundError:
            continue
    return False

def _ensure_ilp_deps():
    try:
        import tracksdata, pyscipopt  # noqa: F401
        print("ILP deps already present — skipping install")
        return
    except ImportError:
        pass
    n = _gather_wheels()
    if n == 0:
        raise RuntimeError("No .whl found under /kaggle/input — attach the wheels dataset.")
    deferred = []
    for w in glob.glob(f"{FLAT}/*.whl"):
        name = _norm(os.path.basename(w))
        if name not in _KEEP and _installed(name):
            os.remove(w); deferred.append(name)
    kept = len(glob.glob(f"{FLAT}/*.whl"))
    print(f"installing {kept} wheels; deferring {len(deferred)} already on Kaggle (incl. "
          f"{', '.join(x for x in ('numpy','scipy','numba','llvmlite') if x in deferred)})")
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--pre",
                    f"--find-links={FLAT}", "tracksdata", "pyscipopt"], check=True)

_ensure_ilp_deps()
'''.strip("\n")

READER = '''
# --- Offline image reader (blosc2 chunk reader; verified byte-identical to zarr) ---
import json as _json

def read_array_meta(zarr_path: Path):
    candidates = [zarr_path / "0" / "zarr.json", zarr_path / "0" / ".zarray"]
    candidates += list(zarr_path.rglob("zarr.json"))[:8]
    candidates += list(zarr_path.rglob(".zarray"))[:8]
    seen = set()
    for meta_path in candidates:
        if meta_path in seen or not meta_path.exists():
            continue
        seen.add(meta_path)
        try:
            meta = _json.load(open(meta_path))
        except Exception:
            continue
        shape = tuple(meta.get("shape", ()))
        if len(shape) != 4:
            continue
        dtype = np.dtype(meta.get("data_type", meta.get("dtype")))
        if "chunk_grid" in meta:
            chunk_shape = tuple(meta["chunk_grid"]["configuration"]["chunk_shape"])
        else:
            chunk_shape = tuple(meta.get("chunks", shape))
        return meta_path.parent, shape, dtype, chunk_shape
    raise FileNotFoundError(f"Could not find 4D zarr metadata under {zarr_path}")


def _chunk_candidates(array_dir: Path, t: int):
    return [array_dir / "c" / str(t) / "0" / "0" / "0",
            array_dir / f"{t}.0.0.0", array_dir / str(t) / "0" / "0" / "0"]


def load_volume(array_dir: Path, shape, dtype, t: int) -> np.ndarray:
    import blosc2
    for cp in _chunk_candidates(array_dir, t):
        if cp.exists():
            raw = open(cp, "rb").read()
            try:
                dec = blosc2.decompress(raw)
            except Exception:
                dec = raw
            arr = np.frombuffer(dec, dtype=dtype)
            expected = int(np.prod(shape[1:]))
            if arr.size < expected:
                out = np.zeros(expected, dtype=dtype); out[:arr.size] = arr; arr = out
            return arr[:expected].reshape(shape[1:])
    raise FileNotFoundError(f"Missing chunk for t={t} in {array_dir}")
'''.strip("\n")

RUN = '''
# --- Run: detect every frame -> global ILP linking -> submission.csv ---------
import time, gc

# Best config (dev-subset 0.648): finer XY detection + global ILP linking.
cfg = PipelineConfig(xy_ds=2, thresh_rel=0.30, link_method="ilp",
                     ilp_appearance=0.1, ilp_disappearance=0.1, ilp_division=1.0)

def run_dataset(zarr_path: Path, dataset: str) -> list:
    array_dir, shape, dtype, _ = read_array_meta(zarr_path)
    T = shape[0]
    node_rows, node_scores = [], {}
    prev_count, next_id = None, 1
    t0 = time.time()
    for t in range(T):
        vol = load_volume(array_dir, shape, dtype, t)
        coords, scores = detect_cells(vol, cfg, prev_count=prev_count)
        del vol; gc.collect()
        if len(coords):
            o = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0])); coords, scores = coords[o], scores[o]
        ids = list(range(next_id, next_id + len(coords))); next_id += len(coords)
        for nid, zyx, sc in zip(ids, coords, scores):
            node_rows.append(node_row(dataset, nid, t, zyx)); node_scores[int(nid)] = float(sc)
        prev_count = len(coords)
    edges = ilp_link(node_rows, cfg, n_neighbors=cfg.link_n_neighbors, delta_t=cfg.link_delta_t,
                     appearance=cfg.ilp_appearance, disappearance=cfg.ilp_disappearance,
                     division=cfg.ilp_division, timeout=cfg.ilp_timeout)
    node_rows, edges, _ = prune_isolated(node_rows, edges, node_scores, cfg)
    print(f"  [{dataset}] {time.time()-t0:.0f}s | {len(node_rows)} nodes, {len(edges)} edges")
    return node_rows + edges


def prune_isolated(node_rows, edge_rows, node_scores, cfg):
    if not cfg.prune_isolated_nodes or not node_rows:
        return node_rows, edge_rows, None
    keep = set()
    for e in edge_rows:
        keep.add(int(e["source_id"])); keep.add(int(e["target_id"]))
    kn = [r for r in node_rows if int(r["node_id"]) in keep]
    kids = {int(r["node_id"]) for r in kn}
    ke = [e for e in edge_rows if int(e["source_id"]) in kids and int(e["target_id"]) in kids]
    return kn, ke, None


def resolve_test_dir() -> Path:
    for c in [Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test"),
              Path("/kaggle/input/biohub-cell-tracking-during-development/test"),
              Path("data/test"), Path("../data/test")]:
        if c.exists() and list(c.glob("*.zarr")):
            return c
    for h in [Path(p) for p in __import__("glob").glob("/kaggle/input/**/test", recursive=True)]:
        if list(h.glob("*.zarr")):
            return h
    raise FileNotFoundError("No test/*.zarr directory found")


TEST_DIR = resolve_test_dir()
OUT = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("submission.csv")
names = sorted(p.name[:-5] for p in TEST_DIR.iterdir() if p.name.endswith(".zarr"))
print(f"TEST_DIR={TEST_DIR}  datasets={names}")

t0 = time.time()
rows = []
for i, n in enumerate(names, 1):
    print(f"[{i}/{len(names)}] {n}")
    rows.extend(run_dataset(TEST_DIR / f"{n}.zarr", n))
sub = assemble(rows)
# Stage-B graph surgery (pilkwang 0.897, CC0): +0.040 on our dev subset (0.648 -> 0.688).
sub, _pp = apply_to_submission(sub)
validate(sub, expected_datasets=set(names))
save(sub, OUT)
print(f"\\nWrote {OUT}: {len(sub):,} rows "
      f"({int((sub.row_type=='node').sum()):,} nodes, {int((sub.row_type=='edge').sum()):,} edges) "
      f"in {(time.time()-t0)/60:.1f} min")
sub.head()
'''.strip("\n")

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# 🧬 Cell Tracking — ILP Submission (offline, best config: 0.648 dev-subset)

Produces `submission.csv` for the [Biohub Cell Tracking](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
code competition using our best pipeline: **finer-XY detection + global ILP linking**
(`xy_ds=2`, `thresh_rel=0.30`, `tracksdata` ILP). Scored **0.648** on our 13-dataset local dev subset,
up from the greedy baseline's 0.534 (≈0.618 LB).

**Setup (one-time):**
1. Run `build_wheels_nb.py` (internet ON) → save its `/kaggle/working/wheels` output as a **private** dataset.
2. In THIS notebook: attach that wheel dataset **and** the competition data, set **Internet: Off**, Run All, Submit.

The ILP solver runs offline via **SCIP** (`pyscipopt`, bundled — no Gurobi/internet). A per-dataset solve
timeout guards the 12h budget. *Generated from the verified `pipeline/` package (`build_submission_ilp_nb.py`).*
""")
md("### Install ILP deps (offline, from the attached wheel dataset)")
code(INSTALL)
md("### Imports")
code("import gc\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd\nimport polars as pl\nimport tracksdata as td")
md("### Configuration")
code(CONFIG)
md("### Detection")
code(DETECT)
md("### Submission assembly & validation")
code(SUBMISSION)
md("### Global ILP linking (tracksdata)")
code(LINK_ILP)
md("### Stage-B post-processing (pilkwang 0.897 graph surgery, CC0)\n"
   "Motion-relink + gap-close + gap2 recovery + safe divisions + short-track filter + line-fit smoothing. "
   "Adds **+0.040** on our dev subset (0.648→0.688), incl. first non-zero division score. Deps "
   "(`scipy`/`pandas`/`numpy`) are pre-installed on Kaggle — no extra wheels.")
code(POSTPROCESS)
md("### Offline image reader")
code(READER)
md("### Run → `submission.csv`")
code(RUN)

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "baseline-classic" / "03_submission_ilp.ipynb"
nbf.write(nb, out)
print("wrote", out)
