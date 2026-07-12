"""Generate a self-contained offline Kaggle submission notebook from pipeline/.

A Kaggle code-competition notebook runs with internet OFF and cannot import our
repo, so we INLINE the verified pipeline source (config/detect/link/submission)
into notebook cells — guaranteeing the submission notebook never drifts from the
reviewed package. The only substitution is the image reader: on Kaggle we use a
dependency-light blosc2 chunk reader (robust to zarr-version mismatches) instead
of the dev-time `zarr` library. The reader was verified byte-identical to zarr
on all 4 test datasets.

Run:  .venv-track/bin/python notebooks/build_submission_nb.py
"""
from pathlib import Path
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "pipeline"


def strip_source(path: Path) -> str:
    """Inline a package module: drop __future__ + relative imports (all names
    share the notebook's global namespace); keep external imports + code."""
    out = []
    for line in path.read_text().splitlines():
        if line.startswith("from __future__"):
            continue
        if re.match(r"^from \.\w* import", line) or re.match(r"^from \. import", line):
            continue
        out.append(line)
    return "\n".join(out).strip("\n")


CONFIG = strip_source(PKG / "config.py")
DETECT = strip_source(PKG / "detect.py")
LINK = strip_source(PKG / "link.py")
SUBMISSION = strip_source(PKG / "submission.py")

READER = '''
# --- Offline image reader (blosc2 chunk reader; no zarr dependency) -----------
# Robust to Kaggle's zarr version. Verified byte-identical to the zarr library
# on all 4 test datasets (shape/dtype/orientation).
import json as _json

def read_array_meta(zarr_path: Path):
    """Return (array_dir, shape, dtype, chunk_shape) for the 4D image array."""
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
            array_dir / f"{t}.0.0.0",
            array_dir / str(t) / "0" / "0" / "0"]


def load_volume(array_dir: Path, shape, dtype, t: int) -> np.ndarray:
    """Load one timepoint as (Z, Y, X)."""
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
# --- Run the pipeline over every test dataset --------------------------------
import time
from collections import Counter

def run_dataset(zarr_path: Path, dataset: str, cfg: PipelineConfig, verbose=True):
    array_dir, shape, dtype, _ = read_array_meta(zarr_path)
    T, Z, Y, X = shape
    t0 = time.time()
    node_rows, edge_rows, node_scores = [], [], {}
    prev_ids, prev_xyz, prev_count, next_id = [], np.empty((0, 3), np.int32), None, 1
    counts, div_est = [], 0
    for t in range(T):
        vol = load_volume(array_dir, shape, dtype, t)
        coords, scores = detect_cells(vol, cfg, prev_count=prev_count)
        del vol; gc.collect()
        if len(coords):
            o = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0])); coords, scores = coords[o], scores[o]
        curr_ids = list(range(next_id, next_id + len(coords))); next_id += len(coords)
        for nid, zyx, sc in zip(curr_ids, coords, scores):
            node_rows.append(node_row(dataset, nid, t, zyx)); node_scores[int(nid)] = float(sc)
        if t > 0:
            links = link_frames(prev_ids, prev_xyz, curr_ids, coords, cfg)
            for s, u in links:
                edge_rows.append(edge_row(dataset, s, u))
            div_est += sum(1 for c in Counter(s for s, _ in links).values() if c >= 2)
        prev_ids, prev_xyz, prev_count = curr_ids, coords, len(coords)
        counts.append(len(coords))
    node_rows, edge_rows, ps = prune_isolated(node_rows, edge_rows, node_scores, cfg)
    if verbose:
        print(f"  [{dataset}] {time.time()-t0:.1f}s | nodes={len(node_rows)} edges={len(edge_rows)} "
              f"isolated_removed={ps['removed_isolated']} div_est={div_est}")
    return node_rows + edge_rows


def resolve_test_dir() -> Path:
    for c in [Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/test"),
              Path("/kaggle/input/biohub-cell-tracking-during-development/test"),
              Path("data/test"), Path("../data/test")]:
        if c.exists() and list(c.glob("*.zarr")):
            return c
    hits = [Path(p) for p in __import__("glob").glob("/kaggle/input/**/test", recursive=True)]
    for h in hits:
        if list(h.glob("*.zarr")):
            return h
    raise FileNotFoundError("No test/*.zarr directory found")


TEST_DIR = resolve_test_dir()
OUT = Path("/kaggle/working/submission.csv") if Path("/kaggle/working").exists() else Path("submission.csv")
cfg = PipelineConfig()   # defaults = starter v2_precision profile
names = sorted(p.name[:-5] for p in TEST_DIR.iterdir() if p.name.endswith(".zarr"))
print(f"TEST_DIR={TEST_DIR}  datasets={names}")

t0 = time.time()
rows = []
for i, n in enumerate(names, 1):
    print(f"[{i}/{len(names)}] {n}")
    rows.extend(run_dataset(TEST_DIR / f"{n}.zarr", n, cfg))
sub = assemble(rows)
validate(sub, expected_datasets=set(names))
save(sub, OUT)
print(f"\\nWrote {OUT}: {len(sub):,} rows "
      f"({int((sub.row_type=='node').sum()):,} nodes, {int((sub.row_type=='edge').sum()):,} edges) "
      f"in {(time.time()-t0)/60:.1f} min")
sub.head()
'''.strip("\n")

# --- assemble notebook ---
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md("""
# 🧬 Cell Tracking — Baseline Submission (offline, self-contained)

Produces `submission.csv` for the [Biohub Cell Tracking](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
code competition. **Internet off, self-contained** — no external repo or downloads.

Classical pipeline: full-Z detection (block-mean → threshold → local-maxima → centroid refinement → physical NMS)
→ Hungarian linking with a conservative division pass → isolated-node pruning. Reads OME-Zarr chunks directly
via `blosc2` (robust to Kaggle's zarr version). Runs the 4 test volumes in well under a minute — far inside the 12h budget.

*Generated from the `pipeline/` package (`notebooks/build_submission_nb.py`); config defaults = the `v2_precision` profile.*
""")
md("### Imports")
code("import gc\nfrom pathlib import Path\nimport numpy as np\nimport pandas as pd")
md("### Configuration (all tunable knobs)")
code(CONFIG)
md("### Detection")
code(DETECT)
md("### Linking, divisions & pruning")
code(LINK)
md("### Submission assembly & validation")
code(SUBMISSION)
md("### Offline image reader")
code(READER)
md("### Run → `submission.csv`")
code(RUN)

nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                  "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "baseline-classic" / "02_submission_baseline.ipynb"
nbf.write(nb, out)
print("wrote", out)
