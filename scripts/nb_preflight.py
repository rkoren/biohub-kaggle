"""Pre-flight a notebook before pushing it to Kaggle — catch the bugs that waste kernel runs.

Concatenates the code cells in execution order and runs pyflakes, so a name used before it's defined
(the `Path` / `PRIMARY_ARTIFACT_MANIFEST` cell-loss bugs that each cost us a failed submission run) is
caught here, locally, in a second. Also flags syntax errors, and for submission notebooks checks
offline-safety (no clone / online pip) + that it writes submission.csv.

    .venv-track/bin/python scripts/nb_preflight.py notebooks/blend-09/blend09_submit.ipynb --submit

Exit code is nonzero if any ERROR is found — wire it in before `kaggle kernels push`.
"""
from __future__ import annotations

import argparse
import ast
import io
import sys
from pathlib import Path

import nbformat
from pyflakes.api import check
from pyflakes.reporter import Reporter


def code_cells(nb) -> list[str]:
    out = []
    for c in nb.cells:
        if c.cell_type != "code":
            continue
        # drop Jupyter shell/magic lines (`!cmd`, `%magic`) that aren't valid Python
        lines = [("" if l.lstrip().startswith(("!", "%")) else l) for l in c.source.splitlines()]
        out.append("\n".join(lines))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook", type=Path)
    ap.add_argument("--submit", action="store_true", help="also enforce offline-safety + submission.csv output")
    args = ap.parse_args()

    nb = nbformat.read(args.notebook, as_version=4)
    cells = code_cells(nb)
    errors, warnings = [], []

    # 1) syntax
    for i, src in enumerate(cells):
        try:
            ast.parse(src)
        except SyntaxError as e:
            errors.append(f"syntax error in code cell {i}: {e}")

    # 2) undefined names (pyflakes on cells concatenated in execution order).
    # Prepend a preamble defining IPython-injected notebook builtins (available in the Kaggle
    # runtime but invisible to pyflakes) so they aren't false-positive "undefined name" errors.
    preamble = "display = Image = HTML = Markdown = get_ipython = None  # ipython builtins\n"
    combined = preamble + "\n".join(cells)
    buf, errbuf = io.StringIO(), io.StringIO()
    check(combined, str(args.notebook), Reporter(buf, errbuf))
    for line in (buf.getvalue() + errbuf.getvalue()).splitlines():
        msg = line.split(":", 3)[-1].strip() if ":" in line else line
        if "undefined name" in line:
            errors.append(f"undefined name → {msg}")
        elif line.strip():
            warnings.append(msg)

    # 3) submission-specific: offline-safe + writes submission.csv
    if args.submit:
        joined = "\n".join(cells)
        if "git clone" in joined:
            errors.append("submission notebook contains `git clone` (needs internet — breaks Internet-OFF run)")
        import re
        if re.search(r"pip install(?!.*--no-index)", joined) and "ensure_dependencies" not in joined:
            errors.append("submission notebook has an online `pip install` (no --no-index / offline path)")
        if "https://" in joined or "requests.get" in joined:
            warnings.append("submission notebook references an https URL — verify it isn't fetched at runtime")
        if not ("SUBMISSION_PATH" in joined and "writerow" in joined) and "submission.csv" not in joined:
            errors.append("submission notebook does not appear to write submission.csv")

    name = args.notebook.name
    for w in warnings:
        print(f"  ⚠ {w}")
    if errors:
        print(f"\n✗ {name}: {len(errors)} ERROR(s) — do NOT push:")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    print(f"\n✓ {name}: pre-flight OK ({len(cells)} code cells, {len(warnings)} warning(s)) — safe to push.")


if __name__ == "__main__":
    main()
