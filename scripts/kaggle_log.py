"""Pull + pretty-print a Kaggle kernel's log, and report its GPU-time — no more grep/sed on raw JSON.

    scripts/kaggle_log.py rkoren/biohub-blend-tuning              # pull latest run, print stdout + duration
    scripts/kaggle_log.py rkoren/biohub-blend-tuning --grep score # only lines matching a pattern
    scripts/kaggle_log.py /tmp/some.log                           # parse a local log file

The kernel log is a JSON array of {stream_name, time, data} records; `time` is seconds since start, so
its max ≈ the run's wall-clock GPU-time. Also appends the run's minutes to docs/gpu_tally.txt.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TALLY = Path(__file__).resolve().parent.parent / "docs" / "gpu_tally.txt"


def fetch_log(ref: str) -> Path:
    """`ref` is a kernel slug (owner/name) → download its output, return the .log path."""
    d = Path(tempfile.mkdtemp(prefix="kaggle_log_"))
    subprocess.run(["kaggle", "kernels", "output", ref, "-p", str(d)],
                   capture_output=True, text=True, check=True)
    logs = list(d.glob("*.log"))
    if not logs:
        sys.exit(f"no .log in output of {ref}")
    return logs[0]


def parse(log_path: Path):
    """Return (records, max_time_seconds). Each record = (stream, text)."""
    raw = log_path.read_text(errors="replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = [json.loads(m.group(0)) for m in re.finditer(r"\{[^{}]*\}", raw)]
    records, tmax = [], 0.0
    for rec in data:
        if not isinstance(rec, dict):
            continue
        tmax = max(tmax, float(rec.get("time", 0) or 0))
        text = rec.get("data", "")
        if text:
            records.append((rec.get("stream_name", "stdout"), text))
    return records, tmax


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="kernel slug (owner/name) or a local .log path")
    ap.add_argument("--grep", default=None, help="only print lines matching this regex")
    ap.add_argument("--stderr", action="store_true", help="include stderr stream too")
    ap.add_argument("--no-tally", action="store_true", help="don't append GPU-time to docs/gpu_tally.txt")
    args = ap.parse_args()

    p = Path(args.target)
    ref = None
    if not p.exists():
        ref = args.target
        p = fetch_log(ref)

    records, tmax = parse(p)
    pat = re.compile(args.grep) if args.grep else None
    for stream, text in records:
        if stream == "stderr" and not args.stderr:
            continue
        for line in text.splitlines():
            if pat is None or pat.search(line):
                print(line)

    minutes = tmax / 60.0
    print(f"\n[run wall-clock ≈ {minutes:.1f} min ({tmax:.0f}s)]", file=sys.stderr)
    if ref and not args.no_tally:
        TALLY.parent.mkdir(exist_ok=True)
        with TALLY.open("a") as f:
            f.write(f"{ref}\t{minutes:.1f}\n")
        total = sum(float(l.split("\t")[1]) for l in TALLY.read_text().splitlines() if "\t" in l)
        print(f"[GPU tally: +{minutes:.1f} min → {total/60:.2f} hr total across {len(TALLY.read_text().splitlines())} runs]",
              file=sys.stderr)


if __name__ == "__main__":
    main()
