#!/usr/bin/env bash
# Build offline ILP wheels (tracksdata + pyscipopt + deps) as Linux wheels matching
# Kaggle's platform, into ./wheels/. Runs a Docker python image (needs internet HERE,
# not on Kaggle). Upload ./wheels as a private Kaggle dataset for the offline submission notebook.
# Usage: bash scripts/build_kaggle_wheels.sh [PYVER]   (default 3.11 = Kaggle's current Python)
set -euo pipefail
PYVER="${1:-3.11}"
NUMPY="${2:-2.0.2}"   # pin to Kaggle's numpy so every wheel resolves compatible (imagecodecs etc.)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/wheels"; rm -f "$ROOT"/wheels/*.whl "$ROOT"/wheels/*.zip "$ROOT"/wheels/*.tar.gz
echo "Building wheels for Python $PYVER, linux/amd64 (Kaggle platform). Confirm Kaggle: !python --version"
# --platform linux/amd64: Kaggle is x86_64 (emulated on Apple Silicon via qemu — slower but correct).
# pip wheel (not download): BUILDS tracksdata's wheel (pure-python) + fetches dep wheels.
# SETUPTOOLS_SCM_PRETEND_VERSION forces a CLEAN tracksdata version (no "+g<hash>" local part):
# the "+" is stripped by Kaggle dataset storage, which makes the wheel filename an invalid PEP440
# version that pip rejects. A clean version avoids the "+" entirely.
docker run --rm --platform linux/amd64 -e SETUPTOOLS_SCM_PRETEND_VERSION=0.1.0 \
  -v "$ROOT/wheels:/wheels" "python:${PYVER}" bash -c \
  "apt-get update -qq && apt-get install -y -qq git >/dev/null 2>&1 && \
   echo 'numpy==${NUMPY}' > /tmp/cons.txt && \
   pip wheel 'tracksdata @ git+https://github.com/royerlab/tracksdata@main' pyscipopt \
     -c /tmp/cons.txt -w /wheels"
echo "Done. $(ls "$ROOT"/wheels/*.whl | wc -l | tr -d ' ') wheels in ./wheels/"
ls "$ROOT"/wheels/ | grep -iE 'tracksdata|pyscipopt|rustworkx'
