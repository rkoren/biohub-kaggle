"""OME-Zarr volume loading — one (Z, Y, X) timepoint at a time.

The competition images are OME-Zarr `(T, Z, Y, X)` with the array under group
`0`. We stream a single timepoint into memory and release it, keeping the
footprint small (the same discipline the 12h/offline Kaggle notebook needs).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import zarr


class VolumeSeries:
    """Lazy accessor over a dataset's 4D image array."""

    def __init__(self, zarr_path: str | Path):
        self.path = Path(zarr_path)
        grp = zarr.open_group(self.path, mode="r")
        self._arr = grp["0"]                       # (T, Z, Y, X)
        self.shape: tuple[int, int, int, int] = tuple(self._arr.shape)  # type: ignore
        self.dtype = self._arr.dtype

    @property
    def T(self) -> int:
        return self.shape[0]

    def volume(self, t: int) -> np.ndarray:
        """Return timepoint ``t`` as a numpy (Z, Y, X) array."""
        return np.asarray(self._arr[t])


def list_dataset_names(data_dir: str | Path) -> list[str]:
    """Sorted dataset stems (folder names without `.zarr`) under ``data_dir``."""
    data_dir = Path(data_dir)
    return sorted(p.name[:-5] for p in data_dir.iterdir() if p.name.endswith(".zarr"))
