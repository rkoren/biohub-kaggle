"""Cell detection in a single 3D volume.

Full-Z, XY block-mean → smooth → robust threshold → local-maxima peaks →
intensity-weighted centroid refinement → physical NMS → border/count guards.

The design bias is *precision + accurate centroids*: the metric matches nodes to
sparse GT with a hard 7µm tolerance (localization is effectively binary), and
over-predicting past the dense node count only costs. So we keep stable centres,
not every bright speck.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter
from scipy.spatial import cKDTree

from .config import PipelineConfig

try:
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    _SKIMAGE = True
except Exception:  # pragma: no cover
    peak_local_max = None
    threshold_otsu = None
    _SKIMAGE = False


def block_mean_xy(vol: np.ndarray, factor: int) -> np.ndarray:
    """Average-pool XY by ``factor`` while preserving Z resolution."""
    Z, Y, X = vol.shape
    Y2, X2 = (Y // factor) * factor, (X // factor) * factor
    x = vol[:, :Y2, :X2].astype(np.float32, copy=False)
    return x.reshape(Z, Y2 // factor, factor, X2 // factor, factor).mean(axis=(2, 4))


def robust_threshold(sm: np.ndarray, thresh_rel: float) -> tuple[float, float, float]:
    """Otsu + relative-rise threshold. Returns (threshold, background, dyn_range)."""
    bg = float(np.median(sm))
    hi = float(np.percentile(sm, 99.9))
    dyn = max(hi - bg, 1e-6)
    rel_thr = bg + thresh_rel * dyn
    try:
        otsu = float(threshold_otsu(sm)) if _SKIMAGE else float(np.percentile(sm, 96.0))
    except Exception:
        otsu = float(np.percentile(sm, 96.0))
    return max(otsu, rel_thr), bg, dyn


def _fallback_peaks(sm: np.ndarray, threshold_abs: float, min_distance: int) -> np.ndarray:
    size = 2 * int(min_distance) + 1
    mx = maximum_filter(sm, size=(size, size, size), mode="nearest")
    mask = (sm >= mx) & (sm > threshold_abs)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return coords.astype(np.int32)
    vals = sm[coords[:, 0], coords[:, 1], coords[:, 2]]
    return coords[np.argsort(-vals)].astype(np.int32)


def physical_nms(coords_vox: np.ndarray, scores: np.ndarray, scale: np.ndarray,
                 radius_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Greedy non-max suppression in physical (micron) space."""
    if len(coords_vox) <= 1:
        return coords_vox, scores
    pts = coords_vox.astype(np.float64) * scale[None, :]
    order = np.argsort(-scores)
    tree = cKDTree(pts)
    suppressed = np.zeros(len(coords_vox), dtype=bool)
    keep = []
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in tree.query_ball_point(pts[i], r=radius_um):
            suppressed[j] = True
    keep = np.array(keep, dtype=np.int64)
    return coords_vox[keep], scores[keep]


def refine_centroid(vol: np.ndarray, approx_zyx: np.ndarray) -> tuple[np.ndarray, float]:
    """Intensity-weighted centroid refinement in original voxel coordinates."""
    Z, Y, X = vol.shape
    z, y, x = (int(round(v)) for v in approx_zyx)
    rz, ry, rx = 2, 5, 5
    z0, z1 = max(0, z - rz), min(Z, z + rz + 1)
    y0, y1 = max(0, y - ry), min(Y, y + ry + 1)
    x0, x1 = max(0, x - rx), min(X, x + rx + 1)
    crop = vol[z0:z1, y0:y1, x0:x1].astype(np.float32, copy=False)
    if crop.size == 0:
        return np.array([z, y, x], dtype=np.float64), 0.0
    bg = float(np.percentile(crop, 20.0))
    w = crop - bg
    w[w < 0] = 0
    total = float(w.sum())
    if total <= 1e-6:
        loc = np.unravel_index(int(np.argmax(crop)), crop.shape)
        return np.array([z0 + loc[0], y0 + loc[1], x0 + loc[2]], dtype=np.float64), float(crop[loc])
    zz, yy, xx = np.indices(crop.shape)
    refined = np.array([
        z0 + float((zz * w).sum() / total),
        y0 + float((yy * w).sum() / total),
        x0 + float((xx * w).sum() / total),
    ], dtype=np.float64)
    return refined, float(w.max())


def detect_cells(vol: np.ndarray, cfg: PipelineConfig,
                 prev_count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return integer centroid coords (z, y, x) and detector scores for one volume."""
    Z, Y, X = vol.shape
    scale = np.asarray(cfg.scale, dtype=np.float64)
    ds = block_mean_xy(vol, cfg.xy_ds)
    sm = gaussian_filter(ds, sigma=cfg.smooth_sigma, mode="nearest")
    threshold_abs, bg, dyn = robust_threshold(sm, cfg.thresh_rel)

    if _SKIMAGE:
        coords_ds = peak_local_max(sm, min_distance=cfg.min_peak_dist,
                                   threshold_abs=threshold_abs, exclude_border=False).astype(np.int32)
    else:
        coords_ds = _fallback_peaks(sm, threshold_abs, cfg.min_peak_dist)

    if coords_ds.size == 0:
        flat = np.argpartition(sm.ravel(), -3)[-3:]
        coords_ds = np.array(np.unravel_index(flat, sm.shape)).T.astype(np.int32)

    peak_scores = sm[coords_ds[:, 0], coords_ds[:, 1], coords_ds[:, 2]].astype(np.float32)
    rel_contrast = (peak_scores - bg) / max(dyn, 1e-6)
    keep = rel_contrast >= cfg.min_rel_contrast
    coords_ds, peak_scores = coords_ds[keep], peak_scores[keep]
    if len(coords_ds) == 0:
        return np.empty((0, 3), np.int32), np.empty((0,), np.float32)

    # XY-block grid → original coords (Z unchanged).
    approx = coords_ds.astype(np.float64)
    approx[:, 1] = approx[:, 1] * cfg.xy_ds + (cfg.xy_ds - 1) / 2.0
    approx[:, 2] = approx[:, 2] * cfg.xy_ds + (cfg.xy_ds - 1) / 2.0

    refined, refined_scores = [], []
    for a, s in zip(approx, peak_scores):
        r, rs = refine_centroid(vol, a)
        refined.append(r)
        refined_scores.append(max(float(s), rs))
    coords = np.vstack(refined).astype(np.float64)
    scores = np.array(refined_scores, dtype=np.float32)

    # Drop weak boundary peaks, keep confident border cells.
    if len(coords):
        cz, cy, cx = coords[:, 0], coords[:, 1], coords[:, 2]
        border = ((cz <= cfg.border_z) | (cz >= (Z - 1 - cfg.border_z)) |
                  (cy <= cfg.border_yx) | (cy >= (Y - 1 - cfg.border_yx)) |
                  (cx <= cfg.border_yx) | (cx >= (X - 1 - cfg.border_yx)))
        floor = float(np.quantile(scores, cfg.border_keep_quantile)) if len(scores) > 8 else -np.inf
        keep = (~border) | (scores >= floor)
        coords, scores = coords[keep], scores[keep]

    coords, scores = physical_nms(coords, scores, scale, cfg.nms_radius_um)

    # Count stabilizer: trim only implausible frame-to-frame explosions.
    if (prev_count is not None and prev_count >= 8
            and len(coords) > prev_count * cfg.max_frame_count_mult + cfg.max_frame_count_add):
        cap = int(prev_count * cfg.max_frame_count_mult + cfg.max_frame_count_add)
        order = np.argsort(-scores)[:cap]
        coords, scores = coords[order], scores[order]
    if len(coords) > cfg.max_nodes_per_frame:
        order = np.argsort(-scores)[:cfg.max_nodes_per_frame]
        coords, scores = coords[order], scores[order]

    coords = np.rint(coords).astype(np.int32)
    if len(coords):
        coords[:, 0] = np.clip(coords[:, 0], 0, Z - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, Y - 1)
        coords[:, 2] = np.clip(coords[:, 2], 0, X - 1)
    return coords, scores.astype(np.float32)
