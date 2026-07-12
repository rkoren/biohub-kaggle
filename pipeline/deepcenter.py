"""DeepCenter veto detector — standalone port of the blend's 2nd-detector repair gate.

A small 3D-UNet center-prior (`biohub-deepcenter-unet3d-center-prior-v1`, CC0) that scores a point:
high heatmap value ⇒ a real cell centre. The ~0.90 blend uses it to VETO candidate repair points
(gap-close midpoints, safe-division candidates) so only real cells get added. Ported verbatim from
`notebooks/gpu-start/pilkwang_09_reference/cells/cell_11.py` (the architecture is self-contained there).

Usage (wire into postprocess via its VETO_FN hook):
    import deepcenter as DC
    bundle = DC.load(Path.home()/ "biohub-deepcenter-unet3d-center-prior-v1"/"weights"/"full_frame_center"/"best.pt")
    scorer = DC.PointScorer(bundle, data_dir="data/train")
    postprocess.VETO_FN = lambda ds, t, pt, kind: scorer.accept(ds, t, pt, kind)
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class _ConvBlock3d(torch.nn.Module):
    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        g = min(8, cout)
        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(cin, cout, 3, padding=1, bias=False), torch.nn.GroupNorm(g, cout), torch.nn.SiLU(inplace=True),
            torch.nn.Conv3d(cout, cout, 3, padding=1, bias=False), torch.nn.GroupNorm(g, cout), torch.nn.SiLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class DeepCenterUNet3D(torch.nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 24) -> None:
        super().__init__()
        c = int(base_channels)
        self.enc1 = _ConvBlock3d(in_channels, c); self.down1 = torch.nn.MaxPool3d(2, 2)
        self.enc2 = _ConvBlock3d(c, c * 2); self.down2 = torch.nn.MaxPool3d(2, 2)
        self.enc3 = _ConvBlock3d(c * 2, c * 4); self.down3 = torch.nn.MaxPool3d(2, 2)
        self.bottleneck = _ConvBlock3d(c * 4, c * 8)
        self.up3 = torch.nn.ConvTranspose3d(c * 8, c * 4, 2, 2); self.dec3 = _ConvBlock3d(c * 8, c * 4)
        self.up2 = torch.nn.ConvTranspose3d(c * 4, c * 2, 2, 2); self.dec2 = _ConvBlock3d(c * 4, c * 2)
        self.up1 = torch.nn.ConvTranspose3d(c * 2, c, 2, 2); self.dec1 = _ConvBlock3d(c * 2, c)
        self.head = torch.nn.Conv3d(c, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.down1(e1)); e3 = self.enc3(self.down2(e2))
        b = self.bottleneck(self.down3(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def load(checkpoint_path: str | Path, device: str = "cpu") -> dict:
    dev = torch.device(device)
    ck = torch.load(str(checkpoint_path), map_location=dev, weights_only=False)
    cfg = SimpleNamespace(**ck.get("config", {}))
    model = DeepCenterUNet3D(base_channels=int(getattr(cfg, "base_channels", 24)))
    model.load_state_dict(ck["model_state"]); model.to(dev); model.eval()
    return {"model": model, "cfg": cfg, "device": dev}


def _pool_xy(vol: np.ndarray, factor: int) -> np.ndarray:
    if factor <= 1:
        return vol.astype(np.float32, copy=False)
    z, y, x = vol.shape
    y2, x2 = (y // factor) * factor, (x // factor) * factor
    return vol[:, :y2, :x2].astype(np.float32, copy=False).reshape(z, y2 // factor, factor, x2 // factor, factor).mean(axis=(2, 4))


def _normalize(vol: np.ndarray, cfg) -> np.ndarray:
    vol = np.asarray(vol, np.float32)
    lo = float(np.percentile(vol, float(getattr(cfg, "norm_lo_pct", 50.0))))
    hi = float(np.percentile(vol, float(getattr(cfg, "norm_hi_pct", 99.5))))
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return np.zeros_like(vol, np.float32)
    return np.clip((vol - lo) / (hi - lo), float(getattr(cfg, "norm_clip_lo", -0.5)),
                   float(getattr(cfg, "norm_clip_hi", 6.0))).astype(np.float32)


class PointScorer:
    """Scores/vetoes candidate repair points. Reads frames from ``data_dir/<dataset>.zarr``."""

    def __init__(self, bundle: dict, data_dir: str | Path = "data/train",
                 gap_threshold: float = 0.10, div_threshold: float = 0.12,
                 win_z: int = 1, win_yx: int = 2):
        self.b = bundle
        self.data_dir = Path(data_dir)
        self.gap_threshold, self.div_threshold = gap_threshold, div_threshold
        self.win_z, self.win_yx = win_z, win_yx
        self._frames: dict = {}
        self._heat: dict = {}

    def _read_frame(self, dataset: str, t: int) -> np.ndarray:
        key = (dataset, t)
        if key in self._frames:
            return self._frames[key]
        zpath = self.data_dir / f"{dataset}.zarr"
        meta = json.loads((zpath / "0" / "zarr.json").read_text())
        shape = tuple(int(v) for v in meta["shape"]); dtype = np.dtype(meta["data_type"])
        fshape = shape[1:]
        try:
            import blosc2
            raw = (zpath / "0" / "c" / str(t) / "0" / "0" / "0").read_bytes()
            arr = np.frombuffer(blosc2.decompress(raw), dtype=dtype)
            frame = arr.reshape(fshape).copy() if arr.size == int(np.prod(fshape)) else None
        except Exception:
            frame = None
        if frame is None:
            import zarr
            frame = np.asarray(zarr.open(zpath / "0", mode="r")[t])
        if len(self._frames) > 8:
            self._frames.pop(next(iter(self._frames)))
        self._frames[key] = frame
        return frame

    def _heatmap(self, dataset: str, t: int) -> np.ndarray:
        key = (dataset, t)
        if key in self._heat:
            return self._heat[key]
        cfg = self.b["cfg"]
        pool = int(getattr(cfg, "pool_factor", 4))
        img = _normalize(_pool_xy(self._read_frame(dataset, t), pool), cfg)
        with torch.no_grad():
            tensor = torch.from_numpy(img[None, None, ...]).to(device=self.b["device"], dtype=torch.float32)
            heat = torch.sigmoid(self.b["model"](tensor))[0, 0].cpu().numpy().astype(np.float32)
        if len(self._heat) > 8:
            self._heat.pop(next(iter(self._heat)))
        self._heat[key] = heat
        return heat

    def score(self, dataset: str, t: int, point) -> float | None:
        heat = self._heatmap(dataset, int(t))
        if heat.size == 0:
            return None
        pool = int(getattr(self.b["cfg"], "pool_factor", 4))
        z = int(round(float(point[0])))
        y = int(round(float(point[1]) / max(pool, 1)))
        x = int(round(float(point[2]) / max(pool, 1)))
        z0, z1 = max(0, z - self.win_z), min(heat.shape[0], z + self.win_z + 1)
        y0, y1 = max(0, y - self.win_yx), min(heat.shape[1], y + self.win_yx + 1)
        x0, x1 = max(0, x - self.win_yx), min(heat.shape[2], x + self.win_yx + 1)
        patch = heat[z0:z1, y0:y1, x0:x1]
        return float(np.max(patch)) if patch.size else None

    def accept(self, dataset: str, t: int, point, kind: str) -> bool:
        """True = keep the repair; False = veto it. kind in {'gap','div'}."""
        thr = self.gap_threshold if kind == "gap" else self.div_threshold
        s = self.score(dataset, t, point)
        return True if s is None else (s >= thr)
