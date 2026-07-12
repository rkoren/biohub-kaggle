"""Pipeline configuration, sourced from menu.yaml.

Every tunable knob lives here as one dataclass so experiments are a single
object to log, diff, and sweep. Defaults match the starter's `v2_precision`
profile (public LB ≈ 0.581).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class PipelineConfig:
    # physical voxel scale, microns per voxel (z, y, x)
    scale: tuple[float, float, float] = (1.625, 0.40625, 0.40625)

    # --- detection ---
    xy_ds: int = 4                    # XY block-mean factor
    smooth_sigma: float = 0.95
    min_peak_dist: int = 3
    thresh_rel: float = 0.34
    min_rel_contrast: float = 0.08
    nms_radius_um: float = 2.65
    border_z: int = 1
    border_yx: int = 2
    border_keep_quantile: float = 0.70
    max_frame_count_mult: float = 1.70
    max_frame_count_add: int = 45
    max_nodes_per_frame: int = 20000

    # --- linking ---
    max_link_dist_um: float = 11.0
    link_method: str = "greedy"       # "greedy" (Hungarian) | "ilp" (global, tracksdata)
    link_n_neighbors: int = 6         # candidate neighbours per node for ILP
    link_delta_t: int = 1             # ILP gap-closing: also link t→t+Δ (Δ>1 bridges missed detections)
    ilp_appearance: float = 0.1
    ilp_disappearance: float = 0.1
    ilp_division: float = 1.0
    ilp_timeout: float = 600.0        # per-dataset ILP solve timeout (s) — guards the 12h budget

    # --- divisions ---
    detect_divisions: bool = True
    div_parent_dist_um: float = 8.75
    div_sister_dist_um: float = 6.25
    div_min_count_gain: int = 1

    # --- node pruning ---
    prune_isolated_nodes: bool = True
    keep_strong_isolated: bool = False
    strong_isolated_quantile: float = 0.97

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_menu(cls, menu: dict) -> "PipelineConfig":
        """Build from a parsed menu.yaml dict (scale/detect/link/divisions/prune)."""
        sc = menu.get("scale", {})
        d = menu.get("detect", {})
        lk = menu.get("link", {})
        dv = menu.get("divisions", {})
        pr = menu.get("prune", {})
        return cls(
            scale=(sc.get("z", 1.625), sc.get("y", 0.40625), sc.get("x", 0.40625)),
            xy_ds=d.get("xy_ds", 4),
            smooth_sigma=d.get("smooth_sigma", 0.95),
            min_peak_dist=d.get("min_peak_dist", 3),
            thresh_rel=d.get("thresh_rel", 0.34),
            min_rel_contrast=d.get("min_rel_contrast", 0.08),
            nms_radius_um=d.get("nms_radius_um", 2.65),
            border_z=d.get("border_z", 1),
            border_yx=d.get("border_yx", 2),
            border_keep_quantile=d.get("border_keep_quantile", 0.70),
            max_frame_count_mult=d.get("max_frame_count_mult", 1.70),
            max_frame_count_add=d.get("max_frame_count_add", 45),
            max_nodes_per_frame=d.get("max_nodes_per_frame", 20000),
            max_link_dist_um=lk.get("max_link_dist_um", 11.0),
            link_method=lk.get("method", "greedy"),
            link_n_neighbors=lk.get("n_neighbors", 6),
            link_delta_t=lk.get("delta_t", 1),
            ilp_appearance=lk.get("ilp_appearance", 0.1),
            ilp_disappearance=lk.get("ilp_disappearance", 0.1),
            ilp_division=lk.get("ilp_division", 1.0),
            ilp_timeout=lk.get("ilp_timeout", 600.0),
            detect_divisions=dv.get("enabled", True),
            div_parent_dist_um=dv.get("parent_dist_um", 8.75),
            div_sister_dist_um=dv.get("sister_dist_um", 6.25),
            div_min_count_gain=dv.get("min_count_gain", 1),
            prune_isolated_nodes=pr.get("isolated_nodes", True),
            keep_strong_isolated=pr.get("keep_strong_isolated", False),
            strong_isolated_quantile=pr.get("strong_isolated_quantile", 0.97),
        )

    @classmethod
    def load(cls, menu_path: str | Path = "menu.yaml") -> "PipelineConfig":
        import yaml
        with open(menu_path) as f:
            return cls.from_menu(yaml.safe_load(f))
