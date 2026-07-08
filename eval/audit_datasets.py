"""Audit all train datasets: SCALE constancy, T_true presence/sanity, GT sizes.

Guards the two silent-failure modes the review flagged:
  * hardcoded SCALE wrong for some dataset -> 7µm matching silently off
  * missing estimated_number_of_nodes (T_true) -> row silently dropped from adj average

Also emits the T_true / GT-size distribution used to pick a stratified dev subset.

Run:  .venv-track/bin/python eval/audit_datasets.py --data-dir data/train --out eval/dataset_audit.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl
import tracksdata as td
import zarr
from geff import GeffMetadata


def _zarr_scale(zarr_path: Path) -> tuple[float, ...] | None:
    try:
        g = zarr.open_group(zarr_path, mode="r")
        attrs = dict(g.attrs)
        if "multiscales" in attrs:
            t = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
            if t.get("type") == "scale":
                return tuple(round(float(v), 6) for v in t["scale"][-3:])
    except Exception:
        pass
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/train", type=Path)
    ap.add_argument("--out", default="eval/dataset_audit.csv", type=Path)
    args = ap.parse_args()

    geffs = sorted(args.data_dir.glob("*.geff"))
    rows = []
    for i, gp in enumerate(geffs):
        name = gp.stem
        rec: dict = {"dataset": name}
        try:
            r = td.graph.IndexedRXGraph.from_geff(gp)
            g = r[0] if isinstance(r, tuple) else r
            ids = g.node_ids()
            rec["gt_nodes"] = g.num_nodes()
            rec["gt_edges"] = g.num_edges()
            rec["gt_divisions"] = int(sum(1 for d in g.out_degree(ids) if d >= 2))
            na = g.node_attrs(attr_keys=["t"])
            rec["t_max"] = int(na["t"].max())
        except Exception as e:
            rec["error"] = f"geff:{e}"
        try:
            m = GeffMetadata.read(gp)
            v = (m.extra or {}).get("estimated_number_of_nodes")
            rec["T_true"] = float(v) if v is not None else None
        except Exception as e:
            rec["T_true"] = None
            rec.setdefault("error", f"meta:{e}")
        rec["scale"] = json.dumps(_zarr_scale(args.data_dir / f"{name}.zarr"))
        rows.append(rec)
        if (i + 1) % 40 == 0:
            print(f"  ...{i+1}/{len(geffs)}")

    df = pl.DataFrame(rows, infer_schema_length=None)
    df.write_csv(args.out)

    print(f"\n{len(df)} datasets audited -> {args.out}")
    print("\n--- SCALE values (must be a single constant) ---")
    print(df["scale"].value_counts(sort=True))
    print("\n--- T_true present? ---")
    n_missing = df.filter(pl.col("T_true").is_null()).height
    print(f"missing T_true: {n_missing}")
    print("\n--- distributions ---")
    for col in ("T_true", "gt_nodes", "gt_edges", "gt_divisions"):
        if col in df.columns:
            s = df[col].drop_nulls()
            if s.len():
                print(f"{col:14} min={s.min():.0f} p25={s.quantile(0.25):.0f} "
                      f"med={s.median():.0f} p75={s.quantile(0.75):.0f} max={s.max():.0f} "
                      f"sum={s.sum():.0f}")
    print(f"\ndatasets with 0 divisions: {df.filter(pl.col('gt_divisions') == 0).height}")
    if "error" in df.columns:
        errs = df.filter(pl.col("error").is_not_null())
        if errs.height:
            print(f"\nERRORS ({errs.height}):"); print(errs.select("dataset", "error"))


if __name__ == "__main__":
    main()
