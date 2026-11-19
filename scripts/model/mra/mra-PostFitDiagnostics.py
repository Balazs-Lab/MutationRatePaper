#!/usr/bin/env python3
from __future__ import annotations

"""
Evaluate mra mutation-supply fit: rate residuals and count calibration by mutation type.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DEFAULT_WT_ONLY = ("C->A", "G->T", "deletion")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mra post-fit diagnostics on supply table.")
    p.add_argument(
        "--model-output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "models" / "mra" / "production",
    )
    p.add_argument("--supply-csv", type=Path, default=None)
    p.add_argument("--outdir", type=Path, default=None)
    return p.parse_args()


def _resolve_rate_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pred_rate_mean" not in out.columns:
        if "bio_rate_mean" in out.columns:
            out["pred_rate_mean"] = pd.to_numeric(out["bio_rate_mean"], errors="coerce")
        else:
            raise ValueError("Supply table needs bio_rate_mean or pred_rate_mean.")
    if "pred_rate_median" not in out.columns:
        out["pred_rate_median"] = out.get("bio_rate_median", out["pred_rate_mean"])
    if "pred_rate_lo95" not in out.columns:
        out["pred_rate_lo95"] = out.get("bio_rate_lo95", np.nan)
    if "pred_rate_hi95" not in out.columns:
        out["pred_rate_hi95"] = out.get("bio_rate_hi95", np.nan)
    return out


def _ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mut_type" not in out.columns and "Mut_Type" in out.columns:
        out["mut_type"] = out["Mut_Type"]
    if "count" not in out.columns and "wt_count" in out.columns:
        out["count"] = out["wt_count"]
    if "coverage" not in out.columns and "wt_coverage" in out.columns:
        out["coverage"] = out["wt_coverage"]
    required = ["ref_position", "mut_type", "count", "coverage"]
    miss = [c for c in required if c not in out.columns]
    if miss:
        raise ValueError(f"Supply table missing required columns: {miss}")
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out["coverage"] = pd.to_numeric(out["coverage"], errors="coerce")
    out = out[(out["coverage"] > 0) & out["count"].notna()].copy()
    out["obs_rate"] = out["count"] / out["coverage"]
    if "error_rate_mean" in out.columns and "wt_total_rate_mean" in out.columns:
        out["pred_rate_mean"] = out["wt_total_rate_mean"]
    out["pred_count_mean"] = out["pred_rate_mean"] * out["coverage"]
    out["rate_residual"] = out["obs_rate"] - out["pred_rate_mean"]
    return out


def main() -> None:
    args = parse_args()
    model_dir = args.model_output_dir
    supply_path = args.supply_csv or (model_dir / "mutation_supply_site_mutation_table.csv")
    outdir = args.outdir or (model_dir / "diagnostics")
    outdir.mkdir(parents=True, exist_ok=True)

    df = _ensure_schema(_resolve_rate_columns(pd.read_csv(supply_path)))

    meta_path = model_dir / "model_metadata.json"
    metadata = {}
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
    wt_only = tuple(metadata.get("wt_only_mut_types", DEFAULT_WT_ONLY))

    agg_spec: dict = {
        "n": ("ref_position", "size"),
        "obs_count": ("count", "sum"),
        "pred_count": ("pred_count_mean", "sum"),
        "rmse_rate": ("rate_residual", lambda s: float(np.sqrt(np.mean(np.square(s))))),
        "mean_obs_rate": ("obs_rate", "mean"),
        "mean_pred_rate": ("pred_rate_mean", "mean"),
    }
    if "control_corrected" in df.columns:
        agg_spec["control_corrected"] = ("control_corrected", "first")
    if "correction_scheme" in df.columns:
        agg_spec["correction_scheme"] = ("correction_scheme", "first")
    if "uses_nucleotide_context" in df.columns:
        agg_spec["uses_nucleotide_context"] = ("uses_nucleotide_context", "first")
    by_mut = df.groupby("mut_type", as_index=False).agg(**agg_spec)
    by_mut["ppc_ratio"] = by_mut["pred_count"] / by_mut["obs_count"].replace(0, np.nan)
    by_mut.to_csv(outdir / "calibration_by_mutation_type.csv", index=False)

    plt.figure(figsize=(6, 5))
    x = by_mut["obs_count"].to_numpy() + 0.5
    y = by_mut["pred_count"].to_numpy() + 0.5
    plt.scatter(x, y)
    lim = [0.5, max(x.max(), y.max()) * 1.2]
    plt.plot(lim, lim, linestyle="--", color="gray")
    for _, r in by_mut.iterrows():
        suffix = "" if r.get("control_corrected", True) else " [WT-only]"
        plt.text(r["obs_count"] + 0.5, r["pred_count"] + 0.5, str(r["mut_type"]) + suffix, fontsize=7)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Observed WT count + 0.5")
    plt.ylabel("Predicted WT count (wt_total_rate * coverage) + 0.5")
    plt.title("mra WT count calibration by mutation type")
    plt.tight_layout()
    plt.savefig(outdir / "wt_count_calibration_by_mut_type.png", dpi=300)
    plt.close()

    summary = {
        "model": metadata.get("model", "mra"),
        "wt_only_mut_types": list(wt_only),
        "context_mut_types": metadata.get("context_mut_types"),
        "no_context_mut_types": metadata.get("no_context_mut_types"),
        "spline_df": metadata.get("spline_df"),
        "n_rows": int(len(df)),
        "n_mutation_types": int(df["mut_type"].nunique()),
        "overall_rate_rmse": float(np.sqrt(np.mean(np.square(df["rate_residual"])))),
        "overall_wt_ppc_ratio": float(by_mut["pred_count"].sum() / by_mut["obs_count"].sum()),
    }
    with open(outdir / "diagnostics_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote diagnostics to {outdir}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
