#!/usr/bin/env python3
from __future__ import annotations

"""
Build full mutation-supply table from mra fit outputs (bio_rate posterior summaries).
"""

import argparse
from pathlib import Path

try:
    import arviz as az
except Exception:
    az = None
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build mra mutation-supply table with bio_rate CI.")
    p.add_argument("--input-data", type=Path, required=True, help="Original mutation_supply_data.csv")
    p.add_argument(
        "--supply-csv",
        type=Path,
        default=None,
        help="mutation_supply_site_mutation_table.csv from mra production run",
    )
    p.add_argument(
        "--idata-nc",
        type=Path,
        default=None,
        help="InferenceData netcdf (derive bio_rate CI if supply csv missing CIs)",
    )
    p.add_argument("--out-csv", type=Path, required=True, help="Output merged CSV")
    return p.parse_args()


def _prepare_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "mut_type" not in out.columns and "Mut_Type" in out.columns:
        out["mut_type"] = out["Mut_Type"]
    if "count" not in out.columns and "WT_mutation_count" in out.columns:
        out["count"] = out["WT_mutation_count"]
    if "coverage" not in out.columns and "WT_coverage" in out.columns:
        out["coverage"] = out["WT_coverage"]
    req = ["ref_position", "mut_type", "count", "coverage"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"Input data missing required columns: {missing}")
    out["count"] = pd.to_numeric(out["count"], errors="coerce")
    out["coverage"] = pd.to_numeric(out["coverage"], errors="coerce")
    out = out[out["coverage"] > 0].copy()
    out["observed_rate"] = out["count"] / out["coverage"]
    return out


def _prepare_supply(df: pd.DataFrame) -> pd.DataFrame:
    p = df.copy()
    rename = {
        "bio_rate_mean": "pred_rate_mean",
        "bio_rate_median": "pred_rate_median",
        "bio_rate_lo95": "pred_rate_lo95",
        "bio_rate_hi95": "pred_rate_hi95",
    }
    for old, new in rename.items():
        if old in p.columns and new not in p.columns:
            p[new] = p[old]
    keep = ["ref_position", "mut_type", "pred_rate_mean", "pred_rate_median", "pred_rate_lo95", "pred_rate_hi95"]
    for opt in (
        "context_3p", "likelihood_family", "control_corrected",
        "correction_scheme", "uses_nucleotide_context",
    ):
        if opt in p.columns:
            keep.append(opt)
    for c in keep:
        if c not in p.columns:
            raise ValueError(f"Supply table missing required column: {c}")
    return p.loc[:, keep].copy()


def _rate_summary_from_idata(idata_path: Path) -> pd.DataFrame:
    if az is None:
        raise ImportError("arviz is required to derive posterior summaries from idata.nc.")
    idata = az.from_netcdf(idata_path)
    if "bio_rate" not in idata.posterior:
        raise ValueError("idata.nc does not contain posterior variable bio_rate.")
    mu = idata.posterior["bio_rate"]
    mean = mu.mean(dim=("chain", "draw")).values
    median = mu.median(dim=("chain", "draw")).values
    hdi = az.hdi(mu, hdi_prob=0.95)
    lo = hdi["bio_rate"].sel(hdi="lower").values
    hi = hdi["bio_rate"].sel(hdi="higher").values
    return pd.DataFrame({
        "pred_rate_mean": mean,
        "pred_rate_median": median,
        "pred_rate_lo95": lo,
        "pred_rate_hi95": hi,
    })


def main() -> None:
    args = parse_args()
    in_df = _prepare_input(pd.read_csv(args.input_data))
    in_df = in_df.sort_values(["ref_position", "mut_type"]).reset_index(drop=True)

    if args.supply_csv is not None and args.supply_csv.exists():
        pred = _prepare_supply(pd.read_csv(args.supply_csv))
        pred = pred.sort_values(["ref_position", "mut_type"]).reset_index(drop=True)
        merged = in_df.merge(pred, on=["ref_position", "mut_type"], how="left")
    else:
        merged = in_df.copy()
        for c in ("pred_rate_mean", "pred_rate_median", "pred_rate_lo95", "pred_rate_hi95"):
            merged[c] = np.nan

    need_fill = merged["pred_rate_mean"].isna().all() or merged["pred_rate_lo95"].isna().all()
    if need_fill and args.idata_nc is not None and args.idata_nc.exists():
        rate_df = _rate_summary_from_idata(args.idata_nc)
        if len(rate_df) != len(merged):
            raise ValueError(
                f"idata rows ({len(rate_df)}) do not match table rows ({len(merged)}). "
                "Use the same preprocessing as the mra production script."
            )
        merged.loc[:, ["pred_rate_mean", "pred_rate_median", "pred_rate_lo95", "pred_rate_hi95"]] = rate_df.values

    merged["pred_count_mean"] = merged["pred_rate_mean"] * merged["coverage"]
    merged["pred_count_lo95"] = merged["pred_rate_lo95"] * merged["coverage"]
    merged["pred_count_hi95"] = merged["pred_rate_hi95"] * merged["coverage"]

    out_cols = [
        "ref_position", "mut_type", "Mut_Type", "ref_nucleotide", "three_prime",
        "context_3p", "uses_nucleotide_context", "likelihood_family",
        "control_corrected", "correction_scheme",
        "count", "coverage", "observed_rate",
        "pred_rate_mean", "pred_rate_median", "pred_rate_lo95", "pred_rate_hi95",
        "pred_count_mean", "pred_count_lo95", "pred_count_hi95",
        "WT_mutation_count", "Control_mutation_count", "WT_coverage", "Control_coverage",
    ]
    out_cols = [c for c in out_cols if c in merged.columns]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.loc[:, out_cols].to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
