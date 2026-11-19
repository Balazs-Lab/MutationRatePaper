#!/usr/bin/env python3
"""
Prefilter corrected UMI CSVs before grouping to drop low-count UMIs.

Input CSV (from belt umi-map / clustering script) is expected to have columns:
  - "Read Name"
  - "UMI1", "UMI2" (corrected UMIs)
  - "UMI1_Raw", "UMI2_Raw" (optional; preserved if present)

We keep only rows whose UMI in the chosen column appears at least `min_count`
times. This reduces the number of UMI groups belt umi-group has to consider.

Usage (example):

  python scripts/pipeline_scirpts/03_filter_umis_for_grouping.py \
    --input data/05-umi_maps/sample_Paired_UMIs_read_names.csv \
    --output data/06-umi_groups/sample_Paired_UMIs_for_grouping.csv \
    --umi-column UMI1 \
    --min-count 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Prefilter corrected UMI CSVs before grouping (drop low-count UMIs)."
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Corrected UMI CSV from umi-map / clustering.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV with only UMIs that pass the count filter.",
    )
    ap.add_argument(
        "--umi-column",
        type=str,
        default="UMI1",
        help='UMI column to count/group by (e.g. "UMI1", "UMI2"). Default: UMI1.',
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=3,
        help="Only keep UMIs with at least this many reads (default 3).",
    )
    ap.add_argument(
        "--log-json",
        type=Path,
        default=None,
        help="Optional JSON log path for stats.",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        return 1

    df = pd.read_csv(args.input)
    if args.umi_column not in df.columns:
        print(f"Error: umi-column '{args.umi_column}' not in columns {list(df.columns)}", file=sys.stderr)
        return 1

    # Restrict to reads with both UMI1 and UMI2 present (paired only)
    # belt umi-map writes one row per read; missing UMI is empty string or NaN
    required = ["UMI1", "UMI2"]
    for c in required:
        if c not in df.columns:
            print(f"Error: required column '{c}' not in CSV (columns: {list(df.columns)})", file=sys.stderr)
            return 1
    n_before = len(df)
    df = df.dropna(subset=required)
    df = df[df["UMI1"].astype(str).str.strip().str.len() > 0]
    df = df[df["UMI2"].astype(str).str.strip().str.len() > 0]
    n_dropped_missing = n_before - len(df)

    # Count UMIs in the chosen column and filter
    counts = df[args.umi_column].astype(str).value_counts()
    keep = df[args.umi_column].astype(str).map(counts) >= args.min_count
    filtered = df[keep].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(args.output, index=False)

    if args.log_json:
        stats = {
            "input": str(args.input),
            "output": str(args.output),
            "umi_column": args.umi_column,
            "min_count": int(args.min_count),
            "total_rows": int(n_before),
            "rows_dropped_missing_umi": int(n_dropped_missing),
            "rows_after_paired_filter": int(len(df)),
            "kept_rows": int(len(filtered)),
            "dropped_rows": int(len(df) - len(filtered)),
            "unique_umis_total": int(counts.size),
            "unique_umis_kept": int(counts[counts >= args.min_count].size),
        }
        args.log_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.log_json, "w") as fh:
            json.dump(stats, fh, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

