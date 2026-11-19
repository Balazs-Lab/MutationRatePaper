#!/usr/bin/env python3
"""
Sliding-window comparison of control-adjusted observed WT rate vs modeled bio rate.

Port of the logic in ERRSeq2/scripts/R_analysis/Figure5.R (Figure 5D section).

Input: FINAL_mutation_supply_dataset.csv (or mutation_supply_site_mutation_table.csv)
  with ref_position, wt_count, wt_coverage, control_count, control_coverage,
  bio_rate_median, error_rate_median (or bio_rate_mean / error_rate_mean).

Outputs per window size/step:
  - sliding_window_observed_vs_bio_rate_{W}bp_step{S}.csv
  - sliding_window_correlation_stats_{W}bp_step{S}.csv
  - figures/sliding_window_observed_vs_bio_rate_linear_{W}bp_step{S}.png
  - figures/sliding_window_observed_vs_bio_rate_log10_{W}bp_step{S}.png
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def resolve_paths(model_version: str) -> tuple[Path | None, Path | None]:
    outdir = os.environ.get("MUT_SUPPLY_OUTDIR")
    if not outdir:
        return None, None
    base = Path(outdir)
    for name in ("FINAL_mutation_supply_dataset.csv", "mutation_supply_site_mutation_table.csv"):
        candidate = base / name
        if candidate.exists():
            return candidate, base / "sliding_window_analysis"
    return None, base / "sliding_window_analysis"


def load_site_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    bio_col = "bio_rate_median" if "bio_rate_median" in df.columns else "bio_rate_mean"
    err_col = "error_rate_median" if "error_rate_median" in df.columns else "error_rate_mean"
    required = [
        "ref_position", "wt_count", "wt_coverage",
        "control_count", "control_coverage", bio_col, err_col,
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.rename(columns={bio_col: "bio_rate", err_col: "error_rate"}).copy()
    for col in ["ref_position", "wt_count", "wt_coverage", "control_count", "control_coverage"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["bio_rate"] = pd.to_numeric(out["bio_rate"], errors="coerce")
    out["error_rate"] = pd.to_numeric(out["error_rate"], errors="coerce")
    out["net_bio_rate"] = out["bio_rate"] - out["error_rate"]
    out = out.loc[
        out["ref_position"].notna()
        & out["wt_coverage"].gt(0)
        & out["control_coverage"].gt(0)
        & out["bio_rate"].notna()
        & out["error_rate"].notna()
    ].copy()
    out["ref_position"] = out["ref_position"].astype(int)
    return out


def make_sliding_windows(df: pd.DataFrame, window_size: int, step_size: int) -> pd.DataFrame:
    min_pos = int(df["ref_position"].min())
    max_pos = int(df["ref_position"].max())
    rows: list[dict] = []

    for start_pos in range(min_pos, max_pos - window_size + 2, step_size):
        end_pos = start_pos + window_size - 1
        center_pos = start_pos + window_size // 2
        w = df.loc[(df["ref_position"] >= start_pos) & (df["ref_position"] <= end_pos)]
        if w.empty:
            continue

        wt_count_sum = float(w["wt_count"].sum())
        wt_cov_sum = float(w["wt_coverage"].sum())
        ctrl_count_sum = float(w["control_count"].sum())
        ctrl_cov_sum = float(w["control_coverage"].sum())

        observed_rate = (wt_count_sum / wt_cov_sum) - (ctrl_count_sum / ctrl_cov_sum)
        modeled_rate_mean = float(w["net_bio_rate"].mean())
        weights = w["wt_coverage"].to_numpy(dtype=float)
        values = w["net_bio_rate"].to_numpy(dtype=float)
        modeled_rate_weighted = float(np.average(values, weights=weights))

        rows.append(
            {
                "window_start": start_pos,
                "window_end": end_pos,
                "window_center": center_pos,
                "n_positions": int(w["ref_position"].nunique()),
                "n_rows": int(len(w)),
                "wt_count_sum": wt_count_sum,
                "wt_coverage_sum": wt_cov_sum,
                "control_count_sum": ctrl_count_sum,
                "control_coverage_sum": ctrl_cov_sum,
                "observed_rate": observed_rate,
                "modeled_rate_mean": modeled_rate_mean,
                "modeled_rate_weighted": modeled_rate_weighted,
                # Legacy column names matching Figure5.R exports
                "bio_rate_median_mean": modeled_rate_mean,
                "bio_rate_median_weighted": modeled_rate_weighted,
            }
        )
    return pd.DataFrame(rows)


def correlation_stats(sliding_df: pd.DataFrame) -> pd.DataFrame:
    valid = sliding_df.dropna(subset=["observed_rate", "modeled_rate_mean"])
    if len(valid) < 3:
        return pd.DataFrame()
    r_mean = valid["observed_rate"].corr(valid["modeled_rate_mean"])
    r_weighted = valid["observed_rate"].corr(valid["modeled_rate_weighted"])
    return pd.DataFrame(
        [
            {
                "level": "sliding_window",
                "comparison": "observed_vs_modeled_rate_mean",
                "pearson_r": r_mean,
                "n_windows": len(valid),
            },
            {
                "level": "sliding_window",
                "comparison": "observed_vs_modeled_rate_weighted",
                "pearson_r": r_weighted,
                "n_windows": len(valid),
            },
        ]
    )


def plot_sliding_window(sliding_df: pd.DataFrame, out_path: Path, window_size: int, step_size: int, log_scale: bool) -> None:
    plot_df = sliding_df.melt(
        id_vars=["window_center"],
        value_vars=["observed_rate", "modeled_rate_mean", "modeled_rate_weighted"],
        var_name="rate_type",
        value_name="rate",
    )
    label_map = {
        "observed_rate": "Observed (WT − control)",
        "modeled_rate_mean": "Modeled net bio (mean)",
        "modeled_rate_weighted": "Modeled net bio (coverage-weighted)",
    }
    plot_df["rate_type"] = plot_df["rate_type"].map(label_map)

    fig, ax = plt.subplots(figsize=(10, 5))
    for label, sub in plot_df.groupby("rate_type"):
        ax.plot(sub["window_center"], sub["rate"], linewidth=0.9, alpha=0.9, label=label)
    if log_scale:
        ax.set_yscale("log")
        ylab = "Mutation rate (log10 scale)"
    else:
        ylab = "Mutation rate"
    ax.set_xlabel("Genome position (window center)")
    ax.set_ylabel(ylab)
    ax.set_title(
        f"Sliding-window observed vs modeled rate\n"
        f"window={window_size} bp, step={step_size} bp"
    )
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_one_config(
    df: pd.DataFrame,
    out_dir: Path,
    window_size: int,
    step_size: int,
    model_version: str,
    input_path: Path,
) -> Path:
    sliding_df = make_sliding_windows(df, window_size=window_size, step_size=step_size)
    tag = f"{window_size}bp_step{step_size}"
    csv_path = out_dir / f"sliding_window_observed_vs_bio_rate_{tag}.csv"
    sliding_df.to_csv(csv_path, index=False)

    corr = correlation_stats(sliding_df)
    if not corr.empty:
        corr.to_csv(out_dir / f"sliding_window_correlation_stats_{tag}.csv", index=False)

    fig_dir = out_dir / "figures"
    plot_sliding_window(
        sliding_df, fig_dir / f"sliding_window_observed_vs_bio_rate_linear_{tag}.png",
        window_size, step_size, log_scale=False,
    )
    plot_sliding_window(
        sliding_df, fig_dir / f"sliding_window_observed_vs_bio_rate_log10_{tag}.png",
        window_size, step_size, log_scale=True,
    )
    return csv_path


def run_grid(
    df: pd.DataFrame,
    out_dir: Path,
    window_sizes: list[int],
    step_sizes: list[int],
) -> Path:
    """Compute Pearson R for every window_size x step_size combination.

    Writes a single summary CSV with one row per combination.
    """
    rows: list[dict] = []
    for window_size in window_sizes:
        for step_size in step_sizes:
            sliding_df = make_sliding_windows(df, window_size=window_size, step_size=step_size)
            valid = sliding_df.dropna(subset=["observed_rate", "modeled_rate_mean"])
            n_windows = int(len(valid))
            if n_windows < 3:
                r_mean = np.nan
                r_weighted = np.nan
            else:
                r_mean = valid["observed_rate"].corr(valid["modeled_rate_mean"])
                r_weighted = valid["observed_rate"].corr(valid["modeled_rate_weighted"])
            rows.append(
                {
                    "window_size": window_size,
                    "step_size": step_size,
                    "pearson_r_mean": r_mean,
                    "pearson_r_weighted": r_weighted,
                    "n_windows": n_windows,
                }
            )
            print(
                f"  window={window_size:>4} step={step_size:>4}  "
                f"r_mean={r_mean:.4f}  r_weighted={r_weighted:.4f}  n={n_windows}"
                if n_windows >= 3
                else f"  window={window_size:>4} step={step_size:>4}  insufficient windows (n={n_windows})"
            )
    summary = pd.DataFrame(rows)
    summary_path = out_dir / "sliding_window_correlation_grid.csv"
    summary.to_csv(summary_path, index=False)
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sliding-window observed vs modeled mutation rate.")
    parser.add_argument("--input", type=Path, default=None, help="FINAL_mutation_supply_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--model-version", type=str, default=os.environ.get("MUT_SUPPLY_MODEL_VERSION", "mra"))
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--step-size", type=int, default=10)
    parser.add_argument(
        "--extra-configs",
        type=str,
        default="150:10,250:25",
        help="Additional window:step pairs comma-separated, e.g. 150:10,250:25",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="Sweep all window x step combinations and write a summary Pearson R grid CSV.",
    )
    parser.add_argument(
        "--grid-windows",
        type=str,
        default="10,50,100,150,250,500",
        help="Comma-separated window sizes for --grid mode.",
    )
    parser.add_argument(
        "--grid-steps",
        type=str,
        default="1,5,10,25,50,100",
        help="Comma-separated step sizes for --grid mode.",
    )
    args = parser.parse_args()

    default_input, default_out = resolve_paths(args.model_version)
    input_path = args.input or default_input
    out_dir = args.output_dir or default_out
    if input_path is None or out_dir is None:
        parser.error("--input and --output-dir required unless MUT_SUPPLY_OUTDIR is set")

    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_site_table(input_path)

    if args.grid:
        window_sizes = [int(x) for x in args.grid_windows.split(",") if x.strip()]
        step_sizes = [int(x) for x in args.grid_steps.split(",") if x.strip()]
        print(
            f"Running sliding-window grid: {len(window_sizes)} window sizes x "
            f"{len(step_sizes)} step sizes = {len(window_sizes) * len(step_sizes)} combinations"
        )
        summary_path = run_grid(df, out_dir, window_sizes, step_sizes)
        print(f"Sliding-window grid complete. Summary: {summary_path}")
        return

    configs = [(args.window_size, args.step_size)]
    if args.extra_configs.strip():
        for token in args.extra_configs.split(","):
            token = token.strip()
            if not token:
                continue
            w, s = token.split(":")
            configs.append((int(w), int(s)))

    written: list[str] = []
    for window_size, step_size in configs:
        path = run_one_config(df, out_dir, window_size, step_size, args.model_version, input_path)
        written.append(path.name)

    meta = {
        "model_version": args.model_version,
        "input": str(input_path),
        "output_dir": str(out_dir),
        "n_site_rows": int(len(df)),
        "n_positions": int(df["ref_position"].nunique()),
        "window_configs": [{"window_size": w, "step_size": s} for w, s in configs],
        "output_csv_files": written,
        "observed_rate_definition": "sum(wt_count)/sum(wt_coverage) - sum(control_count)/sum(control_coverage)",
        "modeled_rate_definition": "bio_rate_median - error_rate_median (net bio supply)",
    }
    (out_dir / "sliding_window_analysis_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Sliding-window analysis complete. Outputs: {out_dir}")


if __name__ == "__main__":
    main()
