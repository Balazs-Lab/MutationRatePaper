#!/usr/bin/env python3
"""Audit the prior specification of an mra production fit.

Reconstructs every sampled parameter's prior exactly as
`mra_MutationSupplyPymc_production_joint_WT_control.py` builds it, then compares each
prior against the corresponding marginal posterior to quantify how much of the posterior
is coming from the data rather than from the prior.

Two diagnostics per parameter (Betancourt's prior-sensitivity pair):

    contraction = 1 - (posterior_sd / prior_sd)^2
        ~1  posterior far tighter than prior -> data-dominated
        ~0  posterior as wide as prior       -> prior-dominated

    z = (posterior_mean - prior_mean) / prior_sd
        |z| small  prior and data agree
        |z| large  posterior moved off the prior centre (data overrode the prior)

The failure mode to look for is low contraction *with* small |z|: the posterior is simply
reproducing the prior. High |z| is reassuring, not alarming -- it means the likelihood was
strong enough to move off the prior centre.

Usage:
    python mra_prior_audit.py [FIT_DIR] [-o OUT_CSV]

FIT_DIR defaults to data/models/mra/production and must contain model_metadata.json and
posterior_summary.csv; transfer_priors.json is read when the fit used transfer priors.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]

HALFNORMAL_MEAN = math.sqrt(2.0 / math.pi)
HALFNORMAL_SD = math.sqrt(1.0 - 2.0 / math.pi)


def halfnormal_moments(scale: float, floor: float = 0.0) -> tuple[float, float]:
    """Mean and sd of floor + HalfNormal(scale)."""
    return floor + scale * HALFNORMAL_MEAN, scale * HALFNORMAL_SD


def normal_moments(mu: float, sd: float) -> tuple[float, float]:
    return mu, sd


class PriorTable:
    """Accumulates one row per sampled scalar parameter."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(
        self,
        param: str,
        block: str,
        family: str,
        prior_mean: float,
        prior_sd: float,
        origin: str,
        note: str = "",
    ) -> None:
        self.rows.append(
            {
                "param": param,
                "block": block,
                "family": family,
                "prior_mean": prior_mean,
                "prior_sd": prior_sd,
                "prior_origin": origin,
                "note": note,
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def build_priors(meta: dict, transfer: dict | None) -> pd.DataFrame:
    """Reproduce the prior for every sampled parameter of the production model."""
    t = PriorTable()

    shrink = float(meta.get("prior_shrinkage") or 1.0)
    theta_sigma = float(meta.get("prior_theta_sigma") or 1.0)
    sigma_intercept = 1.5  # PYMC_PRIOR_SIGMA_INTERCEPT
    sigma_mut = 0.75  # PYMC_PRIOR_SIGMA_MUT
    sigma_sub_context = float(meta["prior_sigma_sub_context"])
    sigma_ga_intercept = float(meta["prior_sigma_ga_intercept"])
    sigma_spline = float(meta["prior_sigma_spline"])
    spline_floor = float(meta["prior_sigma_spline_floor"])

    mut_types = meta["mut_type_levels"]
    shared_sub = meta["shared_sub_mut_types"]
    ga_levels = meta["ga_context_levels"]
    base_levels = meta["sub_context_1base_levels"]
    ga_spline_levels = meta["ga_spline_context_levels"]
    n_basis = int(meta["n_spline_basis"])

    ga_centers = meta["ga_bio_rate_centers_log"]
    sub_centers = meta["sub_type_bio_centers_log"]
    ga_spline_scale = meta["ga_spline_sigma_prior_scale"]
    ga_spline_floor = meta["ga_spline_sigma_floor"]

    has_transfer = bool(meta.get("use_transfer_priors")) and transfer is not None

    # ---- error arm (never transferred) -------------------------------------------------
    t.add(
        "intercept_err",
        "error baseline",
        "Normal",
        *normal_moments(math.log(float(meta["control_base_rate"])), sigma_intercept),
        origin="empirical centre",
        note="centred on log pooled control rate",
    )
    t.add(
        "sigma_mut_err",
        "error type effects",
        "HalfNormal",
        *halfnormal_moments(sigma_mut),
        origin="weakly informative",
    )
    for m in mut_types:
        t.add(
            f"z_mut_err[{m}]",
            "error type effects",
            "Normal",
            *normal_moments(0.0, 1.0),
            origin="unit normal",
            note="non-centred auxiliary",
        )

    # ---- biological baseline -----------------------------------------------------------
    if has_transfer and transfer.get("intercept_bio"):
        mu, sd = transfer["intercept_bio"]
        t.add(
            "intercept_bio",
            "bio baseline",
            "Normal",
            *normal_moments(float(mu), max(float(sd) * shrink, 0.05)),
            origin="transfer",
            note="reference posterior mean/sd",
        )
    else:
        t.add(
            "intercept_bio",
            "bio baseline",
            "Normal",
            *normal_moments(math.log(float(meta["bio_excess_rate"])), sigma_intercept),
            origin="empirical centre",
            note="centred on log WT-minus-control excess",
        )

    t.add(
        "sigma_mut_bio",
        "bio type effects",
        "HalfNormal",
        *halfnormal_moments(sigma_mut),
        origin="weakly informative",
    )
    for m in mut_types:
        t.add(
            f"z_mut_bio[{m}]",
            "bio type effects",
            "Normal",
            *normal_moments(0.0, 1.0),
            origin="unit normal",
            note="non-centred auxiliary",
        )

    # ---- per-type substitution effects -------------------------------------------------
    tmap = (transfer or {}).get("beta_sub_type_bio") or {}
    for m in shared_sub:
        if has_transfer and m in tmap:
            mu, sd = tmap[m]
            t.add(
                f"beta_sub_type_bio[{m}]",
                "bio type effects",
                "Normal",
                *normal_moments(float(mu), max(float(sd) * shrink, 0.05)),
                origin="transfer",
            )
        else:
            t.add(
                f"beta_sub_type_bio[{m}]",
                "bio type effects",
                "Normal",
                *normal_moments(float(sub_centers[m]), sigma_mut),
                origin="empirical centre",
                note="per-type control-corrected empirical log rate",
            )

    # ---- 3' single-base context deviations (never transferred) -------------------------
    t.add(
        "sigma_sub_context_delta",
        "context deviations",
        "HalfNormal",
        *halfnormal_moments(sigma_sub_context),
        origin="weakly informative",
    )
    for m in shared_sub:
        for b in base_levels:
            t.add(
                f"z_sub_context_delta[{m}, {b}]",
                "context deviations",
                "Normal",
                *normal_moments(0.0, 1.0),
                origin="unit normal",
                note="non-centred auxiliary; masked when WT count < 20",
            )

    # ---- G->A per-dinucleotide intercepts ----------------------------------------------
    gmap = (transfer or {}).get("intercept_bio_ga") or {}
    for c in ga_levels:
        if has_transfer and c in gmap:
            mu, sd = gmap[c]
            t.add(
                f"intercept_bio_ga[{c}]",
                "G->A context intercepts",
                "Normal",
                *normal_moments(float(mu), max(float(sd) * shrink, 0.05)),
                origin="transfer",
            )
        else:
            t.add(
                f"intercept_bio_ga[{c}]",
                "G->A context intercepts",
                "Normal",
                *normal_moments(float(ga_centers[c]), sigma_ga_intercept),
                origin="empirical centre",
                note="per-context empirical log rate",
            )

    # ---- spline amplitude scales -------------------------------------------------------
    def spline_scale(key: str, default: float, subtract_floor: bool) -> tuple[float, str]:
        val = (transfer or {}).get(key)
        if has_transfer and val is not None:
            raw = float(val) - spline_floor if subtract_floor else float(val)
            return max(raw, 0.05), "transfer"
        return default, "weakly informative"

    for name, key, subtract in [
        ("sigma_spline_bio_shared_raw", "sigma_spline_shared", True),
        ("sigma_spline_bio_insertion_raw", "sigma_spline_insertion", False),
        ("sigma_spline_bio_deletion_raw", "sigma_spline_deletion", False),
    ]:
        scale, origin = spline_scale(key, sigma_spline, subtract)
        t.add(
            name,
            "spline amplitude",
            "HalfNormal",
            *halfnormal_moments(scale),
            origin=origin,
            note=f"scale={scale:.4g}",
        )

    ga_sig = (transfer or {}).get("sigma_spline_ga") or {}
    for c in ga_spline_levels:
        floor = float(ga_spline_floor[c])
        if has_transfer and c in ga_sig:
            scale = max(float(ga_sig[c]), floor + 1e-6)
            origin = "transfer"
        else:
            scale = float(ga_spline_scale[c])
            origin = "ML-calibrated (mgcv)"
        t.add(
            f"sigma_spline_bio_ga_raw[{c}]",
            "spline amplitude",
            "HalfNormal",
            *halfnormal_moments(scale),
            origin=origin,
            note=f"scale={scale:.4g}, floor={floor:.2g} added downstream",
        )

    # ---- spline shape coefficients -----------------------------------------------------
    def add_theta(name: str, vec, block: str) -> None:
        if has_transfer and vec is not None:
            arr = np.asarray(vec, dtype=float)
            for j in range(len(arr)):
                t.add(
                    f"{name}[{j}]",
                    block,
                    "Normal",
                    *normal_moments(float(arr[j]), theta_sigma),
                    origin="transfer",
                    note="reference profile back-projected onto B-spline basis",
                )
        else:
            for j in range(n_basis):
                t.add(
                    f"{name}[{j}]",
                    block,
                    "Normal",
                    *normal_moments(0.0, 1.0),
                    origin="unit normal",
                )

    add_theta("theta_bio_shared_raw", (transfer or {}).get("theta_shared_raw"), "spline shape")
    add_theta("theta_bio_insertion_raw", (transfer or {}).get("theta_insertion_raw"), "spline shape")
    add_theta("theta_bio_deletion_raw", (transfer or {}).get("theta_deletion_raw"), "spline shape")

    theta_ga = (transfer or {}).get("theta_ga_raw") or {}
    for c in ga_spline_levels:
        if has_transfer and c in theta_ga:
            arr = np.asarray(theta_ga[c], dtype=float)
            for j in range(len(arr)):
                t.add(
                    f"theta_bio_ga_raw[{c}, {j}]",
                    "spline shape",
                    "Normal",
                    *normal_moments(float(arr[j]), theta_sigma),
                    origin="transfer",
                    note="reference profile back-projected onto B-spline basis",
                )
        else:
            for j in range(n_basis):
                t.add(
                    f"theta_bio_ga_raw[{c}, {j}]",
                    "spline shape",
                    "Normal",
                    *normal_moments(0.0, 1.0),
                    origin="unit normal",
                )

    # ---- dispersion (never transferred) ------------------------------------------------
    for arm in ("alpha_control", "alpha_wt"):
        for m in mut_types:
            t.add(
                f"{arm}[{m}]",
                "dispersion",
                "Exponential",
                1.0,
                1.0,
                origin="weakly informative",
                note="Exponential(1)",
            )

    return t.frame()


# Sampled parameters whose priors this script reconstructs. Everything else in the
# posterior group is either a deterministic transform or an observation-sized quantity.
SAMPLED_VARS = (
    "intercept_err",
    "sigma_mut_err",
    "z_mut_err",
    "intercept_bio",
    "sigma_mut_bio",
    "z_mut_bio",
    "beta_sub_type_bio",
    "sigma_sub_context_delta",
    "z_sub_context_delta",
    "intercept_bio_ga",
    "sigma_spline_bio_shared_raw",
    "theta_bio_shared_raw",
    "sigma_spline_bio_insertion_raw",
    "theta_bio_insertion_raw",
    "sigma_spline_bio_deletion_raw",
    "theta_bio_deletion_raw",
    "sigma_spline_bio_ga_raw",
    "theta_bio_ga_raw",
    "alpha_control",
    "alpha_wt",
)


def _label(coord_vals) -> str:
    return ", ".join(str(v) for v in coord_vals)


def posterior_moments(fit_dir: Path) -> pd.DataFrame:
    """Per-parameter posterior mean, sd, ESS and R-hat.

    Prefers the full trace (idata.nc) because the saved posterior_summary.csv omits the
    non-centred auxiliaries and every spline coefficient. Falls back to the CSV summaries.
    """
    trace = fit_dir / "idata.nc"
    if not trace.exists():
        print("[warn] idata.nc not found; falling back to posterior_summary.csv "
              "(spline coefficients and non-centred auxiliaries will be missing)")
        post = pd.read_csv(fit_dir / "posterior_summary.csv")
        post = post.rename(columns={post.columns[0]: "param"})
        return post[["param", "mean", "sd", "ess_bulk", "r_hat"]].rename(
            columns={"mean": "post_mean", "sd": "post_sd"}
        )

    import arviz as az
    import xarray as xr

    ds = xr.open_dataset(trace, group="posterior", engine="h5netcdf")
    present = [v for v in SAMPLED_VARS if v in ds.data_vars]
    sub = ds[present].load()  # small: no observation-sized variables in this list

    summ = az.summary(sub, var_names=present, kind="all", round_to=None)
    summ = summ.reset_index().rename(
        columns={"index": "param", "mean": "post_mean", "sd": "post_sd"}
    )
    ds.close()
    return summ[["param", "post_mean", "post_sd", "ess_bulk", "r_hat"]]


def grade(contraction: float, z: float) -> str:
    """Label a parameter by how much the data, rather than the prior, drives it."""
    if not np.isfinite(contraction) or not np.isfinite(z):
        return "unknown"
    if contraction >= 0.90:
        return "data-dominated"
    if contraction >= 0.50:
        return "data-led"
    if contraction >= 0.10:
        return "prior-influenced"
    if abs(z) >= 1.0:
        return "prior-influenced"  # wide posterior but data still moved it
    return "prior-dominated"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "fit_dir",
        nargs="?",
        default=PROJECT_ROOT / "data" / "models" / "mra" / "production",
        type=Path,
    )
    ap.add_argument("-o", "--out", type=Path, default=None)
    args = ap.parse_args()

    fit_dir = args.fit_dir
    meta = json.loads((fit_dir / "model_metadata.json").read_text())
    transfer_path = fit_dir / "transfer_priors.json"
    transfer = json.loads(transfer_path.read_text()) if transfer_path.exists() else None

    post = posterior_moments(fit_dir)

    priors = build_priors(meta, transfer)
    merged = priors.merge(post, on="param", how="left")

    missing = merged["post_mean"].isna().sum()
    if missing:
        print(f"[warn] {missing} reconstructed priors had no matching posterior row")
        print(merged.loc[merged["post_mean"].isna(), "param"].head(10).to_string(index=False))

    merged["sd_ratio"] = merged["post_sd"] / merged["prior_sd"]
    merged["contraction"] = 1.0 - merged["sd_ratio"] ** 2
    merged["z"] = (merged["post_mean"] - merged["prior_mean"]) / merged["prior_sd"]
    merged["grade"] = [grade(c, z) for c, z in zip(merged["contraction"], merged["z"])]

    out = args.out or (fit_dir / "prior_audit.csv")
    merged.to_csv(out, index=False)

    fit = merged.dropna(subset=["post_mean"])
    print(f"\nPrior audit: {len(fit)} sampled scalar parameters in {fit_dir}")
    print(f"use_transfer_priors = {meta.get('use_transfer_priors')}, "
          f"shrinkage = {meta.get('prior_shrinkage')}, theta_sigma = {meta.get('prior_theta_sigma')}")

    print("\n-- by block --")
    by_block = fit.groupby("block").agg(
        n=("param", "size"),
        median_contraction=("contraction", "median"),
        min_contraction=("contraction", "min"),
        median_abs_z=("z", lambda s: s.abs().median()),
        max_abs_z=("z", lambda s: s.abs().max()),
    )
    print(by_block.round(3).to_string())

    print("\n-- by prior origin --")
    by_origin = fit.groupby("prior_origin").agg(
        n=("param", "size"),
        median_contraction=("contraction", "median"),
        median_abs_z=("z", lambda s: s.abs().median()),
    )
    print(by_origin.round(3).to_string())

    print("\n-- grade distribution --")
    print(fit["grade"].value_counts().to_string())

    flagged = fit.loc[fit["grade"] == "prior-dominated"].sort_values("contraction")
    print(f"\n-- prior-dominated parameters ({len(flagged)}) --")
    if len(flagged):
        cols = ["param", "prior_origin", "prior_mean", "prior_sd", "post_mean", "post_sd", "contraction", "z"]
        print(flagged[cols].head(25).round(3).to_string(index=False))

    print("\n-- least contracted 15 overall --")
    cols = ["param", "prior_origin", "prior_sd", "post_sd", "contraction", "z", "grade"]
    print(fit.nsmallest(15, "contraction")[cols].round(3).to_string(index=False))

    print("\n-- largest |z| 10 (data overrode prior centre) --")
    print(fit.reindex(fit["z"].abs().sort_values(ascending=False).index)[cols].head(10).round(3).to_string(index=False))

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
