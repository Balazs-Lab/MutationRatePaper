from __future__ import annotations

"""
mra production Bayesian mutation-supply model (the paper's primary model).

Context and splines (6 total bio, each sum-to-zero):
  - G->A: AA spline, GA spline, shared spline for all other 2-mers (incl. sparse pool).
  - Non-G->A substitutions: one shared position spline (`theta_bio_shared`).
  - insertion / deletion: dedicated position spline each.
  G->A intercepts remain per 2-mer (sparse CC/CG/TC/TT pooled to OTHER).
  Each P-spline samples n_basis-1 free coefficients; the last is -sum(free) so the
  intercept owns the level and the amplitude owns only shape.

WT-only correction for C->A, G->T, deletion. Released quantity: bio_rate.

Input: `mutation_supply_data.csv` produced by mra_build_model_input.py from the
pipeline outputs in data/pipeline (per-REC mutation filtering is applied there).

Priors and run settings are required command-line paths:

    python3 mra_MutationSupplyPymc_production_joint_WT_control.py \\
        --priors data/models/mra/mra_priors.csv \\
        --run data/models/mra/mra_run_production.csv
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import pandas as pd
import patsy
import pymc as pm
import pytensor.tensor as pt

from mra_priors import FilePriors
from mra_run import RunConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit the mra production mutation-supply model."
    )
    parser.add_argument(
        "--priors",
        type=Path,
        required=True,
        help="Path to mra_priors.csv",
    )
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to mra_run_test.csv or mra_run_production.csv",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to mutation_supply_data.csv (default: data/models/mra/mutation_supply_data.csv)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help="Output directory (default: data/models/mra/production)",
    )
    return parser.parse_args()


# =============================================================================
# CONFIG
# =============================================================================

ARGS = parse_args()
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (ARGS.data or Path(os.environ.get(
    "MUT_SUPPLY_DATA_PATH",
    PROJECT_ROOT / "data" / "models" / "mra" / "mutation_supply_data.csv",
))).expanduser().resolve()
OUTDIR = (ARGS.outdir or Path(os.environ.get(
    "MUT_SUPPLY_OUTDIR",
    PROJECT_ROOT / "data" / "models" / "mra" / "production",
))).expanduser().resolve()
OUTDIR.mkdir(parents=True, exist_ok=True)
PRIORS_PATH = ARGS.priors.expanduser().resolve()
RUN_PATH = ARGS.run.expanduser().resolve()
priors = FilePriors(PRIORS_PATH)
run = RunConfig(RUN_PATH)

RANDOM_SEED = run.get("random_seed")
DROP_INDELS = run.get("drop_indels")

CONTEXT_N = run.get("context_n")
CONTEXT_MODE = run.get("context_mode")
MIN_CONTEXT_COUNT = run.get("min_context_count")

ALL_GA_2MER_CONTEXTS = tuple(f"{a}{b}" for a in "ACGT" for b in "ACGT")
BASE_CONTEXT_LEVELS = ("A", "C", "G", "T")

CHAINS = run.get("chains")
CORES = run.get("cores")
DRAWS = run.get("draws")
TUNE = run.get("tune")
TARGET_ACCEPT = run.get("target_accept")

SPLINE_DF = run.get("spline_df")
INCLUDE_POSITION_BIO = run.get("include_position_bio")
INCLUDE_POSITION_ERR = run.get("include_position_err")

GA_MUT_TYPE = "G->A"
GA_AA_2MER = "AA"
GA_GA_2MER = "GA"
GA_SPLINE_SHARED_LABEL = "SHARED"
INSERTION_MUT_TYPE = "insertion"
DELETION_MUT_TYPE = "deletion"
WT_ONLY_MUT_TYPES = tuple(run.get("wt_only_mut_types"))
INDEL_MUT_TYPES = (INSERTION_MUT_TYPE, DELETION_MUT_TYPE)
INVALID_CONTEXT = frozenset({".", "NA", "INVALID"})

MIN_SUB_CONTEXT_WT_COUNT = run.get("min_sub_context_wt_count")
PRIOR_SIGMA_SPLINE_FLOOR = run.get("spline_sigma_floor")
SPLINE_SMOOTHNESS_SCALE = run.get("spline_smoothness_scale")
SPLINE_SMOOTHNESS_SCALE_GA_AA = run.get("spline_smoothness_scale_ga_aa")
SPLINE_SMOOTHNESS_SCALE_GA_NON_AA = run.get("spline_smoothness_scale_ga_non_aa")
SPLINE_SUM_TO_ZERO = bool(run.get("spline_sum_to_zero"))
GA_POOL_2MERS = tuple(token.upper() for token in run.get("ga_pool_2mers"))
for token in GA_POOL_2MERS:
    if len(token) != 2 or not all(c in "ACGT" for c in token):
        raise ValueError(
            f"Invalid G->A pool 2-mer {token!r} in run file ga_pool_2mers"

        )
DEFAULT_GA_POOL_2MERS = GA_POOL_2MERS

# AA and GA dedicated; all other 2-mers (incl. sparse pool) share one landscape.
GA_SPLINE_CONTEXT_LEVELS = (GA_AA_2MER, GA_GA_2MER, GA_SPLINE_SHARED_LABEL)
N_BIO_POSITION_SPLINES = 6


def ga_spline_context_for_raw_2mer(raw_2mer: str) -> str:
    raw_2mer = raw_2mer.strip().upper()
    if raw_2mer == GA_AA_2MER:
        return GA_AA_2MER
    if raw_2mer == GA_GA_2MER:
        return GA_GA_2MER
    if raw_2mer not in ALL_GA_2MER_CONTEXTS:
        raise ValueError(f"Invalid G->A raw 2-mer for spline mapping: {raw_2mer!r}")
    return GA_SPLINE_SHARED_LABEL


def build_ga_spline_smoothness_arrays() -> np.ndarray:
    """Per G->A spline context: roughness-penalty weight (not a sampled prior)."""
    smoothness = np.full(len(GA_SPLINE_CONTEXT_LEVELS), SPLINE_SMOOTHNESS_SCALE_GA_NON_AA)
    aa_idx = GA_SPLINE_CONTEXT_LEVELS.index(GA_AA_2MER)
    smoothness[aa_idx] = SPLINE_SMOOTHNESS_SCALE_GA_AA
    return smoothness


def add_spline_smoothness_potential_per_context(
    name_prefix: str,
    second_diff: Any,
    sigma: Any,
    context_labels: list[str],
    smoothness_scales: np.ndarray,
) -> None:
    for i, (label, scale) in enumerate(zip(context_labels, smoothness_scales)):
        if scale <= 0:
            continue
        add_spline_smoothness_potential(
            f"{name_prefix}_{label}",
            second_diff[i],
            sigma[i],
            smoothness_scale=float(scale),
        )


def sum_to_zero_complete(theta_free: Any, axis: int = -1) -> Any:
    """Append last coefficient = -sum(free) so coefficients sum to zero along `axis`."""
    last = -pt.sum(theta_free, axis=axis, keepdims=True)
    return pt.concatenate([theta_free, last], axis=axis)


# =============================================================================
# HELPERS
# =============================================================================


def save_json(obj: dict[str, Any], path: Path) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def require_columns(df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def posterior_summary_vector(idata: az.InferenceData, var_name: str, hdi_prob: float = 0.95) -> pd.DataFrame:
    arr = idata.posterior[var_name]
    hdi = az.hdi(arr, hdi_prob=hdi_prob)
    if var_name in getattr(hdi, "data_vars", {}):
        hdi = hdi[var_name]
    return pd.DataFrame({
        f"{var_name}_mean": arr.mean(dim=("chain", "draw")).values,
        f"{var_name}_median": arr.median(dim=("chain", "draw")).values,
        f"{var_name}_lo95": hdi.sel(hdi="lower").to_numpy(),
        f"{var_name}_hi95": hdi.sel(hdi="higher").to_numpy(),
    })


def clean_context(x: pd.Series, n: int) -> pd.Series:
    ctx = x.astype(str).str.upper().str.replace(r"[^ACGT]", "N", regex=True).str[:n]
    return ctx.where(ctx.str.len() == n, "NA")


def clean_base(x: pd.Series) -> pd.Series:
    base = x.astype(str).str.upper().str.replace(r"[^ACGT]", "N", regex=True).str[:1]
    return base.where(base.isin(list("ACGT")), "NA")


def build_r11_context(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    mra context scheme (default mixed_ga2_else1, no context for indels):
      - G->A: 3' 2-mer -> ga_3p2
      - other substitutions: first 3' base -> sub_3p1
      - indels: none -> indel_no_context
    """
    three = df["three_prime"].astype(str).str.upper()
    mut = df["mut_type"].astype(str)

    context = pd.Series(index=df.index, dtype=object)
    scheme = pd.Series(index=df.index, dtype=object)

    ga_mask = mut.eq(GA_MUT_TYPE)
    indel_mask = mut.isin(INDEL_MUT_TYPES)
    sub_mask = ~ga_mask & ~indel_mask

    if CONTEXT_MODE == "uniform":
        ctx = clean_context(three, CONTEXT_N)
        context.loc[sub_mask | ga_mask] = ctx.loc[sub_mask | ga_mask].values
        scheme.loc[sub_mask | ga_mask] = f"context_{CONTEXT_N}p_uniform"
    elif CONTEXT_MODE == "mixed_ga2_else1":
        context.loc[sub_mask] = clean_base(three.loc[sub_mask]).values
        scheme.loc[sub_mask] = "sub_3p1"
        context.loc[ga_mask] = clean_context(three.loc[ga_mask], 2).values
        scheme.loc[ga_mask] = "ga_3p2"
    elif CONTEXT_MODE == "mixed_ga3_else2":
        context.loc[sub_mask] = clean_context(three.loc[sub_mask], 2).values
        scheme.loc[sub_mask] = "sub_3p2"
        context.loc[ga_mask] = clean_context(three.loc[ga_mask], 3).values
        scheme.loc[ga_mask] = "ga_3p3"
    else:
        raise ValueError(
            f"Unsupported PYMC_CONTEXT_MODE={CONTEXT_MODE!r}. "
            "Allowed: 'uniform', 'mixed_ga2_else1', 'mixed_ga3_else2'."
        )

    context.loc[indel_mask] = "."
    scheme.loc[indel_mask] = "indel_no_context"
    return context, scheme


def pool_sparse_contexts(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Collapse low-count context strings to OTHER (R1-style)."""
    if MIN_CONTEXT_COUNT <= 0:
        return df, []
    out = df.copy()
    ctx_counts = out.loc[out["context_3p"] != ".", "context_3p"].value_counts()
    sparse = ctx_counts[ctx_counts < MIN_CONTEXT_COUNT].index.tolist()
    sparse = [c for c in sparse if c not in INVALID_CONTEXT and c != "OTHER"]
    if sparse:
        out.loc[out["context_3p"].isin(sparse), "context_3p"] = "OTHER"
    return out, sparse


def pool_ga_sparse_2mers(
    df: pd.DataFrame, pool_levels: tuple[str, ...]
) -> tuple[pd.DataFrame, list[str]]:
    """Pool selected G->A 2-mer contexts into OTHER (sparse-context stabilization)."""
    if not pool_levels:
        return df, []
    out = df.copy()
    ga_mask = out["mut_type"].eq(GA_MUT_TYPE)
    if not ga_mask.any():
        return out, []
    present = [
        ctx for ctx in pool_levels
        if (ga_mask & out["context_3p"].eq(ctx)).any()
    ]
    if not present:
        return out, []
    out.loc[ga_mask & out["context_3p"].isin(present), "context_3p"] = "OTHER"
    return out, present


def validate_ga_rows(df: pd.DataFrame, ga_scheme: str) -> None:
    ga = df.loc[df["mut_type"].eq(GA_MUT_TYPE)]
    if ga.empty:
        return
    bad_ref = ~ga["ref_nucleotide"].astype(str).str.upper().str[:1].eq("G")
    if bad_ref.any():
        n = int(bad_ref.sum())
        examples = ga.loc[bad_ref, ["ref_position", "ref_nucleotide", "three_prime"]].head()
        raise ValueError(
            f"{n} G->A rows have ref_nucleotide != G (expected by definition).\n{examples}"
        )
    expected_len = 3 if ga_scheme == "ga_3p3" else 2
    bad_ctx = ga["context_3p"].astype(str).str.len() != expected_len
    bad_ctx |= ~ga["context_3p"].astype(str).str.match(r"^[ACGT]+$")
    bad_ctx &= ~ga["context_3p"].isin(["OTHER"])
    if bad_ctx.any():
        n = int(bad_ctx.sum())
        examples = ga.loc[bad_ctx, ["ref_position", "three_prime", "context_3p"]].head()
        raise ValueError(
            f"{n} G->A rows have invalid {expected_len}-mer context.\n{examples}"
        )


def validate_substitution_rows(df: pd.DataFrame, sub_scheme: str) -> None:
    subs = df.loc[~df["mut_type"].isin(INDEL_MUT_TYPES) & ~df["mut_type"].eq(GA_MUT_TYPE)]
    if subs.empty:
        return
    if sub_scheme == "sub_3p1":
        bad = ~subs["context_3p"].isin(list(BASE_CONTEXT_LEVELS) + ["OTHER"])
    else:
        bad = subs["context_3p"].astype(str).str.len() != 2
        bad |= ~subs["context_3p"].astype(str).str.match(r"^[ACGT]+$")
        bad &= ~subs["context_3p"].isin(["OTHER"])
    if bad.any():
        n = int(bad.sum())
        examples = subs.loc[bad, ["ref_position", "mut_type", "three_prime", "context_3p"]].head()
        raise ValueError(
            f"{n} non-G->A substitution rows have invalid context for {sub_scheme}.\n{examples}"
        )


def correction_scheme(mut_type: str) -> str:
    if mut_type in WT_ONLY_MUT_TYPES:
        return "wt_only_no_control"
    return "joint_control_wt"


def likelihood_label(mut_type: str) -> str:
    if mut_type == GA_MUT_TYPE:
        if INCLUDE_POSITION_BIO:
            return "negative_binomial_ga_3p2_aa_ga_shared_splines_joint_control_wt"
        return "negative_binomial_ga_3p2_intercept_joint_control_wt"
    if mut_type in WT_ONLY_MUT_TYPES:
        if mut_type == DELETION_MUT_TYPE and INCLUDE_POSITION_BIO:
            return "negative_binomial_deletion_spline_wt_only"
        tag = "_shared_spline" if INCLUDE_POSITION_BIO and mut_type != DELETION_MUT_TYPE else ""
        if mut_type in ("C->A", "G->T"):
            return f"negative_binomial_sub_3p1_type_context{tag}_wt_only"
        return f"negative_binomial_3p1{tag}_wt_only"
    if mut_type == INSERTION_MUT_TYPE:
        return "negative_binomial_insertion_spline_joint_control_wt" if INCLUDE_POSITION_BIO else "negative_binomial_insertion_joint_control_wt"
    if mut_type == DELETION_MUT_TYPE:
        return "negative_binomial_deletion_spline_wt_only"
    return (
        "negative_binomial_sub_3p1_type_context_shared_spline_joint_control_wt"
        if INCLUDE_POSITION_BIO
        else "negative_binomial_sub_3p1_type_context_joint_control_wt"
    )


def sub_type_bio_rate_centers(df: pd.DataFrame, sub_types: list[str]) -> np.ndarray:
    centers = []
    for mut_type in sub_types:
        sub = df.loc[df["mut_type"] == mut_type]
        if sub.empty:
            centers.append(np.log(1e-12))
            continue
        wt_rate = float(sub["wt_count"].sum() / sub["wt_coverage"].sum())
        if mut_type in WT_ONLY_MUT_TYPES:
            bio_rate = float(max(wt_rate, 1e-12))
        else:
            ctrl_rate = float(sub["control_count"].sum() / sub["control_coverage"].sum())
            bio_rate = float(max(wt_rate - ctrl_rate, 1e-12))
        centers.append(np.log(bio_rate))
    return np.asarray(centers, dtype=np.float64)


def sub_context_wt_counts(
    df: pd.DataFrame, sub_types: list[str], context_levels: list[str]
) -> np.ndarray:
    counts = np.zeros((len(sub_types), len(context_levels)), dtype=np.int64)
    for i, mut_type in enumerate(sub_types):
        for j, ctx in enumerate(context_levels):
            sub = df.loc[(df["mut_type"] == mut_type) & df["context_3p"].eq(ctx)]
            counts[i, j] = int(sub["wt_count"].sum()) if not sub.empty else 0
    return counts


def sub_context_bio_rate_centers(
    df: pd.DataFrame, sub_types: list[str], context_levels: list[str]
) -> np.ndarray:
    centers = np.zeros((len(sub_types), len(context_levels)), dtype=np.float64)
    for i, mut_type in enumerate(sub_types):
        for j, ctx in enumerate(context_levels):
            sub = df.loc[(df["mut_type"] == mut_type) & df["context_3p"].eq(ctx)]
            if sub.empty:
                centers[i, j] = np.log(1e-12)
                continue
            wt_rate = float(sub["wt_count"].sum() / sub["wt_coverage"].sum())
            if mut_type in WT_ONLY_MUT_TYPES:
                bio_rate = float(max(wt_rate, 1e-12))
            else:
                ctrl_rate = float(sub["control_count"].sum() / sub["control_coverage"].sum())
                bio_rate = float(max(wt_rate - ctrl_rate, 1e-12))
            centers[i, j] = np.log(bio_rate)
    return centers


def ga_bio_rate_centers(df: pd.DataFrame, context_levels: list[str]) -> np.ndarray:
    centers = []
    for ctx in context_levels:
        sub = df.loc[(df["mut_type"] == GA_MUT_TYPE) & df["context_3p"].eq(ctx)]
        if sub.empty:
            centers.append(np.log(1e-12))
            continue
        wt_rate = float(sub["wt_count"].sum() / sub["wt_coverage"].sum())
        ctrl_rate = float(sub["control_count"].sum() / sub["control_coverage"].sum())
        bio_rate = float(max(wt_rate - ctrl_rate, 1e-12))
        centers.append(np.log(bio_rate))
    return np.asarray(centers, dtype=np.float64)


def add_spline_smoothness_potential(
    name: str,
    second_diff: Any,
    sigma: Any,
    smoothness_scale: float | None = None,
) -> None:
    scale = SPLINE_SMOOTHNESS_SCALE if smoothness_scale is None else smoothness_scale
    if scale <= 0:
        return
    if getattr(second_diff, "ndim", 0) == 2:
        sigma_eff = pt.maximum(sigma[:, None], 1e-6)
    else:
        sigma_eff = pt.maximum(sigma, 1e-6)
    pm.Potential(
        name,
        scale * pm.logp(pm.Normal.dist(0.0, sigma_eff), second_diff).sum(),
    )


def context_levels_for_mutations(
    df: pd.DataFrame, mut_types: tuple[str, ...] | list[str]
) -> list[str]:
    levels = sorted(
        df.loc[df["mut_type"].isin(mut_types), "context_3p"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return [c for c in levels if c not in INVALID_CONTEXT]


# =============================================================================
# LOAD + PREPROCESS
# =============================================================================

print(f"Reading data: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

rename_map = {
    "Mut_Type": "mut_type",
    "WT_mutation_count": "wt_count",
    "WT_coverage": "wt_coverage",
    "Control_mutation_count": "control_count",
    "Control_coverage": "control_coverage",
    "ref_position": "ref_position",
    "ref_nucleotide": "ref_nucleotide",
    "three_prime": "three_prime",
}
df = df.rename(columns=rename_map)

require_columns(df, [
    "ref_position", "mut_type", "wt_count", "wt_coverage",
    "control_count", "control_coverage", "three_prime", "ref_nucleotide",
])

for c in ["ref_position", "wt_count", "wt_coverage", "control_count", "control_coverage"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.loc[
    df["ref_position"].notna()
    & df["mut_type"].notna()
    & df["wt_count"].notna()
    & df["wt_coverage"].notna()
    & df["control_count"].notna()
    & df["control_coverage"].notna()
    & (df["wt_coverage"] > 0)
    & (df["control_coverage"] > 0)
].copy()

df["mut_type"] = df["mut_type"].astype(str)
if DROP_INDELS:
    df = df.loc[~df["mut_type"].str.lower().isin(["insertion", "deletion", "ins", "del"])].copy()

df["three_prime"] = df["three_prime"].astype(str).str.upper()
df["ref_nucleotide"] = df["ref_nucleotide"].astype(str).str.upper()
df["context_3p"], df["context_scheme"] = build_r11_context(df)
df["ga_2mer_raw"] = pd.NA
ga_rows_pre_pool = df["mut_type"].eq(GA_MUT_TYPE)
df.loc[ga_rows_pre_pool, "ga_2mer_raw"] = df.loc[ga_rows_pre_pool, "context_3p"].values
df, pooled_sparse_contexts = pool_sparse_contexts(df)
df, pooled_ga_2mer_contexts = pool_ga_sparse_2mers(df, GA_POOL_2MERS)

ga_scheme = "ga_3p2"
sub_scheme = "sub_3p1"
if CONTEXT_MODE == "mixed_ga3_else2":
    ga_scheme = "ga_3p3"
    sub_scheme = "sub_3p2"
elif CONTEXT_MODE == "uniform":
    ga_scheme = f"context_{CONTEXT_N}p_uniform"
    sub_scheme = f"context_{CONTEXT_N}p_uniform"

validate_ga_rows(df, ga_scheme if ga_scheme.startswith("ga_") else "ga_3p2")
validate_substitution_rows(df, sub_scheme if sub_scheme.startswith("sub_") else "sub_3p1")

df["uses_nucleotide_context"] = ~df["mut_type"].isin(INDEL_MUT_TYPES)
df["control_corrected"] = ~df["mut_type"].isin(WT_ONLY_MUT_TYPES)
df["correction_scheme"] = df["mut_type"].map(correction_scheme)
df["likelihood_family"] = df["mut_type"].map(likelihood_label)

df["log_wt_exp"] = np.log(df["wt_coverage"])
df["log_control_exp"] = np.log(df["control_coverage"])
df = df.sort_values(["ref_position", "mut_type", "context_3p"]).reset_index(drop=True)

print(f"Preprocessed rows: {len(df):,}")
print(f"Context mode: {CONTEXT_MODE}")
if pooled_sparse_contexts:
    print(f"Pooled sparse contexts to OTHER (wt<{MIN_CONTEXT_COUNT}): {len(pooled_sparse_contexts)} levels")
if pooled_ga_2mer_contexts:
    print(f"Pooled G->A 2-mer contexts to OTHER: {pooled_ga_2mer_contexts}")
elif GA_POOL_2MERS:
    print(f"G->A 2-mer pool list configured but none present in data: {list(GA_POOL_2MERS)}")
print(df[["ref_position", "mut_type", "context_3p", "context_scheme", "control_corrected"]].head())
print("\nCorrection scheme counts:\n", df["correction_scheme"].value_counts())
print("\nContext scheme counts:\n", df["context_scheme"].value_counts())

# =============================================================================
# ENCODING
# =============================================================================

mut_type_cat = pd.Categorical(df["mut_type"])
df["mut_type_code"] = mut_type_cat.codes
mut_type_idx = df["mut_type_code"].to_numpy(dtype=np.int64)
mut_type_levels = list(mut_type_cat.categories)

shared_sub_type_levels = [
    m for m in mut_type_levels if m not in INDEL_MUT_TYPES and m != GA_MUT_TYPE
]
shared_sub_type_to_idx = {m: i for i, m in enumerate(shared_sub_type_levels)}
sub_context_levels = context_levels_for_mutations(df, shared_sub_type_levels)
ga_context_levels = context_levels_for_mutations(df, [GA_MUT_TYPE])
if CONTEXT_MODE == "mixed_ga2_else1":
    missing_ga = [c for c in ALL_GA_2MER_CONTEXTS if c not in ga_context_levels]
    if missing_ga:
        print(f"Note: {len(missing_ga)} G->A 2-mer contexts absent from data (no rows): {missing_ga}")

if not ga_context_levels:
    raise ValueError("No valid G->A context levels found.")
if not sub_context_levels:
    raise ValueError("No valid shared-substitution context levels found.")

sub_type_bio_center = sub_type_bio_rate_centers(df, shared_sub_type_levels)
sub_context_wt = sub_context_wt_counts(df, shared_sub_type_levels, sub_context_levels)
sub_context_bio_center = sub_context_bio_rate_centers(
    df, shared_sub_type_levels, sub_context_levels
)
sub_context_powered = sub_context_wt >= MIN_SUB_CONTEXT_WT_COUNT
sub_context_pooled_cells = [
    (shared_sub_type_levels[i], sub_context_levels[j])
    for i in range(len(shared_sub_type_levels))
    for j in range(len(sub_context_levels))
    if not sub_context_powered[i, j]
]
ga_bio_center = ga_bio_rate_centers(df, ga_context_levels)

n_obs = len(df)
ga_mut_idx = mut_type_levels.index(GA_MUT_TYPE)
insertion_mut_idx = mut_type_levels.index(INSERTION_MUT_TYPE) if INSERTION_MUT_TYPE in mut_type_levels else -1
deletion_mut_idx = mut_type_levels.index(DELETION_MUT_TYPE) if DELETION_MUT_TYPE in mut_type_levels else -1

is_ga = (mut_type_idx == ga_mut_idx).astype(np.float64)
is_insertion = (
    (mut_type_idx == insertion_mut_idx).astype(np.float64)
    if insertion_mut_idx >= 0 else np.zeros(n_obs, dtype=np.float64)
)
is_deletion = (
    (mut_type_idx == deletion_mut_idx).astype(np.float64)
    if deletion_mut_idx >= 0 else np.zeros(n_obs, dtype=np.float64)
)
is_shared_sub = (
    (1.0 - is_ga) * (1.0 - is_insertion) * (1.0 - is_deletion)
).astype(np.float64)

is_control_corrected = df["control_corrected"].to_numpy(dtype=np.float64)
cc_idx = np.where(is_control_corrected > 0.5)[0].astype(np.int64)

ga_context_to_idx = {ctx: i for i, ctx in enumerate(ga_context_levels)}
sub_context_to_idx = {ctx: i for i, ctx in enumerate(sub_context_levels)}

ga_context_idx = np.zeros(n_obs, dtype=np.int64)
sub_context_idx = np.zeros(n_obs, dtype=np.int64)
shared_sub_type_idx = np.zeros(n_obs, dtype=np.int64)
sub_context_powered_obs = np.zeros(n_obs, dtype=np.float64)

for obs_i in range(n_obs):
    ctx_name = str(df["context_3p"].iloc[obs_i])
    if is_ga[obs_i] > 0.5:
        if ctx_name not in ga_context_to_idx:
            raise ValueError(f"G->A obs {obs_i} has unmapped context {ctx_name!r}")
        ga_context_idx[obs_i] = ga_context_to_idx[ctx_name]
    if is_shared_sub[obs_i] > 0.5:
        mut_name = mut_type_levels[mut_type_idx[obs_i]]
        if mut_name not in shared_sub_type_to_idx:
            raise ValueError(f"Shared-sub obs {obs_i} has unmapped type {mut_name!r}")
        type_i = shared_sub_type_to_idx[mut_name]
        shared_sub_type_idx[obs_i] = type_i
        if ctx_name not in sub_context_to_idx:
            raise ValueError(f"Shared-sub obs {obs_i} has unmapped context {ctx_name!r}")
        ctx_i = sub_context_to_idx[ctx_name]
        sub_context_idx[obs_i] = ctx_i
        sub_context_powered_obs[obs_i] = float(sub_context_powered[type_i, ctx_i])

df["sub_context_powered"] = sub_context_powered_obs

ga_other_2mer_levels = list(pooled_ga_2mer_contexts)
ga_spline_context_levels = list(GA_SPLINE_CONTEXT_LEVELS)
ga_spline_context_to_idx = {ctx: i for i, ctx in enumerate(ga_spline_context_levels)}
ga_spline_smoothness = build_ga_spline_smoothness_arrays()
ga_spline_sigma_floor = np.full(len(ga_spline_context_levels), PRIOR_SIGMA_SPLINE_FLOOR)

ga_spline_context_idx = np.zeros(n_obs, dtype=np.int64)
for obs_i in range(n_obs):
    if is_ga[obs_i] < 0.5:
        continue
    raw_ctx = str(df["ga_2mer_raw"].iloc[obs_i])
    spline_ctx = ga_spline_context_for_raw_2mer(raw_ctx)
    ga_spline_context_idx[obs_i] = ga_spline_context_to_idx[spline_ctx]
df["ga_spline_context"] = [
    ga_spline_context_levels[ga_spline_context_idx[i]] if is_ga[i] > 0.5 else pd.NA
    for i in range(n_obs)
]
df["ga_other_2mer"] = df["ga_2mer_raw"].where(df["ga_2mer_raw"].isin(ga_other_2mer_levels))

ga_ctx_wt = df.loc[df["mut_type"].eq(GA_MUT_TYPE)].groupby("context_3p")["wt_count"].sum()

print("Mutation types:", mut_type_levels)
print("Shared-substitution types:", shared_sub_type_levels)
print(f"G->A 2-mer context levels ({len(ga_context_levels)}):", ga_context_levels)
if INCLUDE_POSITION_BIO:
    print(
        "G->A position splines: AA + GA dedicated; all other 2-mers share one spline "
        f"({GA_SPLINE_SHARED_LABEL})"
    )
    print(
        "  spline amplitude priors from mra_priors.csv "
        f"(floor={PRIOR_SIGMA_SPLINE_FLOOR}); "
        f"sum_to_zero={SPLINE_SUM_TO_ZERO}; "
        f"AA smoothness={SPLINE_SMOOTHNESS_SCALE_GA_AA}, "
        f"GA/{GA_SPLINE_SHARED_LABEL} smoothness={SPLINE_SMOOTHNESS_SCALE_GA_NON_AA}"
    )
if pooled_ga_2mer_contexts:
    print(f"G->A sparse 2-mers pooled to OTHER (intercept only): {pooled_ga_2mer_contexts}")
elif GA_POOL_2MERS:
    print(f"G->A 2-mer pool list configured but none present in data: {list(GA_POOL_2MERS)}")
elif not INCLUDE_POSITION_BIO:
    print("G->A position splines disabled (PYMC_INCLUDE_POSITION_BIO=0)")
print(f"Shared-sub 1-base context levels ({len(sub_context_levels)}):", sub_context_levels)
print(
    f"Sub context powered cells (wt>={MIN_SUB_CONTEXT_WT_COUNT}): "
    f"{int(sub_context_powered.sum())}/{sub_context_powered.size}"
)
if sub_context_pooled_cells:
    print(f"Pooled sparse type×context cells: {len(sub_context_pooled_cells)}")
print(f"Control-corrected obs: {len(cc_idx):,}; WT-only obs: {n_obs - len(cc_idx):,}")

# =============================================================================
# POSITION BASIS
# =============================================================================

pos_mean = float(df["ref_position"].mean())
pos_std = float(df["ref_position"].std()) or 1.0
df["ref_position_scaled"] = (df["ref_position"] - pos_mean) / pos_std

spline_df = patsy.dmatrix(
    f"bs(ref_position_scaled, df={SPLINE_DF}, degree=3, include_intercept=False)",
    data=df,
    return_type="dataframe",
)
B = np.asarray(spline_df, dtype=np.float64)
n_basis = B.shape[1]
n_basis_free = n_basis - 1 if SPLINE_SUM_TO_ZERO else n_basis
if SPLINE_SUM_TO_ZERO and n_basis < 2:
    raise ValueError("sum-to-zero P-splines require n_basis >= 2")
print(
    f"Position spline basis: df={SPLINE_DF}, n_basis={n_basis}, "
    f"n_basis_free={n_basis_free}, sum_to_zero={SPLINE_SUM_TO_ZERO}"
)

wt_count = df["wt_count"].to_numpy(dtype=np.int64)
control_count = df["control_count"].to_numpy(dtype=np.int64)
log_wt_exp = df["log_wt_exp"].to_numpy(dtype=np.float64)
log_control_exp = df["log_control_exp"].to_numpy(dtype=np.float64)

cc_mask = df["control_corrected"]
control_base_rate = float(np.clip(
    df.loc[cc_mask, "control_count"].sum() / df.loc[cc_mask, "control_coverage"].sum(),
    1e-12, None,
))
wt_base_rate = float(np.clip(df["wt_count"].sum() / df["wt_coverage"].sum(), 1e-12, None))
bio_excess_rate = float(max(wt_base_rate - control_base_rate, 1e-12))

coords = {
    "obs": np.arange(n_obs),
    "obs_cc": np.arange(len(cc_idx)),
    "mut_type": mut_type_levels,
    "shared_sub_mut_type": shared_sub_type_levels,
    "sub_context_1base": sub_context_levels,
    "ga_context": ga_context_levels,
    "ga_spline_context": ga_spline_context_levels,
    "basis": np.arange(n_basis),
    "basis_free": np.arange(n_basis_free),
}

print(f"Using priors from {PRIORS_PATH}")
print(f"Using run settings from {RUN_PATH}")
shutil.copy2(PRIORS_PATH, OUTDIR / PRIORS_PATH.name)
shutil.copy2(RUN_PATH, OUTDIR / RUN_PATH.name)

# =============================================================================
# MODEL
# =============================================================================

with pm.Model(coords=coords) as model:
    mut_type_data = pm.Data("mut_type_idx", mut_type_idx, dims="obs")
    sub_context_data = pm.Data("sub_context_idx", sub_context_idx, dims="obs")
    shared_sub_type_data = pm.Data("shared_sub_type_idx", shared_sub_type_idx, dims="obs")
    sub_context_powered_data = pm.Data("sub_context_powered", sub_context_powered_obs, dims="obs")
    ga_context_data = pm.Data("ga_context_idx", ga_context_idx, dims="obs")
    ga_spline_context_data = pm.Data("ga_spline_context_idx", ga_spline_context_idx, dims="obs")
    is_ga_data = pm.Data("is_ga", is_ga, dims="obs")
    is_shared_sub_data = pm.Data("is_shared_sub", is_shared_sub, dims="obs")
    is_insertion_data = pm.Data("is_insertion", is_insertion, dims="obs")
    is_deletion_data = pm.Data("is_deletion", is_deletion, dims="obs")
    is_cc_data = pm.Data("is_control_corrected", is_control_corrected, dims="obs")
    log_wt_exp_data = pm.Data("log_wt_exp", log_wt_exp, dims="obs")
    log_control_exp_data = pm.Data("log_control_exp", log_control_exp, dims="obs")
    cc_idx_data = pm.Data("cc_idx", cc_idx)
    B_data = pm.Data("B", B, dims=("obs", "basis"))

    intercept_err_mu, intercept_err_sd = priors.normal("intercept_err")
    intercept_err = pm.Normal("intercept_err", mu=intercept_err_mu, sigma=intercept_err_sd)
    sigma_mut_err = pm.HalfNormal("sigma_mut_err", sigma=priors.halfnormal("sigma_mut_err"))
    z_mut_err_mu, z_mut_err_sd = priors.normal_vector("z_mut_err", mut_type_levels)
    z_mut_err = pm.Normal("z_mut_err", mu=z_mut_err_mu, sigma=z_mut_err_sd, dims="mut_type")
    beta_mut_err = pm.Deterministic("beta_mut_err", z_mut_err * sigma_mut_err, dims="mut_type")

    intercept_bio_mu, intercept_bio_sd = priors.normal("intercept_bio")
    intercept_bio = pm.Normal("intercept_bio", mu=intercept_bio_mu, sigma=intercept_bio_sd)
    sigma_mut_bio = pm.HalfNormal("sigma_mut_bio", sigma=priors.halfnormal("sigma_mut_bio"))
    z_mut_bio_mu, z_mut_bio_sd = priors.normal_vector("z_mut_bio", mut_type_levels)
    z_mut_bio = pm.Normal("z_mut_bio", mu=z_mut_bio_mu, sigma=z_mut_bio_sd, dims="mut_type")
    beta_mut_bio = pm.Deterministic("beta_mut_bio", z_mut_bio * sigma_mut_bio, dims="mut_type")

    sub_mu, sub_sigma = priors.normal_vector("beta_sub_type_bio", shared_sub_type_levels)
    beta_sub_type_bio = pm.Normal(
        "beta_sub_type_bio",
        mu=sub_mu,
        sigma=sub_sigma,
        dims="shared_sub_mut_type",
    )
    sigma_sub_context_delta = pm.HalfNormal(
        "sigma_sub_context_delta", sigma=priors.halfnormal("sigma_sub_context_delta")
    )
    z_ctx_mu, z_ctx_sd = priors.normal_matrix(
        "z_sub_context_delta", shared_sub_type_levels, sub_context_levels
    )
    z_sub_context_delta = pm.Normal(
        "z_sub_context_delta",
        mu=z_ctx_mu,
        sigma=z_ctx_sd,
        dims=("shared_sub_mut_type", "sub_context_1base"),
    )
    beta_sub_context_delta = pm.Deterministic(
        "beta_sub_context_delta",
        z_sub_context_delta * sigma_sub_context_delta,
        dims=("shared_sub_mut_type", "sub_context_1base"),
    )

    ga_mu, ga_sigma = priors.normal_vector("intercept_bio_ga", ga_context_levels)
    intercept_bio_ga = pm.Normal(
        "intercept_bio_ga",
        mu=ga_mu,
        sigma=ga_sigma,
        dims="ga_context",
    )

    if INCLUDE_POSITION_BIO:
        basis_free_levels = list(range(n_basis_free))

        sigma_spline_bio_shared_raw = pm.HalfNormal(
            "sigma_spline_bio_shared_raw",
            sigma=priors.halfnormal("sigma_spline_bio_shared_raw"),
        )
        sigma_spline_bio_shared = pm.Deterministic(
            "sigma_spline_bio_shared",
            PRIOR_SIGMA_SPLINE_FLOOR + sigma_spline_bio_shared_raw,
        )
        theta_shared_mu, theta_shared_sd = priors.normal_vector(
            "theta_bio_shared_raw", basis_free_levels
        )
        theta_bio_shared_raw = pm.Normal(
            "theta_bio_shared_raw",
            mu=theta_shared_mu,
            sigma=theta_shared_sd,
            dims="basis_free",
        )
        theta_bio_shared_full = (
            sum_to_zero_complete(theta_bio_shared_raw)
            if SPLINE_SUM_TO_ZERO
            else theta_bio_shared_raw
        )
        theta_bio_shared = pm.Deterministic(
            "theta_bio_shared",
            theta_bio_shared_full * sigma_spline_bio_shared,
            dims="basis",
        )
        if n_basis >= 3:
            second_diff_shared = (
                theta_bio_shared[:-2] - 2.0 * theta_bio_shared[1:-1] + theta_bio_shared[2:]
            )
            add_spline_smoothness_potential(
                "spline_smoothness_bio_shared", second_diff_shared, sigma_spline_bio_shared
            )

        sigma_spline_bio_insertion_raw = pm.HalfNormal(
            "sigma_spline_bio_insertion_raw",
            sigma=priors.halfnormal("sigma_spline_bio_insertion_raw"),
        )
        sigma_spline_bio_insertion = pm.Deterministic(
            "sigma_spline_bio_insertion",
            PRIOR_SIGMA_SPLINE_FLOOR + sigma_spline_bio_insertion_raw,
        )
        theta_ins_mu, theta_ins_sd = priors.normal_vector(
            "theta_bio_insertion_raw", basis_free_levels
        )
        theta_bio_insertion_raw = pm.Normal(
            "theta_bio_insertion_raw",
            mu=theta_ins_mu,
            sigma=theta_ins_sd,
            dims="basis_free",
        )
        theta_bio_insertion_full = (
            sum_to_zero_complete(theta_bio_insertion_raw)
            if SPLINE_SUM_TO_ZERO
            else theta_bio_insertion_raw
        )
        theta_bio_insertion = pm.Deterministic(
            "theta_bio_insertion",
            theta_bio_insertion_full * sigma_spline_bio_insertion,
            dims="basis",
        )
        if n_basis >= 3:
            second_diff_ins = (
                theta_bio_insertion[:-2] - 2.0 * theta_bio_insertion[1:-1] + theta_bio_insertion[2:]
            )
            add_spline_smoothness_potential(
                "spline_smoothness_bio_insertion", second_diff_ins, sigma_spline_bio_insertion
            )

        sigma_spline_bio_deletion_raw = pm.HalfNormal(
            "sigma_spline_bio_deletion_raw",
            sigma=priors.halfnormal("sigma_spline_bio_deletion_raw"),
        )
        sigma_spline_bio_deletion = pm.Deterministic(
            "sigma_spline_bio_deletion",
            PRIOR_SIGMA_SPLINE_FLOOR + sigma_spline_bio_deletion_raw,
        )
        theta_del_mu, theta_del_sd = priors.normal_vector(
            "theta_bio_deletion_raw", basis_free_levels
        )
        theta_bio_deletion_raw = pm.Normal(
            "theta_bio_deletion_raw",
            mu=theta_del_mu,
            sigma=theta_del_sd,
            dims="basis_free",
        )
        theta_bio_deletion_full = (
            sum_to_zero_complete(theta_bio_deletion_raw)
            if SPLINE_SUM_TO_ZERO
            else theta_bio_deletion_raw
        )
        theta_bio_deletion = pm.Deterministic(
            "theta_bio_deletion",
            theta_bio_deletion_full * sigma_spline_bio_deletion,
            dims="basis",
        )
        if n_basis >= 3:
            second_diff_del = (
                theta_bio_deletion[:-2] - 2.0 * theta_bio_deletion[1:-1] + theta_bio_deletion[2:]
            )
            add_spline_smoothness_potential(
                "spline_smoothness_bio_deletion", second_diff_del, sigma_spline_bio_deletion
            )

        ga_sigma_scales = priors.halfnormal_vector(
            "sigma_spline_bio_ga_raw", ga_spline_context_levels
        )
        sigma_spline_bio_ga_raw = pm.HalfNormal(
            "sigma_spline_bio_ga_raw",
            sigma=ga_sigma_scales,
            dims="ga_spline_context",
        )
        sigma_spline_bio_ga = pm.Deterministic(
            "sigma_spline_bio_ga",
            ga_spline_sigma_floor + sigma_spline_bio_ga_raw,
            dims="ga_spline_context",
        )
        theta_ga_mu, theta_ga_sd = priors.normal_matrix(
            "theta_bio_ga_raw",
            ga_spline_context_levels,
            basis_free_levels,
        )
        theta_bio_ga_raw = pm.Normal(
            "theta_bio_ga_raw",
            mu=theta_ga_mu,
            sigma=theta_ga_sd,
            dims=("ga_spline_context", "basis_free"),
        )
        theta_bio_ga_full = (
            sum_to_zero_complete(theta_bio_ga_raw, axis=-1)
            if SPLINE_SUM_TO_ZERO
            else theta_bio_ga_raw
        )
        theta_bio_ga = pm.Deterministic(
            "theta_bio_ga",
            theta_bio_ga_full * sigma_spline_bio_ga[:, None],
            dims=("ga_spline_context", "basis"),
        )
        if n_basis >= 3:
            second_diff_bio_ga = (
                theta_bio_ga[:, :-2]
                - 2.0 * theta_bio_ga[:, 1:-1]
                + theta_bio_ga[:, 2:]
            )
            add_spline_smoothness_potential_per_context(
                "spline_smoothness_bio_ga",
                second_diff_bio_ga,
                sigma_spline_bio_ga,
                ga_spline_context_levels,
                ga_spline_smoothness,
            )

    alpha_control = pm.Exponential(
        "alpha_control",
        lam=priors.exponential_vector("alpha_control", mut_type_levels),
        dims="mut_type",
    )
    alpha_wt = pm.Exponential(
        "alpha_wt",
        lam=priors.exponential_vector("alpha_wt", mut_type_levels),
        dims="mut_type",
    )

    eta_err = intercept_err + beta_mut_err[mut_type_data]

    eta_bio_shared_sub = (
        beta_sub_type_bio[shared_sub_type_data]
        + sub_context_powered_data
        * beta_sub_context_delta[shared_sub_type_data, sub_context_data]
    )
    eta_bio_indel = intercept_bio + beta_mut_bio[mut_type_data]
    eta_bio_ga = intercept_bio_ga[ga_context_data]
    if INCLUDE_POSITION_BIO:
        spline_ga = pt.sum(B_data * theta_bio_ga[ga_spline_context_data], axis=1)
        eta_bio_ga = eta_bio_ga + spline_ga

    eta_bio = (
        is_shared_sub_data * eta_bio_shared_sub
        + is_insertion_data * eta_bio_indel
        + is_deletion_data * eta_bio_indel
        + is_ga_data * eta_bio_ga
    )

    if INCLUDE_POSITION_BIO:
        spline_shared = pt.sum(B_data * theta_bio_shared, axis=1)
        spline_insertion = pt.sum(B_data * theta_bio_insertion, axis=1)
        spline_deletion = pt.sum(B_data * theta_bio_deletion, axis=1)
        eta_bio = (
            eta_bio
            + is_shared_sub_data * spline_shared
            + is_insertion_data * spline_insertion
            + is_deletion_data * spline_deletion
        )

    error_rate = pm.Deterministic("error_rate", pt.exp(eta_err) * is_cc_data, dims="obs")
    bio_rate = pm.Deterministic("bio_rate", pt.exp(eta_bio), dims="obs")
    wt_total_rate = pm.Deterministic("wt_total_rate", error_rate + bio_rate, dims="obs")

    mu_control_count = pm.Deterministic(
        "mu_control_count", pt.exp(log_control_exp_data) * error_rate, dims="obs"
    )
    mu_wt_count = pm.Deterministic(
        "mu_wt_count", pt.exp(log_wt_exp_data) * wt_total_rate, dims="obs"
    )

    mut_type_cc = mut_type_idx[cc_idx]
    pm.NegativeBinomial(
        "y_control",
        mu=mu_control_count[cc_idx_data],
        alpha=alpha_control[mut_type_cc],
        observed=control_count[cc_idx],
        dims="obs_cc",
    )
    pm.NegativeBinomial(
        "y_wt",
        mu=mu_wt_count,
        alpha=alpha_wt[mut_type_data],
        observed=wt_count,
        dims="obs",
    )

    print(
        f"Sampling draws={DRAWS}, tune={TUNE}, chains={CHAINS}, "
        f"cores={CORES}, target_accept={TARGET_ACCEPT}, "
        f"spline_df={SPLINE_DF}, context_mode={CONTEXT_MODE}"
    )
    idata = pm.sample(
        draws=DRAWS,
        tune=TUNE,
        chains=CHAINS,
        cores=CORES,
        target_accept=TARGET_ACCEPT,
        random_seed=RANDOM_SEED,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
    )

    ppc = pm.sample_posterior_predictive(
        idata,
        var_names=[
            "y_control", "y_wt",
            "error_rate", "bio_rate", "wt_total_rate", "mu_control_count", "mu_wt_count",
        ],
        return_inferencedata=True,
        random_seed=RANDOM_SEED,
    )

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

summary_vars = [
    "intercept_err", "intercept_bio", "intercept_bio_ga",
    "sigma_mut_err", "sigma_mut_bio",
    "alpha_control", "alpha_wt",
    "beta_mut_err", "beta_mut_bio", "beta_sub_type_bio", "sigma_sub_context_delta",
    "beta_sub_context_delta",
]
if INCLUDE_POSITION_BIO:
    summary_vars.extend([
        "sigma_spline_bio_ga",
        "sigma_spline_bio_shared",
        "sigma_spline_bio_insertion",
        "sigma_spline_bio_deletion",
    ])

summary_df = az.summary(idata, var_names=summary_vars, round_to=4)
summary_df.to_csv(OUTDIR / "posterior_summary.csv")
az.summary(idata, kind="diagnostics", var_names=summary_vars, round_to=4).to_csv(
    OUTDIR / "posterior_diagnostics.csv"
)

idata.to_netcdf(OUTDIR / "idata.nc")
ppc.to_netcdf(OUTDIR / "ppc.nc")
df.to_csv(OUTDIR / "model_input_preprocessed.csv", index=False)

context_column_name = {
    "mixed_ga2_else1": "context_mixed_ga2_else1",
    "mixed_ga3_else2": "context_mixed_ga3_else2",
}.get(CONTEXT_MODE, f"context_{CONTEXT_N}p_{CONTEXT_MODE}")

metadata = {
    "model": "mra_joint_control_wt_nb_ga2_sub1_ga3_sum0_bio_splines",
    "data_path": str(DATA_PATH),
    "wt_only_mut_types": list(WT_ONLY_MUT_TYPES),
    "wt_only_rationale": "C->A/G->T: control artifact inflation; deletion: sparse/uninformative control",
    "indel_mut_types": list(INDEL_MUT_TYPES),
    "context_mut_types": [m for m in mut_type_levels if m not in INDEL_MUT_TYPES],
    "no_context_mut_types": list(INDEL_MUT_TYPES),
    "shared_sub_mut_types": list(shared_sub_type_levels),
    "likelihood_by_mut_type": {m: likelihood_label(m) for m in mut_type_levels},
    "context_n": CONTEXT_N,
    "context_mode": CONTEXT_MODE,
    "context_column": context_column_name,
    "min_context_count": MIN_CONTEXT_COUNT,
    "pooled_sparse_contexts": pooled_sparse_contexts,
    "ga_pool_2mer_default": list(DEFAULT_GA_POOL_2MERS),
    "ga_pool_2mer_configured": list(GA_POOL_2MERS),
    "pooled_ga_2mer_contexts": pooled_ga_2mer_contexts,
    "ga_other_2mer_levels": ga_other_2mer_levels,
    "ga_spline_shared_label": GA_SPLINE_SHARED_LABEL,
    "ga_ga_2mer": GA_GA_2MER,
    "ga_spline_context_levels": ga_spline_context_levels,
    "ga_spline_sigma_prior_scale": {
        ctx: float(priors.halfnormal(f"sigma_spline_bio_ga_raw[{ctx}]"))
        for ctx in ga_spline_context_levels
    },
    "ga_spline_sigma_floor": dict(zip(ga_spline_context_levels, ga_spline_sigma_floor.tolist())),
    "ga_spline_smoothness_scale": dict(zip(ga_spline_context_levels, ga_spline_smoothness.tolist())),
    "ga_aa_2mer": GA_AA_2MER,
    "n_bio_position_splines": N_BIO_POSITION_SPLINES,
    "spline_sum_to_zero": SPLINE_SUM_TO_ZERO,
    "n_spline_basis_free": n_basis_free,
    "context_rules": {
        "substitutions_excl_ga": (
            "beta_sub_type_bio[type] + masked beta_sub_context_delta[type, 1base]; "
            f"sparse cells pooled to type-only when wt_count < {MIN_SUB_CONTEXT_WT_COUNT}"
        ),
        GA_MUT_TYPE: (
            "intercept per 2-mer (sparse CC/CG/TC/TT pooled to OTHER); "
            "AA and GA each get a position spline; all other 2-mers share one spline"
        ),
        "indels": "no nucleotide context; intercept_bio + type effect + dedicated spline",
    },
    "spline_rules": {
        "G->A_AA": f"dedicated position spline ({GA_AA_2MER} 2-mer)",
        "G->A_GA": f"dedicated position spline ({GA_GA_2MER} 2-mer)",
        "G->A_shared": (
            f"shared position spline for all other G->A 2-mers (label {GA_SPLINE_SHARED_LABEL})"
        ),
        "shared_substitutions": "one position spline for all non-G->A substitutions",
        INSERTION_MUT_TYPE: "dedicated insertion spline",
        DELETION_MUT_TYPE: "dedicated deletion spline",
        "sum_to_zero": (
            "each bio P-spline samples n_basis-1 free coefficients; "
            "last coefficient = -sum(free) so the intercept owns the level"
        ),
        "total_bio_position_splines": N_BIO_POSITION_SPLINES,
    },
    "all_ga_2mer_contexts": list(ALL_GA_2MER_CONTEXTS),
    "n_ga_contexts_present": len(ga_context_levels),
    "ga_modeling": "intercept_per_2mer_plus_3_ga_splines_sum0",
    "ga_context_definition": "three_prime_2mer_at_G_sites",
    "ga_context_levels": ga_context_levels,
    "ga_context_total_wt_counts": {ctx: int(ga_ctx_wt.get(ctx, 0)) for ctx in ga_context_levels},
    "ga_bio_rate_centers_log": dict(zip(ga_context_levels, ga_bio_center.tolist())),
    "sub_context_1base_levels": sub_context_levels,
    "sub_type_bio_centers_log": dict(zip(shared_sub_type_levels, sub_type_bio_center.tolist())),
    "sub_context_total_wt_counts": {
        shared_sub_type_levels[i]: {
            sub_context_levels[j]: int(sub_context_wt[i, j])
            for j in range(len(sub_context_levels))
        }
        for i in range(len(shared_sub_type_levels))
    },
    "sub_context_powered": {
        shared_sub_type_levels[i]: {
            sub_context_levels[j]: bool(sub_context_powered[i, j])
            for j in range(len(sub_context_levels))
        }
        for i in range(len(shared_sub_type_levels))
    },
    "sub_context_pooled_cells": [
        {"mut_type": mt, "context_3p": ctx} for mt, ctx in sub_context_pooled_cells
    ],
    "min_sub_context_wt_count": MIN_SUB_CONTEXT_WT_COUNT,
    "include_position_splines": INCLUDE_POSITION_BIO,
    "include_position_bio": INCLUDE_POSITION_BIO,
    "include_position_err": INCLUDE_POSITION_ERR,
    "spline_df": SPLINE_DF,
    "n_spline_basis": n_basis,
    "position_mean": pos_mean,
    "position_std": pos_std,
    "mut_type_levels": mut_type_levels,
    "drop_indels": DROP_INDELS,
    "random_seed": RANDOM_SEED,
    "draws": DRAWS,
    "tune": TUNE,
    "chains": CHAINS,
    "target_accept": TARGET_ACCEPT,
    "prior_file": str(PRIORS_PATH),
    "prior_unique": priors.unique_prior_summary(),
    "run_file": str(RUN_PATH),
    "run_settings": run.as_dict(),
    "use_transfer_priors": False,
    "prior_sigma_spline_floor": PRIOR_SIGMA_SPLINE_FLOOR,
    "spline_smoothness_scale": SPLINE_SMOOTHNESS_SCALE,
    "spline_smoothness_scale_ga_aa": SPLINE_SMOOTHNESS_SCALE_GA_AA,
    "spline_smoothness_scale_ga_non_aa": SPLINE_SMOOTHNESS_SCALE_GA_NON_AA,
    "n_obs": n_obs,
    "n_obs_control_corrected": int(len(cc_idx)),
    "n_obs_wt_only": int(n_obs - len(cc_idx)),
    "control_base_rate": control_base_rate,
    "wt_base_rate": wt_base_rate,
    "bio_excess_rate": bio_excess_rate,
    "primary_release_variable": "bio_rate",
}
save_json(metadata, OUTDIR / "model_metadata.json")

rate_summaries = [
    posterior_summary_vector(idata, "error_rate"),
    posterior_summary_vector(idata, "bio_rate"),
    posterior_summary_vector(idata, "wt_total_rate"),
    posterior_summary_vector(idata, "mu_control_count"),
    posterior_summary_vector(idata, "mu_wt_count"),
]
supply = pd.concat([df.reset_index(drop=True)] + [x.reset_index(drop=True) for x in rate_summaries], axis=1)

wt_only_mask = supply["mut_type"].isin(WT_ONLY_MUT_TYPES)
for col in ["error_rate_mean", "error_rate_median", "error_rate_lo95", "error_rate_hi95"]:
    supply.loc[wt_only_mask, col] = 0.0

supply["bio_fraction_of_wt_total_mean"] = (
    supply["bio_rate_mean"] / supply["wt_total_rate_mean"].replace(0, np.nan)
)
supply["error_fraction_of_wt_total_mean"] = (
    supply["error_rate_mean"] / supply["wt_total_rate_mean"].replace(0, np.nan)
)
supply.loc[wt_only_mask, "bio_fraction_of_wt_total_mean"] = 1.0
supply.loc[wt_only_mask, "error_fraction_of_wt_total_mean"] = 0.0

supply.to_csv(OUTDIR / "mutation_supply_site_mutation_table.csv", index=False)

mut_summary = supply.groupby("mut_type", as_index=False).agg(
    n_sites=("ref_position", "size"),
    control_corrected=("control_corrected", "first"),
    uses_nucleotide_context=("uses_nucleotide_context", "first"),
    correction_scheme=("correction_scheme", "first"),
    likelihood_family=("likelihood_family", "first"),
    context_scheme=("context_scheme", "first"),
    total_wt_count=("wt_count", "sum"),
    total_control_count=("control_count", "sum"),
    mean_bio_rate=("bio_rate_mean", "mean"),
    median_bio_rate=("bio_rate_median", "median"),
    mean_error_rate=("error_rate_mean", "mean"),
    mean_wt_total_rate=("wt_total_rate_mean", "mean"),
)
mut_summary.to_csv(OUTDIR / "mutation_type_summary.csv", index=False)

ctx_supply = supply.loc[supply["uses_nucleotide_context"]].copy()
if len(ctx_supply) > 0:
    context_summary = ctx_supply.groupby(
        ["mut_type", "context_3p", "context_scheme", "control_corrected"], as_index=False
    ).agg(
        n_sites=("ref_position", "size"),
        total_wt_count=("wt_count", "sum"),
        total_control_count=("control_count", "sum"),
        mean_bio_rate=("bio_rate_mean", "mean"),
        median_bio_rate=("bio_rate_median", "median"),
        mean_error_rate=("error_rate_mean", "mean"),
    )
    context_summary.to_csv(OUTDIR / "mutation_type_context_summary.csv", index=False)

for metric_name, func in [("loo", az.loo), ("waic", az.waic)]:
    try:
        res = func(idata, var_name="y_wt", pointwise=True)
        with open(OUTDIR / f"{metric_name}_wt.txt", "w") as f:
            f.write(str(res))
    except Exception as e:
        print(f"Could not compute {metric_name.upper()} (y_wt):", repr(e))

print("\nSaved outputs to:", OUTDIR.resolve())
print("\nPosterior summary head:")
print(summary_df.head())
