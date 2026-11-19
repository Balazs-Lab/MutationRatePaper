from __future__ import annotations

"""
Post-fit analysis for the mra production mutation-supply model.

Figures:
  - ppc_wt_by_mut_type.png, ppc_control_by_mut_type.png
  - mutation_spectrum_bio_vs_error.png
  - context_bio_supply_GtoA.png (intercept levels across 2-mers)
  - position_bio_rate_GtoA_AA.png
  - position_bio_rate_GtoA_GA.png
  - position_bio_rate_GtoA_shared.png
  - position_bio_rate_shared_sub.png
  - position_bio_rate_insertion.png
  - position_bio_rate_deletion.png

Run:
    MUT_SUPPLY_OUTDIR=/path/to/mra/output python mra_MutationSupplyPymc_postfit_analysis.py
"""

import json
import os
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

GA_MUT_TYPE = "G->A"
DEFAULT_WT_ONLY = ("C->A", "G->T", "deletion")

PROJECT_ROOT = Path(__file__).resolve().parents[3]

OUTDIR = Path(os.environ.get(
    "MUT_SUPPLY_OUTDIR",
    PROJECT_ROOT / "data" / "models" / "mra" / "production",
))
FIGDIR = OUTDIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

idata = az.from_netcdf(OUTDIR / "idata.nc")
ppc = az.from_netcdf(OUTDIR / "ppc.nc")
df = pd.read_csv(OUTDIR / "model_input_preprocessed.csv")
supply = pd.read_csv(OUTDIR / "mutation_supply_site_mutation_table.csv")
with open(OUTDIR / "model_metadata.json") as f:
    metadata = json.load(f)

WT_ONLY_MUT_TYPES = tuple(metadata.get("wt_only_mut_types", DEFAULT_WT_ONLY))
NO_CONTEXT_MUT_TYPES = tuple(metadata.get("no_context_mut_types", ()))


def posterior_diag_var_names(idata_obj: az.InferenceData) -> list[str]:
    if "posterior" not in idata_obj.groups():
        return []

    skip_exact = {
        "bio_rate", "error_rate", "wt_total_rate",
        "mu_control_count", "mu_wt_count",
        "mut_type_idx", "sub_context_idx", "shared_sub_type_idx", "sub_context_powered",
        "ga_context_idx", "ga_spline_context_idx", "is_ga",
        "is_shared_sub", "is_insertion", "is_deletion",
        "is_control_corrected", "log_wt_exp", "log_control_exp", "cc_idx", "B",
        "theta_bio_shared", "theta_bio_insertion", "theta_bio_deletion",
        "theta_bio_ga",
        "theta_err",
        "theta_bio_shared_raw", "theta_bio_insertion_raw", "theta_bio_deletion_raw",
        "theta_bio_ga_raw",
        "theta_err_raw",
    }
    skip_prefixes = ("log_likelihood",)

    names: list[str] = []
    for v in sorted(idata_obj.posterior.data_vars):
        if v in skip_exact:
            continue
        if any(v.startswith(p) for p in skip_prefixes):
            continue
        da = idata_obj.posterior[v]
        if "obs" in da.dims:
            continue
        names.append(v)
    return names


def ppc_mean_subset(var_name: str) -> np.ndarray:
    arr = ppc.posterior_predictive[var_name]
    return arr.mean(dim=("chain", "draw")).values


def assemble_ppc_counts(supply_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    n_obs = len(supply_df)
    ppc_wt = ppc_mean_subset("y_wt")
    ppc_control = np.full(n_obs, np.nan)

    if "control_corrected" in supply_df.columns:
        is_cc = supply_df["control_corrected"].astype(bool).to_numpy()
    else:
        is_cc = ~supply_df["mut_type"].isin(WT_ONLY_MUT_TYPES).to_numpy()

    if "y_control" in ppc.posterior_predictive:
        y_ctrl = ppc_mean_subset("y_control")
        ppc_control[is_cc] = y_ctrl

    return ppc_control, ppc_wt


def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGDIR / name, dpi=300, bbox_inches="tight")
    plt.close()


# =============================================================================
# 1. Diagnostics
# =============================================================================

diag_var_names = posterior_diag_var_names(idata)
print(f"Computing diagnostics for {len(diag_var_names)} posterior variables.")

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=RuntimeWarning)
    diag = az.summary(idata, kind="diagnostics", var_names=diag_var_names)
diag.to_csv(OUTDIR / "postfit_diagnostics_full.csv")

problem_diag = diag.loc[(diag["r_hat"] > 1.01) | (diag["ess_bulk"] < 400) | (diag["ess_tail"] < 400)].copy()
problem_diag.to_csv(OUTDIR / "postfit_diagnostics_problem_parameters.csv")

with open(OUTDIR / "postfit_model_checklist.txt", "w") as f:
    f.write("mra post-fit checklist\n")
    f.write("=====================\n")
    f.write(f"Parameters with r_hat > 1.01 or ESS < 400: {len(problem_diag)}\n")
    f.write(f"Model: {metadata.get('model', 'mra')}\n")
    f.write(f"Position splines: bio={metadata.get('include_position_bio')}, err={metadata.get('include_position_err')}, df={metadata.get('spline_df')}, smoothness_scale={metadata.get('spline_smoothness_scale')}\n")
    f.write(f"WT-only types: {list(WT_ONLY_MUT_TYPES)}\n")
    f.write(f"No-context types: {list(NO_CONTEXT_MUT_TYPES)}\n")
    f.write(f"Context (nucleotide) types: {metadata.get('context_mut_types')}\n")
    f.write(f"Likelihood map: {json.dumps(metadata.get('likelihood_by_mut_type', {}), indent=2)}\n")
    f.write("Recommended checks:\n")
    f.write("1. Inspect divergences, R-hat, ESS (especially sigma_spline_*).\n")
    f.write("2. WT PPC for all types; control PPC for control-corrected types only.\n")
    f.write("3. Context: G->A 2-mer; other subs 1-base; indels none.\n")
    n_splines = metadata.get("n_bio_position_splines", 6)
    ga_spline_ctx = metadata.get("ga_spline_context_levels", [])
    f.write(
        f"   Bio position splines: {n_splines} "
        f"(G->A {ga_spline_ctx or ['AA', 'GA', 'SHARED']} + non-G->A shared + insertion + deletion); "
        f"sum_to_zero={metadata.get('spline_sum_to_zero', True)}; "
        f"sigma_floor={metadata.get('prior_sigma_spline_floor', metadata.get('ga_spline_sigma_floor'))}.\n"
    )
    if ga_spline_ctx:
        f.write(f"   G->A spline contexts ({len(ga_spline_ctx)}): {ga_spline_ctx}\n")
    shared_label = metadata.get("ga_spline_shared_label", "SHARED")
    ga_2mer = metadata.get("ga_ga_2mer", "GA")
    f.write(
        f"   Dedicated G->A splines: {metadata.get('ga_aa_2mer', 'AA')}, {ga_2mer}; "
        f"remaining 2-mers use {shared_label}.\n"
    )
    pooled_ga = metadata.get("pooled_ga_2mer_contexts", [])
    if pooled_ga:
        f.write(f"   G->A sparse 2-mers pooled to OTHER intercept + {shared_label} spline: {pooled_ga}\n")
    ga_ctx = metadata.get("ga_context_levels", [])
    f.write(f"   G->A 2-mer contexts ({len(ga_ctx)}): {ga_ctx}\n")
    f.write(f"   Shared-sub spline types: {metadata.get('shared_sub_mut_types', [])}\n")
    f.write(f"   All possible 2-mers: {metadata.get('all_ga_2mer_contexts', [])}\n")
    f.write(f"   Spline rules: {json.dumps(metadata.get('spline_rules', {}), indent=2)}\n")
    f.write("4. Review figures: ppc_*, mutation_spectrum, context_bio_supply_GtoA, position_bio_rate_*.\n")
    f.write("5. Confirm G->A calibration and overall WT PPC are well-behaved.\n")

# =============================================================================
# 2. Posterior predictive
# =============================================================================

ppc_control, ppc_wt = assemble_ppc_counts(supply)

supply["ppc_control_count_mean"] = ppc_control
supply["ppc_wt_count_mean"] = ppc_wt

ppc_by_mut = supply.groupby("mut_type", as_index=False).agg(
    control_corrected=("control_corrected", "first"),
    uses_nucleotide_context=("uses_nucleotide_context", "first"),
    correction_scheme=("correction_scheme", "first"),
    likelihood_family=("likelihood_family", "first"),
    obs_control=("control_count", "sum"),
    pred_control=("ppc_control_count_mean", "sum"),
    obs_wt=("wt_count", "sum"),
    pred_wt=("ppc_wt_count_mean", "sum"),
    bio_rate_mean=("bio_rate_mean", "mean"),
    error_rate_mean=("error_rate_mean", "mean"),
)
ppc_by_mut["wt_ppc_ratio"] = ppc_by_mut["pred_wt"] / ppc_by_mut["obs_wt"].replace(0, np.nan)
ppc_by_mut.to_csv(OUTDIR / "postfit_ppc_by_mutation_type.csv", index=False)

ppc_wt_only = ppc_by_mut[["mut_type", "obs_wt", "pred_wt", "wt_ppc_ratio", "control_corrected"]].copy()
ppc_wt_only.to_csv(OUTDIR / "postfit_ppc_wt_by_mutation_type.csv", index=False)

plt.figure(figsize=(5, 5))
x = ppc_by_mut["obs_wt"].to_numpy()
y = ppc_by_mut["pred_wt"].to_numpy()
plt.scatter(x + 0.5, y + 0.5)
lim = [0.5, max(np.nanmax(x), np.nanmax(y)) * 1.2 + 0.5]
plt.plot(lim, lim, linestyle="--")
for _, r in ppc_by_mut.iterrows():
    label = str(r["mut_type"])
    if not r["control_corrected"]:
        label += " (WT-only)"
    plt.text(r["obs_wt"] + 0.5, r["pred_wt"] + 0.5, label, fontsize=7)
plt.xscale("log")
plt.yscale("log")
plt.xlabel("Observed WT count + 0.5")
plt.ylabel("PPC mean WT count + 0.5")
plt.title("mra WT: observed vs PPC (all types)")
savefig("ppc_wt_by_mut_type.png")

ppc_cc = ppc_by_mut.loc[ppc_by_mut["control_corrected"]].copy()
ppc_cc.to_csv(OUTDIR / "postfit_ppc_control_corrected_types.csv", index=False)

if len(ppc_cc) > 0:
    plt.figure(figsize=(5, 5))
    x = ppc_cc["obs_control"].to_numpy()
    y = ppc_cc["pred_control"].to_numpy()
    plt.scatter(x + 0.5, y + 0.5)
    lim = [0.5, max(np.nanmax(x), np.nanmax(y)) * 1.2 + 0.5]
    plt.plot(lim, lim, linestyle="--")
    for _, r in ppc_cc.iterrows():
        plt.text(r["obs_control"] + 0.5, r["pred_control"] + 0.5, str(r["mut_type"]), fontsize=8)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Observed control count + 0.5")
    plt.ylabel("PPC mean control count + 0.5")
    plt.title("mra Control PPC (corrected types only)")
    savefig("ppc_control_by_mut_type.png")

# =============================================================================
# 3. Mutation spectrum
# =============================================================================

mut_spec = supply.groupby("mut_type", as_index=False).agg(
    control_corrected=("control_corrected", "first"),
    likelihood_family=("likelihood_family", "first"),
    bio_rate_mean=("bio_rate_mean", "mean"),
    bio_rate_median=("bio_rate_median", "median"),
    error_rate_mean=("error_rate_mean", "mean"),
    wt_total_rate_mean=("wt_total_rate_mean", "mean"),
    bio_fraction=("bio_fraction_of_wt_total_mean", "mean"),
)
mut_spec = mut_spec.sort_values("bio_rate_mean", ascending=True)
mut_spec.to_csv(OUTDIR / "postfit_mutation_spectrum.csv", index=False)

plt.figure(figsize=(7, 5))
y = np.arange(len(mut_spec))
plt.scatter(mut_spec["error_rate_mean"], y, label="Error/background (0 for WT-only)")
plt.scatter(mut_spec["bio_rate_mean"], y, label="Released supply (bio_rate)")
plt.yticks(y, mut_spec["mut_type"])
plt.xscale("log")
plt.xlabel("Mean rate")
plt.ylabel("Mutation type")
plt.title("mra mutation spectrum")
plt.legend()
savefig("mutation_spectrum_bio_vs_error.png")

# =============================================================================
# 4. Context summaries (all context-bearing types)
# =============================================================================

ctx = supply.loc[supply["uses_nucleotide_context"]].groupby(
    ["mut_type", "context_3p", "context_scheme"], as_index=False
).agg(
    n_sites=("ref_position", "size"),
    bio_rate_mean=("bio_rate_mean", "mean"),
    bio_rate_median=("bio_rate_median", "median"),
    error_rate_mean=("error_rate_mean", "mean"),
    wt_count=("wt_count", "sum"),
    control_count=("control_count", "sum"),
)
ctx.to_csv(OUTDIR / "postfit_context_summary.csv", index=False)

ga_ctx = ctx.loc[ctx["mut_type"] == GA_MUT_TYPE].copy()
if len(ga_ctx) >= 2:
    ga_ctx = ga_ctx.sort_values("bio_rate_mean", ascending=True)
    plt.figure(figsize=(7, max(4, 0.18 * len(ga_ctx))))
    y_pos = np.arange(len(ga_ctx))
    plt.scatter(ga_ctx["bio_rate_mean"], y_pos)
    plt.yticks(y_pos, ga_ctx["context_3p"])
    plt.xscale("log")
    plt.xlabel("Mean bio_rate (released supply)")
    plt.ylabel("G->A 2-mer context (intercept; AA / GA / shared splines)")
    plt.title("G->A context-dependent supply (intercept levels)")
    savefig("context_bio_supply_GtoA.png")

# =============================================================================
# 5. Position profiles — G->A (AA / GA / shared) + shared sub + indels
# =============================================================================

def _position_profile(sub: pd.DataFrame) -> pd.DataFrame:
    pos_prof = sub.groupby("ref_position", as_index=False).agg(
        bio_rate_mean=("bio_rate_mean", "mean"),
        wt_count=("wt_count", "sum"),
        wt_coverage=("wt_coverage", "mean"),
    )
    pos_prof["empirical_rate"] = pos_prof["wt_count"] / pos_prof["wt_coverage"].replace(0, np.nan)
    return pos_prof


def _plot_position_profile(sub: pd.DataFrame, title: str, out_csv: Path, out_png: str) -> None:
    if sub.empty:
        return
    pos_prof = _position_profile(sub)
    pos_prof.to_csv(out_csv, index=False)
    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(pos_prof["ref_position"], pos_prof["bio_rate_mean"], color="C0", linewidth=1, label="bio_rate mean")
    ax1.set_yscale("log")
    ax1.set_xlabel("Reference position")
    ax1.set_ylabel("bio_rate (log scale)", color="C0")
    ax2 = ax1.twinx()
    ax2.bar(
        pos_prof["ref_position"],
        pos_prof["wt_count"],
        width=1.0,
        alpha=0.25,
        color="C1",
        label="WT count",
    )
    ax2.set_ylabel("WT count (sum per position)", color="C1")
    plt.title(title)
    fig.tight_layout()
    savefig(out_png)


ga_aa_2mer = str(metadata.get("ga_aa_2mer", "AA"))
ga_ga_2mer = str(metadata.get("ga_ga_2mer", "GA"))
shared_label = str(metadata.get("ga_spline_shared_label", "SHARED"))
pooled_ga = metadata.get("pooled_ga_2mer_contexts", [])
ga_sub = supply.loc[supply["mut_type"] == GA_MUT_TYPE].copy()

if "ga_2mer_raw" in ga_sub.columns:
    ga_aa_sub = ga_sub.loc[ga_sub["ga_2mer_raw"].astype(str).eq(ga_aa_2mer)]
    ga_ga_sub = ga_sub.loc[ga_sub["ga_2mer_raw"].astype(str).eq(ga_ga_2mer)]
    ga_shared_sub = ga_sub.loc[
        ~ga_sub["ga_2mer_raw"].astype(str).isin([ga_aa_2mer, ga_ga_2mer])
    ]
elif "ga_spline_context" in ga_sub.columns and "context_3p" in ga_sub.columns:
    # Newer fits merge GA into SHARED for the spline index; still split plots by 2-mer.
    ga_aa_sub = ga_sub.loc[ga_sub["context_3p"].astype(str).eq(ga_aa_2mer)]
    ga_ga_sub = ga_sub.loc[ga_sub["context_3p"].astype(str).eq(ga_ga_2mer)]
    ga_shared_sub = ga_sub.loc[~ga_sub["context_3p"].astype(str).isin([ga_aa_2mer, ga_ga_2mer])]
elif "ga_spline_context" in ga_sub.columns:
    ga_aa_sub = ga_sub.loc[ga_sub["ga_spline_context"].astype(str).eq(ga_aa_2mer)]
    ga_ga_sub = ga_sub.loc[ga_sub["ga_spline_context"].astype(str).eq(ga_ga_2mer)]
    ga_shared_sub = ga_sub.loc[ga_sub["ga_spline_context"].astype(str).eq(shared_label)]
else:
    ga_aa_sub = ga_sub.loc[ga_sub["context_3p"].astype(str).eq(ga_aa_2mer)]
    ga_ga_sub = ga_sub.loc[ga_sub["context_3p"].astype(str).eq(ga_ga_2mer)]
    ga_shared_sub = ga_sub.loc[~ga_sub["context_3p"].astype(str).isin([ga_aa_2mer, ga_ga_2mer])]

_plot_position_profile(
    ga_aa_sub,
    f"mra G->A {ga_aa_2mer} (dedicated spline)",
    OUTDIR / "postfit_position_profile_GtoA_AA.csv",
    "position_bio_rate_GtoA_AA.png",
)
_plot_position_profile(
    ga_ga_sub,
    f"mra G->A {ga_ga_2mer} (dedicated spline)",
    OUTDIR / "postfit_position_profile_GtoA_GA.csv",
    "position_bio_rate_GtoA_GA.png",
)
_plot_position_profile(
    ga_shared_sub,
    f"mra G->A all other 2-mers ({shared_label} spline)",
    OUTDIR / f"postfit_position_profile_GtoA_{shared_label}.csv",
    "position_bio_rate_GtoA_shared.png",
)

shared_sub_types = metadata.get("shared_sub_mut_types", [])
shared_sub = supply.loc[supply["mut_type"].isin(shared_sub_types)].copy()
_plot_position_profile(
    shared_sub,
    "mra non-G->A substitutions (shared position spline)",
    OUTDIR / "postfit_position_profile_shared_sub.csv",
    "position_bio_rate_shared_sub.png",
)

for mut_type, title, csv_name, png_name in (
    ("insertion", "mra insertion (dedicated position spline)", "postfit_position_profile_insertion.csv", "position_bio_rate_insertion.png"),
    ("deletion", "mra deletion (dedicated position spline)", "postfit_position_profile_deletion.csv", "position_bio_rate_deletion.png"),
):
    if mut_type not in set(supply["mut_type"]):
        continue
    _plot_position_profile(
        supply.loc[supply["mut_type"] == mut_type],
        title,
        OUTDIR / csv_name,
        png_name,
    )

# =============================================================================
# 6. Site-level R²
# =============================================================================

pred = supply["bio_rate_mean"] * supply["wt_coverage"] + supply["error_rate_mean"] * supply["wt_coverage"]
obs = supply["wt_count"]
ss_res = ((obs - pred) ** 2).sum()
ss_tot = ((obs - obs.mean()) ** 2).sum()
r2_rows = [{"scope": "pooled", "mut_type": "all", "r2": 1 - ss_res / ss_tot, "n": len(supply)}]
for mt, g in supply.groupby("mut_type"):
    o = g["wt_count"]
    p = g["bio_rate_mean"] * g["wt_coverage"] + g["error_rate_mean"] * g["wt_coverage"]
    r2 = 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum() if o.var() > 0 else float("nan")
    r2_rows.append({"scope": "by_type", "mut_type": mt, "r2": r2, "n": len(g)})
r2_df = pd.DataFrame(r2_rows)
r2_df.to_csv(OUTDIR / "postfit_r2_by_mutation_type.csv", index=False)

# =============================================================================
# 7. Release table
# =============================================================================

release_cols = [
    "ref_position", "ref_nucleotide", "mut_type", "context_3p", "context_scheme",
    "uses_nucleotide_context", "control_corrected", "correction_scheme", "likelihood_family",
    "wt_count", "wt_coverage", "control_count", "control_coverage",
    "bio_rate_mean", "bio_rate_median", "bio_rate_lo95", "bio_rate_hi95",
    "error_rate_mean", "error_rate_median", "error_rate_lo95", "error_rate_hi95",
    "wt_total_rate_mean", "wt_total_rate_median", "wt_total_rate_lo95", "wt_total_rate_hi95",
    "bio_fraction_of_wt_total_mean", "error_fraction_of_wt_total_mean",
]
release_cols = [c for c in release_cols if c in supply.columns]
release = supply[release_cols].copy()
release.to_csv(OUTDIR / "FINAL_mutation_supply_dataset.csv", index=False)

print("Saved post-fit outputs to", OUTDIR)
print("Figures saved to", FIGDIR)
print("Problem diagnostics rows:", len(problem_diag))
print("Pooled site R²:", round(1 - ss_res / ss_tot, 4))
print("Final supply table:", OUTDIR / "FINAL_mutation_supply_dataset.csv")
