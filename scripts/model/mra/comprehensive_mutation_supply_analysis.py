#!/usr/bin/env python3
"""
Comprehensive mutation supply analysis (gene-aware AA tables and modules).

mra port of the R11 comprehensive analysis. Reads FINAL_mutation_supply_dataset.csv
and writes the same output bundle as the legacy comprehensive_analysis_R11_product
(site/AA/path tables, modules 1–5, bootstrap summary, aa_mutation_entropy.csv).

Gene-aware AA path enumeration uses NL43_protein_codons.csv (9 HIV-1 NL4-3
proteins). Without it the analysis falls back to a single-frame GENOME protein
(codon_annotation_mode=frame_inferred), which does NOT match the R11 product.

Default paths when MUT_SUPPLY_OUTDIR is set:
  --final-dataset       $MUT_SUPPLY_OUTDIR/FINAL_mutation_supply_dataset.csv
  --output-dir          $MUT_SUPPLY_OUTDIR/comprehensive_analysis_{model_version}_product
  --protein-codons-file <project>/data/mutation_supply/annotation/NL43/NL43_protein_codons.csv
"""

from __future__ import annotations

import argparse
import itertools
import os
import textwrap
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTEIN_CODONS = PROJECT_ROOT / "data/mutation_supply/annotation/NL43/NL43_protein_codons.csv"

GENETIC_CODE: Dict[str, str] = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L", "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*", "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L", "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q", "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M", "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K", "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V", "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E", "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}

BASES = ("A", "C", "G", "T")

CI_METHODOLOGY_TEXT = textwrap.dedent(
    """
    AA-level supply (mu_aa) and 95% interval columns (mu_aa_lo95, mu_aa_hi95)
    ---------------------------------------------------------------------------
    - mu_aa: sum of mu_path over all mutant-codon paths that yield the same
      (protein, AA position, alt_aa) / aa_change.
    - Per path, mu_path: product of bio_rate_median at each changed nucleotide site.
    - mu_path_lo95 / mu_path_hi95: product of bio_rate_lo95 / bio_rate_hi95 along
      the same sites (2- and 3-step paths: product of marginal lower/upper bounds).
    - mu_aa_lo95 / mu_aa_hi95: sum of mu_path_lo95 / mu_path_hi95 over paths.

    These intervals approximate joint uncertainty; they are useful for ranking
    and sensitivity, not as exact simultaneous credible intervals.
    """
).strip()


def normalize_site_rates(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "ref_position": "position",
        "ref_nucleotide": "ref_nt",
        "mut_type": "mutation_type",
        "bio_rate_median": "rate",
        "wt_coverage": "coverage",
    }
    out = df.rename(columns=rename_map).copy()
    if "rate" not in out.columns:
        raise ValueError("Input must contain `bio_rate_median` or normalized `rate` column.")

    if "bio_rate_lo95" in df.columns and "bio_rate_hi95" in df.columns:
        out["rate_lo95"] = pd.to_numeric(df["bio_rate_lo95"], errors="coerce")
        out["rate_hi95"] = pd.to_numeric(df["bio_rate_hi95"], errors="coerce")
    else:
        out["rate_lo95"] = np.nan
        out["rate_hi95"] = np.nan

    out["alt_nt"] = out["mutation_type"].str.split("->").str[1]
    is_sub = out["mutation_type"].str.match(r"^[ACGT]->[ACGT]$")
    out = out[is_sub].copy()
    out["position"] = out["position"].astype(int)
    out["rate"] = pd.to_numeric(out["rate"], errors="coerce")
    return out.dropna(subset=["position", "ref_nt", "alt_nt", "rate"])


def add_logs(df: pd.DataFrame, value_col: str, new_col: str, pseudocount: float) -> pd.DataFrame:
    out = df.copy()
    out[new_col] = np.log10(out[value_col].clip(lower=pseudocount))
    return out


def codon_to_aa(codon: str) -> str:
    return GENETIC_CODE.get(codon.upper(), "X")


def build_codon_table(site_df: pd.DataFrame, frame_offset: int) -> pd.DataFrame:
    unique_pos = site_df[["position", "ref_nt"]].drop_duplicates().sort_values("position")
    unique_pos["codon_idx"] = ((unique_pos["position"] - 1 - frame_offset) // 3).astype(int)
    unique_pos["codon_pos"] = ((unique_pos["position"] - 1 - frame_offset) % 3).astype(int)
    valid = unique_pos[unique_pos["codon_idx"] >= 0].copy()

    rows = []
    for codon_idx, grp in valid.groupby("codon_idx"):
        if set(grp["codon_pos"]) != {0, 1, 2}:
            continue
        grp = grp.sort_values("codon_pos")
        codon = "".join(grp["ref_nt"].tolist())
        positions = grp["position"].tolist()
        rows.append(
            {
                "codon_idx": codon_idx,
                "position_1": positions[0],
                "position_2": positions[1],
                "position_3": positions[2],
                "codon": codon,
                "ref_aa": codon_to_aa(codon),
            }
        )
    return pd.DataFrame(rows)


def build_codon_table_from_annotation(protein_codons_df: pd.DataFrame, site_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "protein",
        "AA_position",
        "AA",
        "codon",
        "codon_genomic_pos_1",
        "codon_genomic_pos_2",
        "codon_genomic_pos_3",
    }
    if not required.issubset(protein_codons_df.columns):
        raise ValueError(f"Protein codons table must include: {sorted(required)}")

    available_positions = set(site_df["position"].astype(int).unique().tolist())
    c = protein_codons_df.copy()
    c["position_1"] = c["codon_genomic_pos_1"].astype(int)
    c["position_2"] = c["codon_genomic_pos_2"].astype(int)
    c["position_3"] = c["codon_genomic_pos_3"].astype(int)

    c = c[
        c["position_1"].isin(available_positions)
        & c["position_2"].isin(available_positions)
        & c["position_3"].isin(available_positions)
    ].copy()

    c = c.rename(columns={"AA": "ref_aa", "AA_position": "aa_position"})
    out_cols = ["protein", "aa_position", "codon", "ref_aa", "position_1", "position_2", "position_3"]
    return c[out_cols].drop_duplicates()


def build_rate_lookup(site_df: pd.DataFrame) -> Dict[Tuple[int, str], Tuple[float, float, float]]:
    """
    (position, mut_type) -> (median, lo95, hi95).
    Missing CI in input: lo and hi are set to median (interval collapses to point).
    Missing keys at lookup time use pseudocount for (med, lo, hi).
    """
    lookup: Dict[Tuple[int, str], Tuple[float, float, float]] = {}
    for r in site_df.itertuples(index=False):
        key = (int(r.position), f"{r.ref_nt}->{r.alt_nt}")
        med = float(r.rate)
        lo = float(r.rate_lo95) if pd.notna(r.rate_lo95) else med
        hi = float(r.rate_hi95) if pd.notna(r.rate_hi95) else med
        lookup[key] = (med, lo, hi)
    return lookup


def _lookup_rate(
    lookup: Dict[Tuple[int, str], Tuple[float, float, float]],
    pos: int,
    mut_type: str,
    pseudocount: float,
) -> Tuple[float, float, float]:
    t = lookup.get((pos, mut_type))
    if t is None:
        return (pseudocount, pseudocount, pseudocount)
    med, lo, hi = t
    if np.isnan(lo) or np.isnan(hi):
        return (med, med, med)
    return (med, lo, hi)


def enumerate_aa_paths(
    codon_df: pd.DataFrame,
    rate_lookup: Dict[Tuple[int, str], Tuple[float, float, float]],
    pseudocount: float,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for row in codon_df.itertuples(index=False):
        wt = row.codon
        wt_aa = row.ref_aa
        pos_map = [row.position_1, row.position_2, row.position_3]

        for alt_bases in itertools.product(BASES, repeat=3):
            alt_codon = "".join(alt_bases)
            if alt_codon == wt:
                continue

            diffs = [i for i in range(3) if wt[i] != alt_codon[i]]
            n_changes = len(diffs)
            rates: List[float] = []
            los: List[float] = []
            his: List[float] = []
            path_parts = []
            for i in diffs:
                mut_type = f"{wt[i]}->{alt_codon[i]}"
                med, lo, hi = _lookup_rate(rate_lookup, pos_map[i], mut_type, pseudocount)
                rates.append(med)
                los.append(lo)
                his.append(hi)
                path_parts.append(mut_type)

            mu_path = float(np.prod(rates))
            mu_path_lo = float(np.prod(los)) if los else float("nan")
            mu_path_hi = float(np.prod(his)) if his else float("nan")

            alt_aa = codon_to_aa(alt_codon)
            aa_pos = int(getattr(row, "aa_position", getattr(row, "codon_idx", -1) + 1))
            protein = str(getattr(row, "protein", "GENOME"))
            aa_change = f"{wt_aa}{aa_pos}{alt_aa}"
            mutation_id = f"{protein}:{aa_change}"
            rows.append(
                {
                    "position": aa_pos,
                    "protein": protein,
                    "codon": wt,
                    "mutant_codon": alt_codon,
                    "ref_aa": wt_aa,
                    "alt_aa": alt_aa,
                    "aa_change": aa_change,
                    "mutation": mutation_id,
                    "codon_position_1": pos_map[0],
                    "codon_position_2": pos_map[1],
                    "codon_position_3": pos_map[2],
                    "n_nt_changes": n_changes,
                    "mutation_type_path": ";".join(path_parts),
                    "mu_path": mu_path,
                    "mu_path_lo95": mu_path_lo,
                    "mu_path_hi95": mu_path_hi,
                    "synonymous": wt_aa == alt_aa,
                }
            )
    return pd.DataFrame(rows)


def summarize_aa_supply(path_df: pd.DataFrame, pseudocount: float) -> pd.DataFrame:
    aa = path_df.groupby(
        ["protein", "position", "aa_change", "mutation", "ref_aa", "alt_aa", "synonymous"],
        as_index=False,
    ).agg(
        mu_aa=("mu_path", "sum"),
        mu_aa_lo95=("mu_path_lo95", "sum"),
        mu_aa_hi95=("mu_path_hi95", "sum"),
        min_nt_changes=("n_nt_changes", "min"),
        n_paths=("mu_path", "size"),
    )

    aa["accessible_class"] = aa["min_nt_changes"].astype(int)
    aa["log10_mu_aa"] = np.log10(aa["mu_aa"].clip(lower=pseudocount))
    aa["log10_mu_aa_lo95"] = np.log10(aa["mu_aa_lo95"].clip(lower=pseudocount))
    aa["log10_mu_aa_hi95"] = np.log10(aa["mu_aa_hi95"].clip(lower=pseudocount))
    return aa


def compute_aa_mutation_entropy(path_df: pd.DataFrame) -> pd.DataFrame:
    """Shannon entropy of nonsynonymous single-step AA mutation supply per protein position."""
    aa_supply = path_df.loc[path_df["n_nt_changes"] == 1].copy()
    aa_supply = aa_supply.loc[~aa_supply["synonymous"] & (aa_supply["ref_aa"] != "*")]

    aa_freq = aa_supply.copy()
    aa_freq["rel_freq"] = aa_freq.groupby(["protein", "position"])["mu_path"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0.0
    )

    def _entropy(group: pd.Series) -> float:
        p = group[group > 0]
        if p.empty:
            return 0.0
        return float(-(p * np.log(p)).sum())

    aa_entropy = (
        aa_freq.groupby(["protein", "position"], as_index=False)
        .agg(entropy=("rel_freq", _entropy), total_mutations=("rel_freq", "size"))
    )
    ref_aa = aa_supply[["protein", "position", "ref_aa"]].drop_duplicates()
    return aa_entropy.merge(ref_aa, on=["protein", "position"], how="left")


def plot_module_1(nt_df: pd.DataFrame, aa_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if sns is not None:
        sns.kdeplot(data=nt_df, x="log10_mu_nt", hue="mutation_type", common_norm=False, ax=axes[0], linewidth=1.0)
    else:
        for mut_type, grp in nt_df.groupby("mutation_type"):
            axes[0].hist(grp["log10_mu_nt"], bins=40, alpha=0.25, density=True, label=mut_type)
        axes[0].legend(fontsize=7, ncol=2)
    axes[0].set_title("NT mutation supply distribution")
    axes[0].set_xlabel("log10(mu_nt)")

    axes[1].hist(aa_df["log10_mu_aa"], bins=60, alpha=0.85)
    axes[1].set_title("AA mutation supply distribution")
    axes[1].set_xlabel("log10(mu_aa)")
    fig.tight_layout()
    fig.savefig(out_dir / "module1_global_distributions.png", dpi=200)
    plt.close(fig)

    group_key = "mutation" if "mutation" in aa_df.columns else "aa_change"
    grp = aa_df.groupby(group_key)["log10_mu_aa"]
    within_var = float(grp.var().fillna(0.0).mean())
    between_var = float(grp.mean().var())
    summary = pd.DataFrame([{"within_aa_variance": within_var, "between_aa_variance": between_var}])
    summary.to_csv(out_dir / "module1_variance_decomposition.csv", index=False)
    return summary


def plot_module_2(aa_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    group_key = "mutation" if "mutation" in aa_df.columns else "aa_change"
    stats = aa_df.groupby(group_key, as_index=False)["log10_mu_aa"].agg(
        mean="mean", q1=lambda x: x.quantile(0.25), q3=lambda x: x.quantile(0.75), min="min", max="max"
    )
    stats["iqr"] = stats["q3"] - stats["q1"]
    stats = stats.sort_values("mean", ascending=False)
    stats.to_csv(out_dir / "module2_within_between_aa_summary.csv", index=False)

    plot_stats = stats.head(80).sort_values("mean")
    fig, ax = plt.subplots(figsize=(10, 14))
    y = np.arange(len(plot_stats))
    ax.hlines(y=y, xmin=plot_stats["q1"], xmax=plot_stats["q3"], color="steelblue", linewidth=2)
    ax.plot(plot_stats["mean"], y, "o", color="black", markersize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_stats[group_key], fontsize=7)
    ax.set_xlabel("log10(mu_aa)")
    ax.set_title("Module 2: Within vs Between AA variation (top 80 by mean)")
    fig.tight_layout()
    fig.savefig(out_dir / "module2_interval_plot.png", dpi=220)
    plt.close(fig)
    return stats


def plot_module_3(aa_df: pd.DataFrame, out_dir: Path) -> None:
    idx = "mutation" if "mutation" in aa_df.columns else "aa_change"
    hm = aa_df.pivot_table(index=idx, columns="position", values="log10_mu_aa", aggfunc="mean")
    fig_h = min(24, max(8, 0.2 * hm.shape[0]))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    if sns is not None:
        sns.heatmap(hm, cmap="viridis", ax=ax, cbar_kws={"label": "log10(mu_aa)"})
    else:
        im = ax.imshow(hm.fillna(np.nan).values, aspect="auto", cmap="viridis", interpolation="nearest")
        plt.colorbar(im, ax=ax, label="log10(mu_aa)")
        ax.set_yticks(np.arange(hm.shape[0]))
        ax.set_yticklabels(hm.index.tolist(), fontsize=6)
        ax.set_xticks(np.linspace(0, hm.shape[1] - 1, min(20, hm.shape[1]), dtype=int))
        ax.set_xticklabels([str(hm.columns[i]) for i in np.linspace(0, hm.shape[1] - 1, min(20, hm.shape[1]), dtype=int)])
    ax.set_title("Module 3: AA supply landscape")
    fig.tight_layout()
    fig.savefig(out_dir / "module3_aa_supply_heatmap.png", dpi=220)
    plt.close(fig)


def plot_module_4(aa_df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if sns is not None:
        sns.boxplot(data=aa_df, x="accessible_class", y="log10_mu_aa", ax=ax)
    else:
        grouped = [grp["log10_mu_aa"].values for _, grp in aa_df.groupby("accessible_class")]
        labels = [str(k) for k in sorted(aa_df["accessible_class"].unique())]
        ax.boxplot(grouped, labels=labels)
    ax.set_xlabel("Number of nucleotide changes")
    ax.set_ylabel("log10(mu_aa)")
    ax.set_title("Module 4: Accessibility constraints")
    fig.tight_layout()
    fig.savefig(out_dir / "module4_accessibility_boxplot.png", dpi=220)
    plt.close(fig)


def plot_module_5(aa_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    agg = aa_df.groupby("alt_aa", as_index=False)["mu_aa"].sum().sort_values("mu_aa", ascending=False)
    agg["log10_total_supply"] = np.log10(agg["mu_aa"].clip(lower=1e-20))
    agg.to_csv(out_dir / "module5_alt_aa_total_supply.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    if sns is not None:
        sns.barplot(data=agg, x="alt_aa", y="log10_total_supply", color="slateblue", ax=ax)
    else:
        ax.bar(agg["alt_aa"], agg["log10_total_supply"], color="slateblue")
    ax.set_title("Module 5: Aggregate amino acid accessibility")
    ax.set_ylabel("log10(total incoming supply)")
    fig.tight_layout()
    fig.savefig(out_dir / "module5_ranked_alt_aa_supply.png", dpi=220)
    plt.close(fig)
    return agg


def ols_fit(df: pd.DataFrame, y_col: str, x_col: str) -> Optional[Dict[str, float]]:
    if sm is None or len(df) < 4:
        return None
    x = sm.add_constant(df[x_col].astype(float))
    y = df[y_col].astype(float)
    fit = sm.OLS(y, x).fit()
    return {
        "slope": float(fit.params[x_col]),
        "intercept": float(fit.params["const"]),
        "r2": float(fit.rsquared),
        "pvalue_slope": float(fit.pvalues[x_col]),
    }


def run_optional_escape_modules(aa_df: pd.DataFrame, escape_path: Path, out_dir: Path, pseudocount: float) -> None:
    esc = pd.read_csv(escape_path)
    needed = {"mutation", "category"}
    if not needed.issubset(esc.columns):
        raise ValueError(f"Escape file must include columns: {sorted(needed)}")

    if "mutation" in aa_df.columns:
        merged = esc.merge(aa_df, on="mutation", how="left")
        if merged["mu_aa"].notna().sum() == 0:
            merged = esc.merge(aa_df, left_on="mutation", right_on="aa_change", how="left")
    else:
        merged = esc.merge(aa_df, left_on="mutation", right_on="aa_change", how="left")
    merged["log10_mu_aa"] = np.log10(merged["mu_aa"].fillna(0.0).clip(lower=pseudocount))
    merged.to_csv(out_dir / "module6_escape_merged.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    if sns is not None:
        sns.boxplot(data=merged, x="category", y="log10_mu_aa", ax=ax)
    else:
        cats = list(merged["category"].dropna().unique())
        values = [merged.loc[merged["category"] == c, "log10_mu_aa"].values for c in cats]
        ax.boxplot(values, labels=cats)
    ax.set_title("Module 6: Escape mutation supply by category")
    fig.tight_layout()
    fig.savefig(out_dir / "module6_escape_supply_distribution.png", dpi=220)
    plt.close(fig)

    ranked = merged.sort_values("mu_aa", ascending=False)
    ranked.to_csv(out_dir / "module6_escape_ranked_by_supply.csv", index=False)

    if "observed_frequency" in merged.columns:
        model_df = merged.dropna(subset=["observed_frequency", "log10_mu_aa"]).copy()
        fit = ols_fit(model_df, y_col="observed_frequency", x_col="log10_mu_aa")
        if fit is not None:
            pd.DataFrame([fit]).to_csv(out_dir / "module7_supply_vs_observed_regression.csv", index=False)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.scatter(model_df["log10_mu_aa"], model_df["observed_frequency"], s=25)
            xs = np.linspace(model_df["log10_mu_aa"].min(), model_df["log10_mu_aa"].max(), 100)
            ys = fit["intercept"] + fit["slope"] * xs
            ax.plot(xs, ys, color="crimson")
            ax.set_title("Module 7: Supply vs observed escape")
            fig.tight_layout()
            fig.savefig(out_dir / "module7_supply_vs_observed_scatter.png", dpi=220)
            plt.close(fig)

    if "fitness_effect" in merged.columns and "category" in merged.columns:
        fig, ax = plt.subplots(figsize=(7, 5))
        for cat, grp in merged.groupby("category"):
            ax.scatter(grp["log10_mu_aa"], grp["fitness_effect"], s=30, label=str(cat))
        ax.legend(fontsize=7)
        ax.set_title("Module 8: Within-epitope/category supply vs effect")
        fig.tight_layout()
        fig.savefig(out_dir / "module8_within_epitope_supply_vs_effect.png", dpi=220)
        plt.close(fig)


def run_optional_evolution_modules(aa_df: pd.DataFrame, evolution_path: Path, out_dir: Path, pseudocount: float) -> None:
    evo = pd.read_csv(evolution_path)
    needed = {"position", "diversity", "evolution_rate"}
    if not needed.issubset(evo.columns):
        raise ValueError(f"Evolution file must include columns: {sorted(needed)}")

    site_supply = aa_df.groupby("position", as_index=False)["mu_aa"].sum().rename(columns={"mu_aa": "total_supply"})
    merged = evo.merge(site_supply, on="position", how="left")
    merged["log10_supply"] = np.log10(merged["total_supply"].fillna(0.0).clip(lower=pseudocount))
    merged.to_csv(out_dir / "merged_evolution_dataset.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(merged["log10_supply"], merged["diversity"], s=20)
    axes[0].set_title("Module 9: Supply vs diversity")
    axes[1].scatter(merged["log10_supply"], merged["evolution_rate"], s=20)
    axes[1].set_title("Module 10: Supply vs evolution rate")
    fig.tight_layout()
    fig.savefig(out_dir / "module9_10_supply_vs_diversity_evolution.png", dpi=220)
    plt.close(fig)

    fit = ols_fit(merged.dropna(subset=["log10_supply", "evolution_rate"]), y_col="evolution_rate", x_col="log10_supply")
    if fit is not None:
        pred = fit["intercept"] + fit["slope"] * merged["log10_supply"]
        merged["evolution_residual"] = merged["evolution_rate"] - pred
        merged.sort_values("evolution_residual", ascending=False).to_csv(
            out_dir / "module10_evolution_residuals_ranked.csv", index=False
        )
        pd.DataFrame([fit]).to_csv(out_dir / "module10_supply_vs_evolution_regression.csv", index=False)


def save_bootstrap_summary(site_df: pd.DataFrame, out_dir: Path, n_boot: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    positions = site_df["position"].unique()
    rows = []
    for _ in range(n_boot):
        sample_positions = rng.choice(positions, size=len(positions), replace=True)
        boot = site_df[site_df["position"].isin(sample_positions)]
        rows.append(
            {
                "mean_log10_mu_nt": float(boot["log10_mu_nt"].mean()),
                "median_log10_mu_nt": float(boot["log10_mu_nt"].median()),
            }
        )
    pd.DataFrame(rows).describe().to_csv(out_dir / "robustness_bootstrap_site_summary.csv")


def resolve_default_paths(model_version: str) -> Tuple[Optional[Path], Optional[Path], Path]:
    outdir = os.environ.get("MUT_SUPPLY_OUTDIR")
    if outdir:
        base = Path(outdir)
        final_dataset = base / "FINAL_mutation_supply_dataset.csv"
        output_dir = base / f"comprehensive_analysis_{model_version}_product"
    else:
        final_dataset = None
        output_dir = None
    protein_codons = DEFAULT_PROTEIN_CODONS
    return final_dataset, output_dir, protein_codons


def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive mutation supply analysis (with AA-level CIs).")
    parser.add_argument("--final-dataset", type=Path, default=None, help="Path to FINAL_mutation_supply_dataset.csv")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for outputs")
    parser.add_argument(
        "--model-version",
        type=str,
        default=os.environ.get("MUT_SUPPLY_MODEL_VERSION", "mra"),
        help="Model label for default output dir (e.g. mra)",
    )
    parser.add_argument("--frame-offset", type=int, default=0, help="Reading frame offset relative to position 1 (0,1,2)")
    parser.add_argument(
        "--protein-codons-file",
        type=Path,
        default=None,
        help="CDS/codon annotation CSV (gene-aware). Defaults to NL43_protein_codons.csv when present.",
    )
    parser.add_argument("--pseudocount", type=float, default=1e-10, help="Pseudocount for log10 transforms and missing rates")
    parser.add_argument("--escape-file", type=Path, default=None, help="Optional AA-level escape/resistance file")
    parser.add_argument("--evolution-file", type=Path, default=None, help="Optional site-level evolution/diversity file")
    parser.add_argument("--bootstrap-iterations", type=int, default=200, help="Bootstrap samples for robustness summary")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    default_final, default_output, default_protein_codons = resolve_default_paths(args.model_version)
    if args.final_dataset is None:
        args.final_dataset = default_final
    if args.output_dir is None:
        args.output_dir = default_output
    if args.protein_codons_file is None and default_protein_codons.exists():
        args.protein_codons_file = default_protein_codons

    if args.final_dataset is None or args.output_dir is None:
        parser.error("--final-dataset and --output-dir are required unless MUT_SUPPLY_OUTDIR is set")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if sns is not None:
        sns.set_theme(style="whitegrid")

    (args.output_dir / "aa_level_supply_ci_methodology.txt").write_text(CI_METHODOLOGY_TEXT + "\n", encoding="utf-8")

    raw = pd.read_csv(args.final_dataset)
    nt_df = normalize_site_rates(raw)
    nt_df = add_logs(nt_df, "rate", "log10_mu_nt", args.pseudocount)
    nt_df.to_csv(args.output_dir / "site_level_supply_table.csv", index=False)

    rate_lookup = build_rate_lookup(nt_df)
    if args.protein_codons_file is not None and args.protein_codons_file.exists():
        protein_codons_df = pd.read_csv(args.protein_codons_file)
        codon_df = build_codon_table_from_annotation(protein_codons_df, nt_df)
        codon_mode = "gene_annotated"
    else:
        codon_df = build_codon_table(nt_df, frame_offset=args.frame_offset)
        codon_df["protein"] = "GENOME"
        codon_df["aa_position"] = codon_df["codon_idx"].astype(int) + 1
        codon_mode = "frame_inferred"
    if len(codon_df) == 0:
        raise ValueError("No complete codons inferred. Check frame offset and position continuity.")

    path_df = enumerate_aa_paths(codon_df, rate_lookup=rate_lookup, pseudocount=args.pseudocount)
    path_df.to_csv(args.output_dir / "aa_mutation_path_table.csv", index=False)

    aa_df = summarize_aa_supply(path_df, pseudocount=args.pseudocount)
    aa_df.to_csv(args.output_dir / "aa_level_supply_table.csv", index=False)

    compute_aa_mutation_entropy(path_df).to_csv(args.output_dir / "aa_mutation_entropy.csv", index=False)

    var_summary = plot_module_1(nt_df, aa_df, args.output_dir)
    _ = plot_module_2(aa_df, args.output_dir)
    plot_module_3(aa_df, args.output_dir)
    plot_module_4(aa_df, args.output_dir)
    _ = plot_module_5(aa_df, args.output_dir)
    save_bootstrap_summary(nt_df, args.output_dir, n_boot=args.bootstrap_iterations, seed=args.seed)

    if args.escape_file is not None and args.escape_file.exists():
        run_optional_escape_modules(aa_df, args.escape_file, args.output_dir, args.pseudocount)

    if args.evolution_file is not None and args.evolution_file.exists():
        run_optional_evolution_modules(aa_df, args.evolution_file, args.output_dir, args.pseudocount)

    figure_aliases = {
        "figure1_mutation_supply_distributions.png": "module1_global_distributions.png",
        "figure2_aa_supply_heatmap_accessibility.png": "module3_aa_supply_heatmap.png",
        "figure3_escape_supply.png": "module6_escape_supply_distribution.png",
        "figure4_supply_vs_evolution.png": "module9_10_supply_vs_diversity_evolution.png",
    }
    for alias, src_name in figure_aliases.items():
        src = args.output_dir / src_name
        if src.exists():
            (args.output_dir / alias).write_bytes(src.read_bytes())

    has_site_ci = "bio_rate_lo95" in raw.columns and "bio_rate_hi95" in raw.columns
    run_meta = {
        "model_version": args.model_version,
        "final_dataset": str(args.final_dataset),
        "n_site_rows": len(nt_df),
        "n_positions_nt": int(nt_df["position"].nunique()),
        "n_codons_inferred": len(codon_df),
        "n_proteins_annotated": int(codon_df["protein"].nunique()) if "protein" in codon_df.columns else 1,
        "codon_annotation_mode": codon_mode,
        "n_aa_paths": len(path_df),
        "n_aa_changes": len(aa_df),
        "within_aa_variance": float(var_summary["within_aa_variance"].iloc[0]),
        "between_aa_variance": float(var_summary["between_aa_variance"].iloc[0]),
        "site_level_bio_rate_ci_used": has_site_ci,
        "aa_ci_columns": "mu_aa_lo95,mu_aa_hi95,log10_mu_aa_lo95,log10_mu_aa_hi95",
    }
    pd.DataFrame([run_meta]).to_csv(args.output_dir / "analysis_run_metadata.csv", index=False)
    print(f"Analysis complete. Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
