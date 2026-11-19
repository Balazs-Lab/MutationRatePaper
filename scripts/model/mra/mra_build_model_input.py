#!/usr/bin/env python3
from __future__ import annotations

"""
Build the mra model input (`mutation_supply_data.csv`) directly from the
pipeline outputs in ``data/pipeline``.

This is the single place where per-REC quality filtering happens. By default we
keep RECs (consensus reads, identified by ``sample_id`` + UMI barcode) with
**at most 1000 total mutations per REC** (``<= 1000``, matching R11). RECs above
that threshold are dropped from *both* the mutation numerator and the coverage
denominator so the per-site rates stay internally consistent.

Library amplicon ends are trimmed by ``EDGE_TRIM`` positions (default 20 bp per end),
matching the R1/R11 data-prep rule applied to both coverage mapping and mutations.

Inputs (under ``data/pipeline``):
  - 10-mutation_analysis_.../all_mutations.csv         (one row per called mutation)
  - 10-mutation_analysis_.../consensus_sequence_data.csv (one row per REC, has Total_Mutations)
  - 00-reference-map/LP###.csv                         (library <-> reference position maps;
                                                        ref_pos/ref_base define the reference)
  - 00-reference-map/library_to_biological_reference.yaml

Output:
  - data/models/mra/mutation_supply_data.csv
  - data/models/mra/mutation_supply_data.build_metadata.json

The output schema matches what the mra production model expects:
  ref_position, Mut_Type, WT_mutation_count, Control_mutation_count,
  WT_coverage, Control_coverage, ref_nucleotide, five_prime, three_prime
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mra_run import RunConfig

# =============================================================================
# CONFIG
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mutation_supply_data.csv for the mra model.")
    parser.add_argument(
        "--run",
        type=Path,
        required=True,
        help="Path to mra_run_test.csv or mra_run_production.csv",
    )
    return parser.parse_args()


ARGS = parse_args()
PROJECT_ROOT = Path(__file__).resolve().parents[3]
PIPELINE_DIR = Path(os.environ.get("MRA_PIPELINE_DIR", PROJECT_ROOT / "data" / "pipeline"))

MUT_ANALYSIS_DIR = Path(os.environ.get(
    "MRA_MUT_ANALYSIS_DIR",
    PIPELINE_DIR / "10-mutation_analysis_ERRSeq-PAPER-SAMPLES",
))
ALL_MUTATIONS_PATH = Path(os.environ.get(
    "MRA_ALL_MUTATIONS_PATH", MUT_ANALYSIS_DIR / "all_mutations.csv"
))
CONSENSUS_SEQ_PATH = Path(os.environ.get(
    "MRA_CONSENSUS_SEQ_PATH", MUT_ANALYSIS_DIR / "consensus_sequence_data.csv"
))

REFERENCE_MAP_DIR = Path(os.environ.get(
    "MRA_REFERENCE_MAP_DIR", PIPELINE_DIR / "00-reference-map"
))
LIBRARY_REF_YAML = Path(os.environ.get(
    "MRA_LIBRARY_REF_YAML", REFERENCE_MAP_DIR / "library_to_biological_reference.yaml"
))
REFERENCE_STRAIN = os.environ.get("MRA_REFERENCE_STRAIN", "NL43")

OUTPUT_PATH = Path(os.environ.get(
    "MRA_MODEL_INPUT_PATH",
    PROJECT_ROOT / "data" / "models" / "mra" / "mutation_supply_data.csv",
))
RUN_PATH = ARGS.run.expanduser().resolve()
run = RunConfig(RUN_PATH)

# Keep RECs with at most this many total mutations (per REC); R11 used <= 1000.
MAX_MUTS_PER_REC = run.get("max_muts_per_rec")
# Trim this many library positions from each end of every library amplicon (R1/R11 default).
EDGE_TRIM = run.get("edge_trim")
# Flanking context length (bases) captured in five_prime / three_prime.
CONTEXT_K = run.get("context_k")

# Sample-name group token (3rd hyphen field, e.g. NL43-LP253-WT-1-N -> "WT").
# NEG / Sparsomycin / pUC19 are WT-equivalent (no resistance selection) and are
# pooled into the WT class. K65R / M184I / M184V are true variants and excluded.
WT_GROUPS = frozenset(run.get("wt_groups"))
CONTROL_GROUPS = frozenset(run.get("control_groups"))

BASES = ("A", "C", "G", "T")
MUT_TYPES_BY_REF = {
    "A": ["A->C", "A->G", "A->T", "deletion", "insertion"],
    "C": ["C->A", "C->G", "C->T", "deletion", "insertion"],
    "G": ["G->A", "G->C", "G->T", "deletion", "insertion"],
    "T": ["T->A", "T->C", "T->G", "deletion", "insertion"],
}


# =============================================================================
# HELPERS
# =============================================================================


def build_reference_base_map(libraries: list[str]) -> dict[int, str]:
    """Reference sequence keyed by ref_pos, taken from the reference-map CSVs.

    The reference-map ``ref_pos``/``ref_base`` columns define the exact coordinate
    system used by all_mutations (``ref_position``) and the coverage maps. We build
    the reference sequence from these columns rather than from a FASTA file, because
    the FASTA in 00-reference-map is not guaranteed to share this coordinate frame.
    All target libraries must agree on the base at each shared position.
    """
    combined: dict[int, str] = {}
    for lib in libraries:
        ref_map = pd.read_csv(REFERENCE_MAP_DIR / f"{lib}.csv")
        if not {"ref_pos", "ref_base"}.issubset(ref_map.columns):
            raise ValueError(f"{lib}.csv must contain ref_pos and ref_base")
        sub = ref_map.dropna(subset=["ref_base"])
        for pos, b in zip(sub["ref_pos"].astype(int), sub["ref_base"].astype(str).str.upper()):
            if pos in combined and combined[pos] != b:
                raise ValueError(
                    f"Reference base conflict at ref_pos {pos}: "
                    f"{combined[pos]!r} vs {b!r} (from {lib})"
                )
            combined[pos] = b
    if not combined:
        raise ValueError("No reference bases found in reference-map CSVs.")
    return combined


def library_from_sample(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    return parts[1] if len(parts) > 1 else ""


def group_from_sample(sample_id: str) -> str:
    parts = str(sample_id).split("-")
    return parts[2] if len(parts) > 2 else ""


def classify_group(group: str) -> str | float:
    if group in WT_GROUPS:
        return "WT"
    if group in CONTROL_GROUPS:
        return "Control"
    return np.nan


def reference_libraries() -> list[str]:
    """Libraries mapped to REFERENCE_STRAIN that also have a reference-map CSV."""
    with open(LIBRARY_REF_YAML) as fh:
        mapping = yaml.safe_load(fh) or {}
    libs = [lib for lib, strain in mapping.items() if str(strain) == REFERENCE_STRAIN]
    present = [lib for lib in libs if (REFERENCE_MAP_DIR / f"{lib}.csv").exists()]
    missing_map = sorted(set(libs) - set(present))
    if missing_map:
        print(f"  Note: {REFERENCE_STRAIN} libraries without a reference map (skipped): {missing_map}")
    if not present:
        raise ValueError(
            f"No reference-map CSVs found for {REFERENCE_STRAIN} libraries in {REFERENCE_MAP_DIR}"
        )
    return sorted(present)


def end_trim_positions(positions: pd.Series, trim: int) -> pd.Series:
    """Boolean mask keeping library positions strictly inside the trimmed amplicon ends.

    Matches the R1/R11 rule: keep positions where
    ``trim < pos < (max(pos) - trim)`` within each library amplicon.
    """
    if trim <= 0:
        return pd.Series(True, index=positions.index)
    pos = pd.to_numeric(positions, errors="coerce")
    max_pos = pos.max()
    if pd.isna(max_pos):
        return pd.Series(False, index=positions.index)
    return (pos > trim) & (pos < (max_pos - trim))


def apply_end_trim(
    df: pd.DataFrame,
    pos_col: str,
    library_col: str | None,
    trim: int,
) -> pd.DataFrame:
    """Drop rows at the first/last ``trim`` library positions (per library if grouped)."""
    if trim <= 0:
        return df
    out = df.copy()
    out[pos_col] = pd.to_numeric(out[pos_col], errors="coerce")
    out = out.loc[out[pos_col].notna()].copy()
    if library_col is None:
        keep = end_trim_positions(out[pos_col], trim)
    else:
        max_pos = out.groupby(library_col)[pos_col].transform("max")
        keep = (out[pos_col] > trim) & (out[pos_col] < (max_pos - trim))
    return out.loc[keep].copy()


# =============================================================================
# STEP 1 — reference context table (every position x possible mutation)
# =============================================================================


def build_context_table(ref_base_map: dict[int, str], k: int) -> pd.DataFrame:
    """Every reference position x its possible mutation types, with k-mer context.

    ``five_prime`` / ``three_prime`` are the k upstream / downstream reference bases
    (NaN when any flanking base is missing, e.g. at the reference ends).
    """
    positions = sorted(ref_base_map)

    def flank(pos: int, offsets: range) -> float | str:
        bases = []
        for off in offsets:
            b = ref_base_map.get(pos + off)
            if b is None:
                return np.nan
            bases.append(b)
        return "".join(bases)

    rows = {
        "position": positions,
        "nucleotide": [ref_base_map[p] for p in positions],
        "five_prime": [flank(p, range(-k, 0)) for p in positions],
        "three_prime": [flank(p, range(1, k + 1)) for p in positions],
    }
    base = pd.DataFrame(rows)
    base = base[base["nucleotide"].isin(BASES)].copy()

    base["mut_type"] = base["nucleotide"].map(MUT_TYPES_BY_REF)
    ctx = base.explode("mut_type", ignore_index=True)
    return ctx


# =============================================================================
# STEP 2 — coverage per reference position (REC-filtered)
# =============================================================================


def load_rec_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rec_id_col = "UMI_ID" if "UMI_ID" in df.columns else ("REC_ID" if "REC_ID" in df.columns else None)
    if "Sample" not in df.columns or rec_id_col is None or "Total_Mutations" not in df.columns:
        raise ValueError(
            f"{path} must contain Sample, UMI_ID/REC_ID and Total_Mutations columns; got {list(df.columns)}"
        )
    df = df.rename(columns={rec_id_col: "REC_ID"})
    df["library"] = df["Sample"].map(library_from_sample)
    df["group"] = df["Sample"].map(group_from_sample)
    df["class"] = df["group"].map(classify_group)
    df["Total_Mutations"] = pd.to_numeric(df["Total_Mutations"], errors="coerce")
    return df


def compute_coverage(rec_df: pd.DataFrame, libraries: list[str]) -> tuple[dict, dict]:
    keep = rec_df.loc[
        rec_df["library"].isin(libraries)
        & rec_df["class"].isin(["WT", "Control"])
        & rec_df["Total_Mutations"].notna()
    ].copy()
    total_recs = len(keep)
    kept = keep.loc[keep["Total_Mutations"] <= MAX_MUTS_PER_REC].copy()
    cov = kept.groupby(["library", "class"]).size()
    cov_map = {(lib, cls): int(n) for (lib, cls), n in cov.items()}
    stats = {
        "recs_considered": int(total_recs),
        "recs_kept": int(len(kept)),
        "recs_dropped": int(total_recs - len(kept)),
    }
    return cov_map, stats


def build_position_coverage(cov_map: dict, libraries: list[str]) -> pd.DataFrame:
    frames = []
    for lib in libraries:
        ref_map = pd.read_csv(REFERENCE_MAP_DIR / f"{lib}.csv")
        pos_col, base_col = f"{lib}_pos", f"{lib}_base"
        if not {"ref_pos", "ref_base", pos_col, base_col}.issubset(ref_map.columns):
            raise ValueError(f"{lib}.csv must contain ref_pos, ref_base, {pos_col}, {base_col}")
        m = ref_map.loc[ref_map["ref_base"] == ref_map[base_col]].copy()
        m[pos_col] = pd.to_numeric(m[pos_col], errors="coerce")
        m = m.loc[m[pos_col].notna()]
        m = apply_end_trim(m, pos_col, library_col=None, trim=EDGE_TRIM)
        m = m.assign(
            WT=cov_map.get((lib, "WT"), 0),
            Control=cov_map.get((lib, "Control"), 0),
        )
        frames.append(m[["ref_pos", "WT", "Control"]])
    coverage = (
        pd.concat(frames, ignore_index=True)
        .groupby("ref_pos", as_index=False)[["WT", "Control"]]
        .sum()
    )
    return coverage


# =============================================================================
# STEP 3 — mutation counts per reference position x type (REC-filtered)
# =============================================================================


def load_mutations(path: Path, libraries: list[str], kept_recs: set[tuple[str, str]]) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = ["ref_position", "type", "ref_base", "cons_base", "umi_id", "sample_id", "library_position"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")

    if "library_name" in df.columns:
        df["library"] = df["library_name"].astype(str)
    else:
        df["library"] = df["sample_id"].map(library_from_sample)
    df["group"] = df["sample_id"].map(group_from_sample)
    df["class"] = df["group"].map(classify_group)

    df = df.loc[df["library"].isin(libraries) & df["class"].isin(["WT", "Control"])].copy()

    # REC filter: keep only mutations belonging to kept RECs.
    rec_key = list(zip(df["sample_id"].astype(str), df["umi_id"].astype(str)))
    df = df.loc[[k in kept_recs for k in rec_key]].copy()

    # Edge trim by library position (per library amplicon; matches R1/R11).
    df = apply_end_trim(df, "library_position", library_col="library", trim=EDGE_TRIM)

    # Mutation type label.
    df["mut_type"] = df["ref_base"].astype(str) + "->" + df["cons_base"].astype(str)
    df.loc[df["type"] == "deletion", "mut_type"] = "deletion"
    df.loc[df["type"] == "insertion", "mut_type"] = "insertion"
    return df


def counts_per_position(mut_df: pd.DataFrame) -> pd.DataFrame:
    counts = (
        mut_df.groupby(["ref_position", "class", "mut_type"]).size().reset_index(name="n")
    )
    wide = counts.pivot_table(
        index=["ref_position", "mut_type"],
        columns="class",
        values="n",
        fill_value=0,
    ).reset_index()
    for cls in ("WT", "Control"):
        if cls not in wide.columns:
            wide[cls] = 0
    return wide[["ref_position", "mut_type", "WT", "Control"]]


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    print("=" * 70)
    print("mra — build model input (with per-REC mutation filter)")
    print("=" * 70)
    print(f"Reference strain      : {REFERENCE_STRAIN}")
    print(f"Keep RECs with        : Total_Mutations <= {MAX_MUTS_PER_REC}")
    print(f"Edge trim (bp/end)    : {EDGE_TRIM}")
    print(f"WT groups             : {sorted(WT_GROUPS)}")
    print(f"Control groups        : {sorted(CONTROL_GROUPS)}")
    print(f"all_mutations.csv     : {ALL_MUTATIONS_PATH}")
    print(f"consensus_seq_data.csv: {CONSENSUS_SEQ_PATH}")

    libraries = reference_libraries()
    print(f"{REFERENCE_STRAIN} libraries        : {libraries}")

    ref_base_map = build_reference_base_map(libraries)
    print(f"Reference positions   : {len(ref_base_map)} (from reference-map ref_pos/ref_base)")
    ctx = build_context_table(ref_base_map, CONTEXT_K)

    rec_df = load_rec_table(CONSENSUS_SEQ_PATH)
    cov_map, rec_stats = compute_coverage(rec_df, libraries)
    print(
        f"RECs                  : kept {rec_stats['recs_kept']:,} / "
        f"{rec_stats['recs_considered']:,} "
        f"(dropped {rec_stats['recs_dropped']:,} with > {MAX_MUTS_PER_REC} mutations)"
    )
    coverage = build_position_coverage(cov_map, libraries)
    print(f"Covered ref positions : {len(coverage):,}")

    kept_mask = (
        rec_df["library"].isin(libraries)
        & rec_df["class"].isin(["WT", "Control"])
        & rec_df["Total_Mutations"].notna()
        & (rec_df["Total_Mutations"] <= MAX_MUTS_PER_REC)
    )
    kept_recs = set(zip(rec_df.loc[kept_mask, "Sample"].astype(str),
                        rec_df.loc[kept_mask, "REC_ID"].astype(str)))

    mut_df = load_mutations(ALL_MUTATIONS_PATH, libraries, kept_recs)
    print(f"Mutations (post-filter): {len(mut_df):,}")
    counts = counts_per_position(mut_df)

    # Merge context (all positions x types) with coverage, then counts.
    ctx_cov = ctx.merge(coverage, left_on="position", right_on="ref_pos", how="inner")
    supply = ctx_cov.merge(
        counts,
        left_on=["position", "mut_type"],
        right_on=["ref_position", "mut_type"],
        how="left",
        suffixes=("_cov", "_mut"),
    )

    out = pd.DataFrame({
        "ref_position": supply["position"].astype(int),
        "Mut_Type": supply["mut_type"],
        "WT_mutation_count": supply["WT_mut"].fillna(0).astype(int),
        "Control_mutation_count": supply["Control_mut"].fillna(0).astype(int),
        "WT_coverage": supply["WT_cov"].fillna(0).astype(int),
        "Control_coverage": supply["Control_cov"].fillna(0).astype(int),
        "ref_nucleotide": supply["nucleotide"],
        "five_prime": supply["five_prime"],
        "three_prime": supply["three_prime"],
    })
    out = out.sort_values(["ref_position", "Mut_Type"]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)

    metadata = {
        "run_file": str(RUN_PATH),
        "reference_strain": REFERENCE_STRAIN,
        "libraries": libraries,
        "max_muts_per_rec_inclusive": MAX_MUTS_PER_REC,
        "rec_filter_rule": f"keep RECs with Total_Mutations <= {MAX_MUTS_PER_REC}",
        "edge_trim": EDGE_TRIM,
        "context_k": CONTEXT_K,
        "wt_groups": sorted(WT_GROUPS),
        "control_groups": sorted(CONTROL_GROUPS),
        "inputs": {
            "all_mutations": str(ALL_MUTATIONS_PATH),
            "consensus_sequence_data": str(CONSENSUS_SEQ_PATH),
            "reference_map_dir": str(REFERENCE_MAP_DIR),
            "reference_source": "reference-map ref_pos/ref_base columns",
        },
        "rec_stats": rec_stats,
        "n_mutations_post_filter": int(len(mut_df)),
        "n_covered_positions": int(len(coverage)),
        "n_output_rows": int(len(out)),
        "output": str(OUTPUT_PATH),
    }
    meta_path = OUTPUT_PATH.with_suffix(".build_metadata.json")
    with open(meta_path, "w") as fh:
        json.dump(metadata, fh, indent=2)

    print(f"\nWrote model input     : {OUTPUT_PATH} ({len(out):,} rows)")
    print(f"Wrote build metadata  : {meta_path}")
    print("\nMut_Type row counts:")
    print(out["Mut_Type"].value_counts())


if __name__ == "__main__":
    main()
