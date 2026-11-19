#!/usr/bin/env python3
"""
Build the mra variant-supply table from the comprehensive AA-supply output.

Converts the raw per-polyprotein mra amino-acid supply
(`aa_level_supply_table.csv`, keyed by Gag/Pol polyprotein positions) into a
mature-protein variant-supply table (Pro, RT, IN, p24) with positions renumbered
to each mature protein's own coordinate system. This is the programmatic version
of the previously manual `variant_supply.xlsx`.

Mature-protein boundaries are derived directly from the NL4-3 GenBank annotation
(`NL43.gb`) mature-peptide (`misc_feature`) coordinates, mapped onto the parent
CDS reading frame. This guarantees canonical numbering, e.g.:
  - Protease N-terminus  = Pol 52  (PQITLWQRP...)
  - RT N-terminus        = Pol 151 (PISPIETVPV...)
  - Integrase N-terminus = Pol 711 (FLDGIDKA...)
  - p24 (capsid)         = Gag 133

Output columns mirror the reference workbook:
  protein, Polposition, position, aa_change, mutation, ref_aa, alt_aa,
  synonymous, mu_aa, mu_aa_lo95, mu_aa_hi95, min_nt_changes, n_paths,
  accessible_class, log10_mu_aa, log10_mu_aa_lo95, log10_mu_aa_hi95

where `Polposition` is the original parent-polyprotein AA position and
`position` is the mature-protein position (1-based).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

try:
    from Bio import SeqIO
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Biopython is required (pip install biopython). Import failed: %s" % exc
    )

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_AA_SUPPLY = (
    PROJECT_ROOT
    / "data/models/mra/production/comprehensive_analysis_mra_product/aa_level_supply_table.csv"
)
DEFAULT_GENBANK = PROJECT_ROOT / "data/pipeline/00-bed_files/NL43.gb"
DEFAULT_OUT = PROJECT_ROOT / "data/mutation_supply/mra_variant_supply.csv"
DEFAULT_REFERENCE_XLSX = PROJECT_ROOT / "data/mutation_supply/variant_supply.xlsx"

# Mature proteins to extract. Each is defined by its parent CDS (as named in the
# aa_level_supply_table `protein` column and the GenBank CDS standard_name) and a
# list of GenBank mature-peptide feature name substrings that compose it. When
# multiple features are listed they are merged into one contiguous mature protein
# (e.g. RT p66 = RT polymerase domain + RNase H domain).
MATURE_PROTEIN_MAP: List[Dict[str, object]] = [
    {"name": "Pro", "parent": "Pol", "features": ["Protease"]},
    {"name": "RT", "parent": "Pol", "features": ["Reverse Transcriptase", "RNase"]},
    {"name": "IN", "parent": "Pol", "features": ["Integrase"]},
    {"name": "p24", "parent": "Gag", "features": ["p24"]},
]

OUTPUT_COLUMNS = [
    "protein",
    "Polposition",
    "position",
    "aa_change",
    "mutation",
    "ref_aa",
    "alt_aa",
    "synonymous",
    "mu_aa",
    "mu_aa_lo95",
    "mu_aa_hi95",
    "min_nt_changes",
    "n_paths",
    "accessible_class",
    "log10_mu_aa",
    "log10_mu_aa_lo95",
    "log10_mu_aa_hi95",
]


def _feature_name(feat) -> str:
    for key in ("standard_name", "gene", "product", "label", "note"):
        if key in feat.qualifiers:
            return str(feat.qualifiers[key][0])
    return ""


def parse_genbank_features(genbank_path: Path) -> Tuple[Dict[str, Tuple[int, int]], List[Tuple[str, int, int]]]:
    """Return (CDS map name->(gstart,gend), list of mature (name,gstart,gend)), 1-based inclusive."""
    record = SeqIO.read(str(genbank_path), "genbank")
    cds: Dict[str, Tuple[int, int]] = {}
    mature: List[Tuple[str, int, int]] = []
    for feat in record.features:
        name = _feature_name(feat)
        gstart = int(feat.location.start) + 1
        gend = int(feat.location.end)
        if feat.type == "CDS":
            # Keep the first (widest) definition for a given standard_name.
            cds.setdefault(name, (gstart, gend))
        elif feat.type == "misc_feature":
            mature.append((name, gstart, gend))
    return cds, mature


def genomic_to_parent_aa(genomic_pos: int, cds_start: int) -> int:
    """1-based AA position within a CDS for a 1-based genomic position."""
    return (genomic_pos - cds_start) // 3 + 1


def resolve_mature_ranges(
    cds: Dict[str, Tuple[int, int]],
    mature: List[Tuple[str, int, int]],
) -> List[Dict[str, object]]:
    """Compute parent-AA start/end for each configured mature protein."""
    resolved: List[Dict[str, object]] = []
    for spec in MATURE_PROTEIN_MAP:
        parent = str(spec["parent"])
        if parent not in cds:
            raise ValueError(f"Parent CDS '{parent}' not found in GenBank CDS features: {sorted(cds)}")
        cds_start = cds[parent][0]

        matched: List[Tuple[str, int, int]] = []
        for sub in spec["features"]:  # type: ignore[union-attr]
            hits = [m for m in mature if sub.lower() in m[0].lower()]
            if not hits:
                raise ValueError(
                    f"Mature feature matching '{sub}' not found for {spec['name']}. "
                    f"Available: {[m[0] for m in mature]}"
                )
            if len(hits) > 1:
                raise ValueError(f"Ambiguous mature feature match for '{sub}': {[h[0] for h in hits]}")
            matched.append(hits[0])

        gstart = min(m[1] for m in matched)
        gend = max(m[2] for m in matched)
        aa_start = genomic_to_parent_aa(gstart, cds_start)
        aa_end = genomic_to_parent_aa(gend, cds_start)
        resolved.append(
            {
                "name": spec["name"],
                "parent": parent,
                "aa_start": aa_start,
                "aa_end": aa_end,
                "source_features": [m[0] for m in matched],
                "genomic": (gstart, gend),
            }
        )
    return resolved


def build_variant_supply(aa_df: pd.DataFrame, resolved: List[Dict[str, object]]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for r in resolved:
        parent = r["parent"]
        aa_start = int(r["aa_start"])  # type: ignore[arg-type]
        aa_end = int(r["aa_end"])  # type: ignore[arg-type]
        sub = aa_df[(aa_df["protein"] == parent) & (aa_df["position"].between(aa_start, aa_end))].copy()
        if sub.empty:
            raise ValueError(
                f"No {parent} rows in [{aa_start},{aa_end}] for output protein {r['name']}."
            )
        sub["Polposition"] = sub["position"].astype(int)
        sub["position"] = sub["position"].astype(int) - aa_start + 1
        sub["protein"] = r["name"]
        frames.append(sub)

    out = pd.concat(frames, ignore_index=True)
    for col in OUTPUT_COLUMNS:
        if col not in out.columns:
            raise ValueError(f"Expected column '{col}' missing from AA supply input.")
    out = out[OUTPUT_COLUMNS]
    out = out.sort_values(["protein", "position", "alt_aa"]).reset_index(drop=True)
    return out


def verify_against_reference(out: pd.DataFrame, reference_xlsx: Path) -> None:
    ref = pd.read_excel(reference_xlsx, sheet_name="Sheet1")
    print("\n=== Verification against reference workbook ===")
    print(f"reference: {reference_xlsx}")
    print(f"rows: generated={len(out)}  reference={len(ref)}")

    key = ["protein", "position", "aa_change"]
    for prot in sorted(out["protein"].unique()):
        g = out[out["protein"] == prot]
        r = ref[ref["protein"] == prot]
        g_rng = (int(g["Polposition"].min()), int(g["Polposition"].max()))
        r_rng = (int(r["Polposition"].min()), int(r["Polposition"].max())) if len(r) else (None, None)
        status = "MATCH" if g_rng == r_rng else "DIFF"
        print(
            f"  {prot:4s} generated Polpos {g_rng}  reference {r_rng}  [{status}]  "
            f"(pos 1..{int(g['position'].max())}, n={len(g)})"
        )

    merged = out.merge(ref, on=key, how="outer", suffixes=("_gen", "_ref"), indicator=True)
    only_gen = int((merged["_merge"] == "left_only").sum())
    only_ref = int((merged["_merge"] == "right_only").sum())
    both = merged[merged["_merge"] == "both"]
    print(f"  keys only in generated: {only_gen}   only in reference: {only_ref}   shared: {len(both)}")
    if len(both):
        import numpy as np

        mu_close = np.isclose(
            both["mu_aa_gen"].astype(float), both["mu_aa_ref"].astype(float), rtol=1e-6, atol=0
        ).mean()
        print(f"  mu_aa agreement on shared keys: {mu_close*100:.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--aa-supply", type=Path, default=DEFAULT_AA_SUPPLY, help="aa_level_supply_table.csv")
    parser.add_argument("--genbank", type=Path, default=DEFAULT_GENBANK, help="NL43.gb annotation")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output CSV path")
    parser.add_argument(
        "--reference-xlsx",
        type=Path,
        default=DEFAULT_REFERENCE_XLSX,
        help="Reference workbook for position verification (skip with --no-verify)",
    )
    parser.add_argument("--no-verify", action="store_true", help="Skip verification against the reference workbook")
    args = parser.parse_args()

    if not args.aa_supply.exists():
        parser.error(f"AA supply table not found: {args.aa_supply}")
    if not args.genbank.exists():
        parser.error(f"GenBank annotation not found: {args.genbank}")

    cds, mature = parse_genbank_features(args.genbank)
    resolved = resolve_mature_ranges(cds, mature)

    print("Mature-protein boundaries (from GenBank):")
    for r in resolved:
        print(
            f"  {r['name']:4s} <- {r['parent']} AA {r['aa_start']}..{r['aa_end']}  "
            f"(genomic {r['genomic'][0]}..{r['genomic'][1]}; features={r['source_features']})"
        )

    aa_df = pd.read_csv(args.aa_supply)
    out = build_variant_supply(aa_df, resolved)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} rows across {out['protein'].nunique()} mature proteins -> {args.out}")

    if not args.no_verify and args.reference_xlsx.exists():
        verify_against_reference(out, args.reference_xlsx)


if __name__ == "__main__":
    main()
