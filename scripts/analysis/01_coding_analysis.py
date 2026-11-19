"""
01_coding_analysis.py — Coding-region mutational effects

Maps pipeline mutation calls to GenBank CDS features and classifies each variant
as synonymous, missense, or nonsense (plus frameshift/lethal flags for indels).

Inputs (injected by Pipeline.Snakefile):
  GENBANK_FILE, MUTATIONS_CSV, REFERENCE_FA, OUTPUT_DIR

Output (under OUTPUT_DIR/):
  coding_mutations.csv      — per-mutation / per-CDS rows
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from parse_genbank import parse_genbank_file, Feature
from aa_calculator import calculate_aa_change
from frameshift_analyzer import analyze_frameshift

# Runtime configuration — injected by Pipeline.Snakefile
GENBANK_FILE = None
MUTATIONS_CSV = None
REFERENCE_FA = None
OUTPUT_DIR = None
CODING_GENES = [
    "Env", "Gag", "Nef", "Pol", "Rev", "Tat", "Vif", "Vpr", "Vpu",
]


def _validate_runtime_config() -> None:
    required = {
        "GENBANK_FILE": GENBANK_FILE,
        "MUTATIONS_CSV": MUTATIONS_CSV,
        "REFERENCE_FA": REFERENCE_FA,
        "OUTPUT_DIR": OUTPUT_DIR,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing runtime configuration from Pipeline.Snakefile: "
            + ", ".join(missing)
        )


def find_overlapping_features(position: int, features: List[Feature]) -> List[Feature]:
    return [feat for feat in features if feat.contains_position(position)]


def get_misc_feature_names(position: int, features: List[Feature]) -> List[str]:
    names = []
    for feat in features:
        if feat.feature_type == "misc_feature" and feat.contains_position(position):
            names.append(feat.name)
    return names


def analyze_mutation(genomic_sequence: str, features: List[Feature], mutation_row: pd.Series) -> Dict:
    position = int(mutation_row["position"])
    mutation_type = mutation_row["type"]
    ref_base = str(mutation_row["ref_base"])
    cons_base = str(mutation_row["cons_base"])
    sample_id = mutation_row["sample_id"]
    umi_id = mutation_row.get("umi_id", "")

    result = {
        "sample_id": sample_id,
        "position": position,
        "mutation_type": mutation_type,
        "ref_base": ref_base,
        "cons_base": cons_base,
        "umi_id": umi_id,
        "misc_features": get_misc_feature_names(position, features),
        "primary_cds": None,
        "secondary_cds": None,
        "primary_ref_aa": None,
        "primary_aa_position": None,
        "primary_cons_aa": None,
        "primary_effect": None,
        "primary_classification": None,
        "secondary_ref_aa": None,
        "secondary_aa_position": None,
        "secondary_cons_aa": None,
        "secondary_effect": None,
        "secondary_classification": None,
    }

    overlapping_cds = [
        f for f in find_overlapping_features(position, features) if f.feature_type == "CDS"
    ]

    for i, cds_feature in enumerate(overlapping_cds[:2]):
        prefix = "primary" if i == 0 else "secondary"
        cds_key = f"{prefix}_cds"
        ref_aa_key = f"{prefix}_ref_aa"
        aa_pos_key = f"{prefix}_aa_position"
        cons_aa_key = f"{prefix}_cons_aa"
        effect_key = f"{prefix}_effect"
        fs_key = f"{prefix}_frameshift"

        result[cds_key] = cds_feature.name

        aa_result = calculate_aa_change(
            genomic_sequence, cds_feature, mutation_type, position, ref_base, cons_base
        )

        if aa_result:
            result[ref_aa_key] = aa_result.get("ref_aa", "")
            codon_pos = aa_result.get("codon_position", "")
            result[aa_pos_key] = int(codon_pos) if codon_pos else ""
            result[cons_aa_key] = aa_result.get("cons_aa", "")

            base_effect = aa_result.get("effect", "")
            effects = [base_effect] if base_effect else []
            ref_aa = aa_result.get("ref_aa", "")
            cons_aa = aa_result.get("cons_aa", "")
            classification = None
            creates_stop = False

            if mutation_type in ["deletion", "insertion"]:
                fs_result = analyze_frameshift(
                    genomic_sequence, cds_feature, mutation_type, position, ref_base, cons_base
                )
                result[fs_key] = {
                    "is_frameshift": fs_result.get("is_frameshift", False),
                    "creates_stop_codon": fs_result.get("creates_stop_codon", False),
                    "is_lethal": fs_result.get("is_lethal", False),
                    "stop_codon_aa_position": fs_result.get("stop_codon_aa_position"),
                }
                creates_stop = fs_result.get("creates_stop_codon", False)
                if creates_stop:
                    stop_pos = fs_result.get("stop_codon_aa_position")
                    if stop_pos:
                        effects.append(f"introduces_premature_stop_codon_at_AA{stop_pos}")
                    else:
                        effects.append("introduces_premature_stop_codon")
                classification = "nonsense" if creates_stop else "missense"
            else:
                result[fs_key] = None
                if ref_aa == cons_aa:
                    classification = "synonymous"
                elif cons_aa == "*":
                    classification = "nonsense"
                else:
                    classification = "missense"

            result[f"{prefix}_classification"] = classification
            result[effect_key] = ";".join(effects) if effects else ""
        else:
            result[ref_aa_key] = None
            result[aa_pos_key] = None
            result[cons_aa_key] = None
            result[effect_key] = None
            result[f"{prefix}_classification"] = None
            result[fs_key] = None

    return result


def _mutation_is_lethal(classification: str, frameshift_info: Optional[dict]) -> bool:
    if (classification or "") == "nonsense":
        return True
    if frameshift_info and frameshift_info.get("is_lethal"):
        return True
    return False


def results_to_dataframe(results: List[Dict]) -> pd.DataFrame:
    rows = []
    mutation_id = 0

    for result in results:
        mutation_id += 1
        base_row = {
            "mutation_id": mutation_id,
            "sample_id": result["sample_id"],
            "position": result["position"],
            "mutation_type": result["mutation_type"],
            "ref_base": result["ref_base"],
            "cons_base": result["cons_base"],
            "umi_id": result.get("umi_id", ""),
            "misc_features": ";".join(result["misc_features"]) if result["misc_features"] else "",
        }

        for prefix, cds_key in [("primary", "primary_cds"), ("secondary", "secondary_cds")]:
            if not result.get(cds_key):
                continue
            row = base_row.copy()
            row["cds_type"] = prefix
            row["cds_name"] = result[cds_key] or ""
            row["ref_aa"] = result.get(f"{prefix}_ref_aa") or ""
            aa_pos = result.get(f"{prefix}_aa_position")
            row["aa_position"] = aa_pos if aa_pos not in [None, ""] else ""
            row["cons_aa"] = result.get(f"{prefix}_cons_aa") or ""
            row["effect"] = result.get(f"{prefix}_effect") or ""
            row["classification"] = result.get(f"{prefix}_classification") or ""
            fs = result.get(f"{prefix}_frameshift")
            if fs:
                row["frameshift"] = fs.get("is_frameshift", False)
                row["creates_stop"] = fs.get("creates_stop_codon", False)
                row["stop_aa_pos"] = fs.get("stop_codon_aa_position") or ""
            else:
                row["frameshift"] = False
                row["creates_stop"] = False
                row["stop_aa_pos"] = ""
            row["is_lethal"] = _mutation_is_lethal(row["classification"], fs)
            rows.append(row)

        if not result.get("primary_cds") and not result.get("secondary_cds"):
            row = base_row.copy()
            row.update({
                "cds_type": "noncoding",
                "cds_name": "noncoding",
                "ref_aa": "",
                "aa_position": "",
                "cons_aa": "",
                "effect": "",
                "classification": "",
                "frameshift": False,
                "creates_stop": False,
                "is_lethal": False,
                "stop_aa_pos": "",
            })
            rows.append(row)

    return pd.DataFrame(rows)


def _classification_counts(sub: pd.DataFrame) -> Dict[str, int]:
    return {
        "n_synonymous": int((sub["classification"] == "synonymous").sum()),
        "n_missense": int((sub["classification"] == "missense").sum()),
        "n_nonsense": int((sub["classification"] == "nonsense").sum()),
    }


def build_gene_classification_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Per-gene substitution classification summary (primary CDS rows only)."""
    primary = results_df[results_df["cds_type"] == "primary"]
    subs = primary[primary["mutation_type"] == "substitution"]
    rows = []
    for gene in CODING_GENES:
        sub = subs[subs["cds_name"] == gene]
        n_total = len(sub)
        counts = _classification_counts(sub) if n_total else {
            "n_synonymous": 0, "n_missense": 0, "n_nonsense": 0,
        }
        rows.append({
            "cds_name": gene,
            "n_mutations": n_total,
            **counts,
            "fraction_synonymous": round(counts["n_synonymous"] / n_total, 6) if n_total else 0.0,
            "fraction_missense": round(counts["n_missense"] / n_total, 6) if n_total else 0.0,
            "fraction_nonsense": round(counts["n_nonsense"] / n_total, 6) if n_total else 0.0,
        })
    return pd.DataFrame(rows)


def build_overall_classification_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    """Genome-wide coding-region classification summary."""
    primary = results_df[results_df["cds_type"] == "primary"]
    noncoding = results_df[results_df["cds_type"] == "noncoding"]
    scopes = {
        "primary_cds_all_types": primary,
        "primary_cds_substitutions": primary[primary["mutation_type"] == "substitution"],
        "primary_cds_indels": primary[primary["mutation_type"].isin(["insertion", "deletion"])],
    }
    rows = []
    for scope, sub in scopes.items():
        n_total = len(sub)
        counts = _classification_counts(sub) if n_total else {
            "n_synonymous": 0, "n_missense": 0, "n_nonsense": 0,
        }
        rows.append({
            "scope": scope,
            "n_rows": n_total,
            **counts,
            "n_lethal": int((sub["is_lethal"] == True).sum()) if n_total else 0,
            "fraction_synonymous": round(counts["n_synonymous"] / n_total, 6) if n_total else 0.0,
            "fraction_missense": round(counts["n_missense"] / n_total, 6) if n_total else 0.0,
            "fraction_nonsense": round(counts["n_nonsense"] / n_total, 6) if n_total else 0.0,
        })
    rows.append({
        "scope": "noncoding",
        "n_rows": len(noncoding),
        "n_synonymous": 0,
        "n_missense": 0,
        "n_nonsense": 0,
        "n_lethal": 0,
        "fraction_synonymous": 0.0,
        "fraction_missense": 0.0,
        "fraction_nonsense": 0.0,
    })
    return pd.DataFrame(rows)


def load_genomic_sequence(reference_fa: str, genbank_file: str) -> str:
    ref_path = Path(reference_fa)
    if ref_path.is_file():
        with open(ref_path) as f:
            ref_lines = f.readlines()
        if len(ref_lines) > 1:
            seq = "".join(ref_lines[1:]).strip().upper().replace("\n", "")
            print(f"  Loaded reference sequence from FASTA: {len(seq)} bp")
            return seq
    genomic_sequence, _ = parse_genbank_file(genbank_file)
    print(f"  Using sequence from GenBank: {len(genomic_sequence)} bp")
    return genomic_sequence


def annotate_umi_mutation_counts(results_df: pd.DataFrame) -> pd.DataFrame:
    umi_groups = results_df.groupby(["sample_id", "umi_id"])

    def count_unique_mutations(group):
        return group.groupby(["position", "mutation_type", "ref_base", "cons_base"]).ngroups

    umi_mutation_counts = umi_groups.apply(count_unique_mutations, include_groups=False).to_dict()
    results_df = results_df.copy()
    results_df["umi_key"] = list(zip(results_df["sample_id"], results_df["umi_id"]))
    results_df["num_umi_mutations"] = results_df["umi_key"].map(umi_mutation_counts)
    results_df = results_df.drop(columns=["umi_key"])
    mutation_counts = results_df.groupby("mutation_id").size().to_dict()
    results_df["num_cds_affected"] = results_df["mutation_id"].map(mutation_counts)
    return results_df


def main() -> None:
    _validate_runtime_config()

    gb_file = Path(GENBANK_FILE)
    mutations_file = Path(MUTATIONS_CSV)
    reference_file = Path(REFERENCE_FA)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gb_file.is_file():
        raise FileNotFoundError(f"GenBank file not found: {gb_file}")
    if not mutations_file.is_file():
        raise FileNotFoundError(f"Mutations CSV not found: {mutations_file}")

    print(f"Parsing GenBank file: {gb_file}")
    _, features = parse_genbank_file(str(gb_file))
    print(f"  Features found: {len(features)}")

    genomic_sequence = load_genomic_sequence(str(reference_file), str(gb_file))

    print(f"\nLoading mutations from: {mutations_file}")
    mutations_df = pd.read_csv(mutations_file)
    print(f"  Total mutations: {len(mutations_df)}")

    print("\nAnalyzing mutations...")
    results = []
    for idx, row in mutations_df.iterrows():
        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(mutations_df)} mutations...")
        try:
            results.append(analyze_mutation(genomic_sequence, features, row))
        except Exception as exc:
            print(f"  Error processing mutation at position {row.get('position', 'unknown')}: {exc}")

    print(f"\nCompleted analysis of {len(results)} mutations")
    results_df = annotate_umi_mutation_counts(results_to_dataframe(results))

    if "aa_position" in results_df.columns:
        results_df["aa_position"] = results_df["aa_position"].apply(
            lambda x: int(x) if pd.notna(x) and x != "" else ""
        )

    csv_output = output_dir / "coding_mutations.csv"

    results_df.to_csv(csv_output, index=False)

    print(f"\nSaved CSV output: {csv_output}")

    primary_cds = results_df[results_df["cds_type"] == "primary"]
    print("\n=== Summary Statistics ===")
    print(f"Total mutation events analyzed: {len(results)}")
    print(f"Total rows in output: {len(results_df)}")
    print(f"Primary CDS rows: {len(primary_cds)}")
    print(f"Non-coding rows: {len(results_df[results_df['cds_type'] == 'noncoding'])}")
    if len(primary_cds) > 0:
        print("Classification summary (primary CDS, all types):")
        for label in ["synonymous", "missense", "nonsense"]:
            print(f"  {label}: {(primary_cds['classification'] == label).sum()}")
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
