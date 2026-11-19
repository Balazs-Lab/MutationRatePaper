"""
00_prepare_data.py — REC Analysis Data Preparation

Loads raw pipeline outputs and produces consensus_sequence_data.csv, which is
the single shared input consumed by every downstream figure script
(01_figure_1d.py, 02_figure_1e.py, …).

Key processing steps:
  1. Load NL43 nucleotide-context reference, per-REC mutation counts, and the
     mutation-specific JSON report.
  2. Count all 12 substitution types per REC.
  3. Assign actual per-REC sequence lengths from compiled consensus FASTAs
     (falls back to the reference length when a FASTA is missing).
  4. Map samples to their group name using the sample-groups YAML.
  5. Filter to the requested INCLUDE_GROUPS.
  6. Write consensus_sequence_data.csv.

Outputs
-------
  OUTPUT_DIR/consensus_sequence_data.csv
"""

import json
import os
import re
import sys
import yaml
import pandas as pd
from typing import List, Dict, Optional

# =============================================================================
# CONFIGURATION — determined by the config file used by snakefiles/Analysis.Snakefile
# =============================================================================

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from sample_ids import expand_sample_ids_for_matching_tables, pipeline_output_sample_id

PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DATA_DIR     = os.path.join(PROJECT_ROOT, 'data')

# Runtime-configured inputs/outputs. These are injected by
# snakefiles/Analysis.Snakefile from the selected analysis config (e.g.
# configs/analysis/ERRSeq-Paper/Observational-Analyses.yaml).
CONSENSUS_SEQUENCE_DATA_CSV = None
MUTATION_COUNT_DATA_CSV = None
MUTATION_SPECIFIC_DATA_CSV = None
SAMPLE_GROUPS_CONFIG_YAML = None
COMPILED_CONSENSUS_DIR = None
INCLUDE_GROUPS = None
OUTPUT_DIR = None


def _validate_runtime_config() -> None:
    """Fail fast if required runtime config is not injected by Snakefile."""
    required = {
        "CONSENSUS_SEQUENCE_DATA_CSV": CONSENSUS_SEQUENCE_DATA_CSV,
        "MUTATION_COUNT_DATA_CSV": MUTATION_COUNT_DATA_CSV,
        "MUTATION_SPECIFIC_DATA_CSV": MUTATION_SPECIFIC_DATA_CSV,
        "SAMPLE_GROUPS_CONFIG_YAML": SAMPLE_GROUPS_CONFIG_YAML,
        "COMPILED_CONSENSUS_DIR": COMPILED_CONSENSUS_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise RuntimeError(
            "Missing runtime configuration from snakefiles/Analysis.Snakefile: "
            + ", ".join(missing)
            + ". Use snakemake with snakefiles/Analysis.Snakefile and an analysis config."
        )

# =============================================================================
# DATA LOADING
# =============================================================================


def load_consensus_sequence_data(file_path):
    """Load NL43 nucleotide-context reference CSV.

    Returns DataFrame with columns: position, nucleotide, downstream_nucleotide,
    dinucleotide_context.
    """
    df = pd.read_csv(file_path)
    df['dinucleotide_context'] = df['nucleotide'] + df['downstream_nucleotide']
    return df


def load_mutation_count_data(file_path):
    """Load per-REC mutation count CSV."""
    return pd.read_csv(file_path)


def load_mutation_specific_data(file_path):
    """Load and flatten the mutation-specific JSON report.

    Uses mutations_by_position only (not per_read_analysis).
    Returns a flat DataFrame of all mutations across all samples.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        content = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', content)
        report_data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error loading JSON: {e}. Attempting aggressive cleaning…")
        content = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', content)
        report_data = json.loads(content)

    all_mutations = []
    for sample_result in report_data['results']:
        sample_name = sample_result['sample_name']
        if 'mutations_by_position' in sample_result:
            for mutation in sample_result['mutations_by_position']:
                mutation['sample_name'] = sample_name
                all_mutations.append(mutation)
        else:
            print(f"Warning: No mutations_by_position for {sample_name}")

    return pd.DataFrame(all_mutations)


def load_sample_groups_config(file_path):
    """Load sample-groups YAML. Returns list of {name, samples} dicts."""
    with open(file_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg.get('sample_groups', [])


def get_sample_group(sample_id: str, groups: List[Dict]) -> Optional[str]:
    """Return the group name for sample_id, preferring specific groups over
    the catch-all 'All_conditions'."""
    for group in groups:
        if group.get('name') == 'All_conditions':
            continue
        for sid in group.get('samples', []):
            if sample_id == sid or sample_id == pipeline_output_sample_id(sid):
                return group['name']
    for group in groups:
        for sid in group.get('samples', []):
            if sample_id == sid or sample_id == pipeline_output_sample_id(sid):
                return group['name']
    return None


def get_samples_in_groups(group_names: List[str], groups: List[Dict]) -> set:
    """Return sample IDs to match pipeline tables (NL43-LP###-… in mutation_analysis)."""
    wanted = set(group_names)
    raw: list = []
    for g in groups:
        if g.get('name') in wanted:
            raw.extend(g.get('samples', []))
    return set(expand_sample_ids_for_matching_tables(raw))


# =============================================================================
# SUBSTITUTION COUNTING
# =============================================================================

_ALL_SUB_TYPES = [
    'A->C', 'A->G', 'A->T',
    'C->A', 'C->G', 'C->T',
    'G->A', 'G->C', 'G->T',
    'T->A', 'T->C', 'T->G',
]


def count_substitution_types_per_umi(mutation_specific_data):
    """Count all 12 substitution types per (sample, UMI) pair.

    Returns a pivot DataFrame with one row per (sample_name, umi_id) and one
    column per substitution type.
    """
    subs = mutation_specific_data[
        mutation_specific_data['ref_base'].isin(['A', 'C', 'G', 'T']) &
        mutation_specific_data['cons_base'].isin(['A', 'C', 'G', 'T'])
    ].copy()

    key_cols = ['umi_id', 'position', 'ref_base', 'cons_base']
    if all(c in subs.columns for c in key_cols):
        before = len(subs)
        subs = subs.drop_duplicates(subset=key_cols, keep='first')
        removed = before - len(subs)
        if removed:
            print(f"  Removed {removed} duplicate mutations before counting types")

    subs['substitution_type'] = subs['ref_base'] + '->' + subs['cons_base']

    group_cols = (['sample_name', 'umi_id', 'substitution_type']
                  if 'sample_name' in subs.columns
                  else ['umi_id', 'substitution_type'])
    counts = subs.groupby(group_cols).size().reset_index(name='count')

    if 'sample_name' in counts.columns:
        pivot = counts.pivot_table(
            index=['sample_name', 'umi_id'],
            columns='substitution_type',
            values='count',
            fill_value=0,
        ).astype(int)
    else:
        pivot = counts.pivot(
            index='umi_id',
            columns='substitution_type',
            values='count',
        ).fillna(0).astype(int)

    for st in _ALL_SUB_TYPES:
        if st not in pivot.columns:
            pivot[st] = 0
    return pivot[_ALL_SUB_TYPES]


# =============================================================================
# SEQUENCE LENGTH LOADING
# =============================================================================


def load_actual_sequence_lengths(compiled_consensus_dir):
    """Read per-REC sequence lengths from compiled-consensus FASTA files.

    Returns dict mapping (sample_name, rec_id) -> int.
    Maps LP###-NL43-… filenames to NL43-LP###-… to match the Sample column.
    """
    from pathlib import Path

    seq_lengths: dict = {}
    compiled_dir = Path(compiled_consensus_dir)

    if not compiled_dir.exists():
        print(f"Warning: compiled consensus directory not found: {compiled_dir}")
        return seq_lengths

    fasta_files = list(compiled_dir.glob('*.fa'))
    print(f"Loading sequence lengths from {len(fasta_files)} FASTA files…")

    total_seqs = 0
    for fa in fasta_files:
        raw = fa.stem.replace('-compiled-consensus', '')
        parts = raw.split('-')
        if len(parts) >= 2 and parts[0].startswith('LP') and parts[1] == 'NL43':
            sample = '-'.join([parts[1], parts[0]] + parts[2:])
        elif len(parts) >= 2 and parts[0] == 'NL43' and parts[1].startswith('LP'):
            sample = raw
        else:
            sample = raw

        with open(fa) as fh:
            lines = fh.readlines()

        rec_id = None
        seq_parts: list = []
        for line in lines:
            if line.startswith('>'):
                if rec_id and seq_parts:
                    seq_lengths[(sample, rec_id)] = len(''.join(seq_parts))
                    total_seqs += 1
                rec_id = line.strip()[1:]
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if rec_id and seq_parts:
            seq_lengths[(sample, rec_id)] = len(''.join(seq_parts))
            total_seqs += 1

    print(f"  Loaded lengths for {total_seqs} RECs")
    if seq_lengths:
        lengths = list(seq_lengths.values())
        print(f"  Length range: {min(lengths)}–{max(lengths)} nt  "
              f"| mean: {sum(lengths)/len(lengths):.0f} nt")
    return seq_lengths


# =============================================================================
# DATA PREPARATION
# =============================================================================


def prepare_consensus_sequence_data(
    mutation_count_data,
    mutation_specific_data,
    consensus_data,
    sample_groups,
    compiled_consensus_dir=None,
):
    """Build the final consensus_sequence_data DataFrame.

    Parameters
    ----------
    mutation_count_data : DataFrame
        Raw per-REC mutation counts (from pipeline).
    mutation_specific_data : DataFrame
        Flat list of individual mutations (from JSON report).
    consensus_data : DataFrame
        NL43 reference nucleotide-context table.
    sample_groups : list[dict]
        Loaded sample-groups YAML data.
    compiled_consensus_dir : str or None
        Path to compiled consensus FASTAs for actual sequence lengths.

    Returns
    -------
    DataFrame with columns: Sample, Condition, Dose, Group, REC_ID,
        Total_Mutations, Substitutions, Insertions, Deletions, Total_Bases,
        A->C … T->G.
    """
    df = mutation_count_data.copy()

    # Substitution type counts
    print("Counting substitution types per REC…")
    sub_counts = count_substitution_types_per_umi(mutation_specific_data).reset_index()

    if 'sample_name' in sub_counts.columns:
        df = df.merge(
            sub_counts,
            left_on=['Sample', 'UMI_ID'],
            right_on=['sample_name', 'umi_id'],
            how='left',
        )
    else:
        df = df.merge(sub_counts, left_on='UMI_ID', right_on='umi_id', how='left')

    for col in _ALL_SUB_TYPES:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(int)

    for drop_col in ('umi_id', 'sample_name'):
        if drop_col in df.columns:
            df = df.drop(columns=[drop_col])

    if 'UMI_ID' in df.columns:
        df = df.rename(columns={'UMI_ID': 'REC_ID'})

    # Reference length
    if 'position' in consensus_data.columns:
        max_pos = consensus_data['position'].max()
        min_pos = consensus_data['position'].min()
        ref_len = int(max_pos + 1) if min_pos == 0 else int(max_pos)
    else:
        ref_len = len(consensus_data)
    print(f"Reference length (NL43): {ref_len} nt")

    # Per-REC actual sequence lengths
    if compiled_consensus_dir:
        actual = load_actual_sequence_lengths(compiled_consensus_dir)
        if actual:
            df['Total_Bases'] = df.apply(
                lambda r: actual.get((r['Sample'], r['REC_ID']), ref_len), axis=1
            )
            dist = df['Total_Bases'].value_counts().sort_index()
            print("\n  Total_Bases distribution:")
            for length, cnt in dist.items():
                print(f"    {length} nt: {cnt} RECs ({cnt/len(df)*100:.1f}%)")
        else:
            print("  No actual lengths found; using reference length.")
            df['Total_Bases'] = ref_len
    else:
        df['Total_Bases'] = ref_len

    # Condition / Dose from sample name
    if 'Condition' not in df.columns:
        df['Condition'] = df['Sample'].str.split('-').str[2]
    if 'Dose' not in df.columns:
        df['Dose'] = df['Sample'].str.split('-').str[3]

    # Group assignment
    print("Mapping samples to groups…")
    df['Group'] = df['Sample'].apply(lambda s: get_sample_group(s, sample_groups))

    # Column selection / ordering
    required = [
        'Sample', 'Condition', 'Dose', 'Group', 'REC_ID',
        'Total_Mutations', 'Substitutions', 'Insertions', 'Deletions',
        'Total_Bases',
    ] + _ALL_SUB_TYPES
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Warning: missing columns: {missing}")
        required = [c for c in required if c in df.columns]

    return df[required].copy()


# =============================================================================
# MAIN
# =============================================================================


def main():
    """Load raw data, prepare consensus_sequence_data.csv, write to OUTPUT_DIR."""
    _validate_runtime_config()

    print("=" * 60)
    print("REC ANALYSIS — PREPARE DATA")
    print("=" * 60)

    print("\nLoading data…")
    ref_data    = load_consensus_sequence_data(CONSENSUS_SEQUENCE_DATA_CSV)
    count_data  = load_mutation_count_data(MUTATION_COUNT_DATA_CSV)
    mut_data    = load_mutation_specific_data(MUTATION_SPECIFIC_DATA_CSV)
    print(f"  Reference positions : {len(ref_data)}")
    print(f"  RECs                : {len(count_data)}")
    print(f"  Mutations           : {len(mut_data)}")

    sample_groups = load_sample_groups_config(SAMPLE_GROUPS_CONFIG_YAML)

    if INCLUDE_GROUPS:
        allowed = get_samples_in_groups(INCLUDE_GROUPS, sample_groups)
        before_r, before_m = len(count_data), len(mut_data)
        count_data = count_data[count_data['Sample'].isin(allowed)].copy()
        if 'sample_name' in mut_data.columns:
            mut_data = mut_data[mut_data['sample_name'].isin(allowed)].copy()
        print(f"\nFiltered to groups {INCLUDE_GROUPS}:")
        print(f"  RECs      : {len(count_data)}/{before_r}")
        print(f"  Mutations : {len(mut_data)}/{before_m}")

    print("\n" + "=" * 60)
    print("PREPARING CONSENSUS SEQUENCE DATA")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    result = prepare_consensus_sequence_data(
        mutation_count_data=count_data,
        mutation_specific_data=mut_data,
        consensus_data=ref_data,
        sample_groups=sample_groups,
        compiled_consensus_dir=COMPILED_CONSENSUS_DIR,
    )

    out_path = os.path.join(OUTPUT_DIR, 'consensus_sequence_data.csv')
    result.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
    print(f"  Rows    : {len(result)}")
    print(f"  Columns : {list(result.columns)}")

    return result


if __name__ == '__main__':
    main()
