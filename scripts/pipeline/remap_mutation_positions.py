#!/usr/bin/env python3
"""
Remap mutation positions from library reference to biological reference
Uses the reference map CSV to convert positions
"""

import argparse
import json
import csv
import pandas as pd
import yaml
from pathlib import Path


def load_reference_map(map_file):
    """
    Load reference position mapping from CSV
    
    Args:
        map_file: Path to reference map CSV
    
    Returns:
        dict: Mapping from library_pos -> ref_pos
    """
    df = pd.read_csv(map_file)
    
    # Create mapping: library_pos -> ref_pos
    # Filter out rows where library position is NaN
    df_valid = df[df.iloc[:, 2].notna()].copy()  # Column 2 is LP249_pos or similar
    
    # Column names: ref_pos, ref_base, library_pos, library_base
    lib_col = df.columns[2]  # e.g., "LP249_pos"
    ref_col = df.columns[0]  # e.g., "ref_pos"
    
    position_map = {}
    for _, row in df_valid.iterrows():
        lib_pos = int(row[lib_col])
        ref_pos = int(row[ref_col])
        position_map[lib_pos] = ref_pos
    
    return position_map


def load_library_to_biological_mapping(mapping_file):
    """
    Load mapping from library names to biological reference names
    
    Args:
        mapping_file: Path to YAML mapping file
    
    Returns:
        dict: Mapping from library_name -> biological_reference_name
    """
    with open(mapping_file, 'r') as f:
        mapping = yaml.safe_load(f)
    return mapping


def remap_json_file(json_file, position_map, output_file, library_name, biological_ref, offset=0):
    """
    Remap positions in mutation JSON file
    
    Args:
        json_file: Input JSON file path
        position_map: Dictionary mapping library_pos -> ref_pos
        output_file: Output JSON file path
        library_name: Name of the library (e.g., "LP249")
        biological_ref: Name of the biological reference (e.g., "NL43")
        offset: 5' edge-trim length (bp) applied to the reference used for
            mutation calling. Positions are shifted back into the original
            trimmed-reference (library) coordinate system before lookup.
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Update sample_name to start with biological reference for validation
    # This ensures belt mutation-summary uses the correct reference
    # Preserve middle token (e.g. "Lib" vs "NL43") so positive-control Lib samples appear separately.
    original_sample_name = data.get('sample_name', '')
    if original_sample_name.startswith(library_name + '-'):
        sample_parts = original_sample_name.split('-')
        # parts[0]=library (LP249), parts[1]=Lib or NL43 or other, parts[2:]=rest (e.g. WT-1-A)
        if len(sample_parts) >= 2 and sample_parts[1] == biological_ref:
            # e.g. LP249-NL43-WT-1-A -> NL43-LP249-WT-1-A
            new_sample_name = biological_ref + '-' + library_name + '-' + '-'.join(sample_parts[2:])
        else:
            # e.g. LP252-Lib-WT-1-A -> NL43-LP252-Lib-WT-1-A (keep "Lib" for positive control)
            new_sample_name = biological_ref + '-' + library_name + '-' + '-'.join(sample_parts[1:])
        data['sample_name'] = new_sample_name
        # Also update sample_id in mutations
        for mutation in data.get('mutations_by_position', []):
            if 'sample_id' in mutation:
                mutation['sample_id'] = new_sample_name
    
    # Remap positions in mutations_by_position
    if 'mutations_by_position' in data:
        remapped_mutations = []
        for mutation in data['mutations_by_position']:
            if 'position' in mutation:
                lib_pos_0based = mutation['position']
                # Convert from 0-based to 1-based indexing, shifting by the 5'
                # edge-trim offset back into original library coordinates.
                lib_pos_1based = lib_pos_0based + 1 + offset
                if lib_pos_1based in position_map:
                    # Add both library and reference positions (1-based)
                    mutation['library_position'] = lib_pos_1based
                    mutation['library_name'] = library_name
                    mutation['ref_position'] = position_map[lib_pos_1based]
                    # Keep 'position' as ref_position for backwards compatibility
                    mutation['position'] = position_map[lib_pos_1based]
                    remapped_mutations.append(mutation)
        
        # Update with only successfully remapped mutations
        data['mutations_by_position'] = remapped_mutations
        
        # Update counts
        data['total_mutations'] = len(remapped_mutations)
        
        # Recount mutation types - ensure all types are present even if count is 0
        mutation_types = {'substitution': 0, 'insertion': 0, 'deletion': 0}
        for mutation in remapped_mutations:
            mut_type = mutation.get('type', 'unknown')
            mutation_types[mut_type] = mutation_types.get(mut_type, 0) + 1
        data['mutation_types'] = mutation_types
    
    # Write remapped data
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)


def remap_csv_file(csv_file, position_map, output_file, library_name, offset=0):
    """
    Remap positions in per-read CSV file
    
    Args:
        csv_file: Input CSV file path
        position_map: Dictionary mapping library_pos -> ref_pos
        output_file: Output CSV file path
        library_name: Name of the library (e.g., "LP249")
        offset: 5' edge-trim length (bp) applied to the mutation-calling
            reference; positions are shifted back into original library
            coordinates before lookup.
    """
    df = pd.read_csv(csv_file)
    
    # Check if there's a position column
    if 'position' in df.columns:
        # Convert from 0-based to 1-based indexing, shifting by the 5' edge-trim
        # offset back into original library coordinates.
        df['library_position'] = df['position'] + 1 + offset
        df['library_name'] = library_name
        # Remap to reference position using 1-based library position
        df['ref_position'] = df['library_position'].map(position_map)
        # Keep 'position' as ref_position for backwards compatibility
        df['position'] = df['ref_position']
        # Drop rows where position couldn't be mapped (outside library region)
        df = df.dropna(subset=['position'])
        df['position'] = df['position'].astype(int)
        df['ref_position'] = df['ref_position'].astype(int)
        df['library_position'] = df['library_position'].astype(int)
    
    # Write remapped data
    df.to_csv(output_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Remap mutation positions to biological reference')
    parser.add_argument('--map-file', required=True, help='Reference map CSV file')
    parser.add_argument('--json-file', required=True, help='Input mutations JSON file')
    parser.add_argument('--csv-file', required=True, help='Input per-read CSV file')
    parser.add_argument('--output-json', required=True, help='Output remapped JSON file')
    parser.add_argument('--output-csv', required=True, help='Output remapped CSV file')
    parser.add_argument('--library-name', required=True, help='Library name (e.g., LP249)')
    parser.add_argument('--library-mapping-file', required=True, help='Library to biological reference mapping YAML file')
    parser.add_argument('--offset', type=int, default=0,
                        help="5' edge-trim length (bp) applied to the mutation-calling "
                             "reference. Positions are shifted back into original "
                             "trimmed-reference (library) coordinates before lookup. Default: 0")
    
    args = parser.parse_args()
    
    # Load library to biological reference mapping
    print(f"Loading library to biological reference mapping from {args.library_mapping_file}")
    library_to_bio = load_library_to_biological_mapping(args.library_mapping_file)
    biological_ref = library_to_bio.get(args.library_name)
    
    if not biological_ref:
        raise ValueError(f"Library '{args.library_name}' not found in mapping file. Available libraries: {list(library_to_bio.keys())}")
    
    print(f"Library {args.library_name} maps to biological reference: {biological_ref}")
    
    # Load position mapping
    print(f"Loading reference map from {args.map_file}")
    position_map = load_reference_map(args.map_file)
    print(f"Loaded {len(position_map)} position mappings")
    
    if args.offset:
        print(f"Applying 5' edge-trim offset: {args.offset} bp")

    # Remap JSON file
    print(f"Remapping JSON file: {args.json_file}")
    remap_json_file(args.json_file, position_map, args.output_json, args.library_name, biological_ref, offset=args.offset)
    print(f"Wrote remapped JSON to: {args.output_json}")
    
    # Remap CSV file  
    print(f"Remapping CSV file: {args.csv_file}")
    remap_csv_file(args.csv_file, position_map, args.output_csv, args.library_name, offset=args.offset)
    print(f"Wrote remapped CSV to: {args.output_csv}")
    
    print("Position remapping complete!")


if __name__ == "__main__":
    main()
