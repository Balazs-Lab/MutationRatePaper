"""
Analyze frameshift mutations to determine if they result in stop codons.
"""

from typing import Dict, Optional, Tuple
from parse_genbank import Feature


# Standard genetic code
GENETIC_CODE = {
    'TTT': 'F', 'TTC': 'F', 'TTA': 'L', 'TTG': 'L',
    'TCT': 'S', 'TCC': 'S', 'TCA': 'S', 'TCG': 'S',
    'TAT': 'Y', 'TAC': 'Y', 'TAA': '*', 'TAG': '*',
    'TGT': 'C', 'TGC': 'C', 'TGA': '*', 'TGG': 'W',
    'CTT': 'L', 'CTC': 'L', 'CTA': 'L', 'CTG': 'L',
    'CCT': 'P', 'CCC': 'P', 'CCA': 'P', 'CCG': 'P',
    'CAT': 'H', 'CAC': 'H', 'CAA': 'Q', 'CAG': 'Q',
    'CGT': 'R', 'CGC': 'R', 'CGA': 'R', 'CGG': 'R',
    'ATT': 'I', 'ATC': 'I', 'ATA': 'I', 'ATG': 'M',
    'ACT': 'T', 'ACC': 'T', 'ACA': 'T', 'ACG': 'T',
    'AAT': 'N', 'AAC': 'N', 'AAA': 'K', 'AAG': 'K',
    'AGT': 'S', 'AGC': 'S', 'AGA': 'R', 'AGG': 'R',
    'GTT': 'V', 'GTC': 'V', 'GTA': 'V', 'GTG': 'V',
    'GCT': 'A', 'GCC': 'A', 'GCA': 'A', 'GCG': 'A',
    'GAT': 'D', 'GAC': 'D', 'GAA': 'E', 'GAG': 'E',
    'GGT': 'G', 'GGC': 'G', 'GGA': 'G', 'GGG': 'G',
}

STOP_CODONS = {'TAA', 'TAG', 'TGA'}


def get_cds_sequence(genomic_sequence: str, feature: Feature) -> str:
    """Extract the CDS sequence from genomic sequence."""
    cds_seq = ""
    for start, end in feature.locations:
        cds_seq += genomic_sequence[start - 1:end]
    return cds_seq


def codon_to_aa(codon: str) -> str:
    """Convert a codon to amino acid."""
    return GENETIC_CODE.get(codon.upper(), 'X')


def apply_deletion(genomic_sequence: str, feature: Feature, 
                   genomic_pos: int, deleted_seq: str) -> str:
    """
    Apply deletion to CDS sequence and return mutated CDS.
    """
    cds_pos = feature.get_cds_position(genomic_pos)
    if cds_pos is None:
        return None
    
    cds_seq = get_cds_sequence(genomic_sequence, feature)
    
    # Convert to 0-based for slicing
    pos_0based = cds_pos - 1
    deletion_length = len(deleted_seq)
    
    # Create mutated CDS
    mutated_cds = cds_seq[:pos_0based] + cds_seq[pos_0based + deletion_length:]
    
    return mutated_cds


def apply_insertion(genomic_sequence: str, feature: Feature,
                    genomic_pos: int, inserted_seq: str) -> str:
    """
    Apply insertion to CDS sequence and return mutated CDS.
    """
    cds_pos = feature.get_cds_position(genomic_pos)
    if cds_pos is None:
        return None
    
    cds_seq = get_cds_sequence(genomic_sequence, feature)
    
    # Convert to 0-based for slicing
    pos_0based = cds_pos - 1
    
    # Create mutated CDS
    mutated_cds = cds_seq[:pos_0based] + inserted_seq.upper() + cds_seq[pos_0based:]
    
    return mutated_cds


def find_stop_codon(cds_sequence: str, start_from_pos: int = 0) -> Optional[int]:
    """
    Find the first stop codon in a CDS sequence.
    
    Args:
        cds_sequence: CDS sequence
        start_from_pos: Position to start searching from (0-based nucleotide position)
    
    Returns:
        Position of stop codon (0-based nucleotide position) or None
    """
    # Start from the beginning of the codon containing start_from_pos
    codon_start = (start_from_pos // 3) * 3
    
    for i in range(codon_start, len(cds_sequence) - 2, 3):
        codon = cds_sequence[i:i+3]
        if codon in STOP_CODONS:
            return i
    
    return None


def analyze_frameshift(genomic_sequence: str, feature: Feature,
                      mutation_type: str, genomic_pos: int,
                      ref_base: str, cons_base: str) -> Dict:
    """
    Analyze a frameshift mutation to determine if it creates a stop codon.
    
    Returns dict with:
    - is_frameshift: True if mutation causes frameshift
    - creates_stop_codon: True if frameshift creates a premature stop codon
    - stop_codon_position: CDS position of stop codon (1-based) or None
    - stop_codon_aa_position: Amino acid position of stop codon (1-based) or None
    - is_lethal: True if creates stop codon (premature termination)
    - mutated_cds_length: Length of mutated CDS
    - original_cds_length: Length of original CDS
    """
    result = {
        'is_frameshift': False,
        'creates_stop_codon': False,
        'stop_codon_position': None,
        'stop_codon_aa_position': None,
        'is_lethal': False,
        'mutated_cds_length': None,
        'original_cds_length': None
    }
    
    # Get original CDS
    original_cds = get_cds_sequence(genomic_sequence, feature)
    result['original_cds_length'] = len(original_cds)
    
    # Apply mutation
    if mutation_type == 'deletion':
        deleted_length = len(ref_base)
        if (deleted_length % 3) != 0:
            result['is_frameshift'] = True
            mutated_cds = apply_deletion(genomic_sequence, feature, genomic_pos, ref_base)
        else:
            # In-frame deletion, not a frameshift
            mutated_cds = apply_deletion(genomic_sequence, feature, genomic_pos, ref_base)
            result['mutated_cds_length'] = len(mutated_cds) if mutated_cds else None
            return result
    
    elif mutation_type == 'insertion':
        inserted_length = len(cons_base)
        if (inserted_length % 3) != 0:
            result['is_frameshift'] = True
            mutated_cds = apply_insertion(genomic_sequence, feature, genomic_pos, cons_base)
        else:
            # In-frame insertion, not a frameshift
            mutated_cds = apply_insertion(genomic_sequence, feature, genomic_pos, cons_base)
            result['mutated_cds_length'] = len(mutated_cds) if mutated_cds else None
            return result
    
    else:
        # Substitutions don't cause frameshifts
        return result
    
    if not mutated_cds:
        return result
    
    result['mutated_cds_length'] = len(mutated_cds)
    
    # Find stop codon after the mutation site
    cds_pos = feature.get_cds_position(genomic_pos)
    if cds_pos:
        # Start searching from the mutation site (convert to 0-based)
        search_start = cds_pos - 1
        stop_pos = find_stop_codon(mutated_cds, search_start)
        
        if stop_pos is not None:
            result['creates_stop_codon'] = True
            result['stop_codon_position'] = stop_pos + 1  # Convert to 1-based
            result['stop_codon_aa_position'] = (stop_pos // 3) + 1  # 1-based codon number
            result['is_lethal'] = True
    
    return result
