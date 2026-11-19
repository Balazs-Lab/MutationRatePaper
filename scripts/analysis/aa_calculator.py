"""
Calculate amino acid changes for mutations in CDS features.
Handles substitutions, insertions, and deletions.
"""

from typing import List, Dict, Optional, Tuple
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


def get_cds_sequence(genomic_sequence: str, feature: Feature) -> str:
    """
    Extract the CDS sequence from genomic sequence.
    For join() features, concatenates the exons.
    """
    cds_seq = ""
    for start, end in feature.locations:
        # GenBank uses 1-based coordinates, Python uses 0-based
        cds_seq += genomic_sequence[start - 1:end]
    return cds_seq


def codon_to_aa(codon: str) -> str:
    """Convert a codon to amino acid."""
    return GENETIC_CODE.get(codon.upper(), 'X')


def get_codon_at_position(cds_sequence: str, cds_pos: int) -> Tuple[str, int]:
    """
    Get the codon containing the given CDS position (1-based).
    Returns (codon, codon_number) where codon_number is 1-based.
    """
    # CDS position is 1-based, convert to 0-based for indexing
    pos_0based = cds_pos - 1
    codon_start = (pos_0based // 3) * 3
    codon_number = (pos_0based // 3) + 1
    codon = cds_sequence[codon_start:codon_start + 3]
    return codon, codon_number


def calculate_aa_change_substitution(genomic_sequence: str, feature: Feature, 
                                     genomic_pos: int, ref_base: str, cons_base: str) -> Dict:
    """
    Calculate amino acid change for a substitution mutation.
    
    Returns dict with:
    - cds_position: nucleotide position in CDS (1-based)
    - codon_position: codon number (1-based)
    - ref_codon: reference codon
    - cons_codon: consensus (mutated) codon
    - ref_aa: reference amino acid
    - cons_aa: consensus amino acid (same as ref_aa if synonymous)
    - aa_change: string like "L123F" or "synonymous"
    - effect: effect label (e.g., "disrupts_start_codon", "introduces_stop_codon", "missense", "synonymous")
    """
    cds_pos = feature.get_cds_position(genomic_pos)
    if cds_pos is None:
        return None
    
    # Get CDS sequence
    cds_seq = get_cds_sequence(genomic_sequence, feature)
    
    # Get codon information
    ref_codon, codon_num = get_codon_at_position(cds_seq, cds_pos)
    
    # Create mutated codon
    pos_in_codon = (cds_pos - 1) % 3
    cons_codon = list(ref_codon)
    cons_codon[pos_in_codon] = cons_base.upper()
    cons_codon = ''.join(cons_codon)
    
    # Get amino acids
    ref_aa = codon_to_aa(ref_codon)
    cons_aa = codon_to_aa(cons_codon)
    
    # Determine effect
    effects = []
    
    # Check for start codon disruption (ATG at position 1)
    if codon_num == 1 and ref_codon == 'ATG' and cons_codon != 'ATG':
        effects.append('disrupts_start_codon')
    
    # Check for stop codon introduction
    if ref_aa != '*' and cons_aa == '*':
        effects.append('introduces_premature_stop_codon')
    
    # Check for stop codon loss
    if ref_aa == '*' and cons_aa != '*':
        effects.append('removes_stop_codon')
    
    # Classify mutation type
    if ref_aa == cons_aa:
        effects.append('synonymous')
    elif cons_aa == '*':
        effects.append('nonsense')
    else:
        effects.append('missense')
    
    # Format change string
    if ref_aa == cons_aa:
        aa_change = "synonymous"
    else:
        aa_change = f"{ref_aa}{codon_num}{cons_aa}"
    
    # Join effects (most specific first)
    effect_label = ';'.join(effects) if effects else 'unknown'
    
    return {
        'cds_position': cds_pos,
        'codon_position': codon_num,
        'ref_codon': ref_codon,
        'cons_codon': cons_codon,
        'ref_aa': ref_aa,
        'cons_aa': cons_aa,
        'aa_change': aa_change,
        'effect': effect_label
    }


def calculate_aa_change_deletion(genomic_sequence: str, feature: Feature,
                                genomic_pos: int, deleted_seq: str) -> Dict:
    """
    Calculate amino acid change for a deletion mutation.
    
    Returns dict with:
    - cds_position: nucleotide position in CDS (1-based)
    - codon_position: codon number (1-based)
    - deletion_length: number of bases deleted
    - frameshift: True if deletion length is not a multiple of 3
    - ref_aa: reference amino acid at the codon (for in-frame deletions)
    - cons_aa: consensus amino acid (same as ref_aa for in-frame, N/A for frameshift)
    - aa_change: description of the change
    - effect: effect label
    """
    cds_pos = feature.get_cds_position(genomic_pos)
    if cds_pos is None:
        return None
    
    deletion_length = len(deleted_seq)
    frameshift = (deletion_length % 3) != 0
    
    # Get CDS sequence
    cds_seq = get_cds_sequence(genomic_sequence, feature)
    
    # Get codon information
    ref_codon, codon_num = get_codon_at_position(cds_seq, cds_pos)
    ref_aa = codon_to_aa(ref_codon)
    
    # Determine effects
    effects = []
    
    # Check for start codon disruption
    if codon_num == 1 and ref_codon == 'ATG':
        effects.append('disrupts_start_codon')
    
    if frameshift:
        effects.append('frameshift_deletion')
        cons_aa = ref_aa  # Keep ref_aa for display, but frameshift changes everything downstream
        aa_change = f"frameshift_deletion_{deletion_length}bp"
    else:
        effects.append('in_frame_deletion')
        cons_aa = ref_aa  # In-frame deletion, same AA context
        aa_change = f"in_frame_deletion_{deletion_length}bp"
    
    effect_label = ';'.join(effects) if effects else 'unknown'
    
    return {
        'cds_position': cds_pos,
        'codon_position': codon_num,
        'deletion_length': deletion_length,
        'frameshift': frameshift,
        'ref_aa': ref_aa,
        'cons_aa': cons_aa,
        'aa_change': aa_change,
        'effect': effect_label
    }


def calculate_aa_change_insertion(genomic_sequence: str, feature: Feature,
                                 genomic_pos: int, inserted_seq: str) -> Dict:
    """
    Calculate amino acid change for an insertion mutation.
    
    Returns dict with:
    - cds_position: nucleotide position in CDS (1-based)
    - codon_position: codon number (1-based)
    - insertion_length: number of bases inserted
    - frameshift: True if insertion length is not a multiple of 3
    - ref_aa: reference amino acid at the codon
    - cons_aa: consensus amino acid (same as ref_aa, insertion doesn't change the codon itself)
    - aa_change: description of the change
    - effect: effect label
    """
    cds_pos = feature.get_cds_position(genomic_pos)
    if cds_pos is None:
        return None
    
    insertion_length = len(inserted_seq)
    frameshift = (insertion_length % 3) != 0
    
    # Get CDS sequence
    cds_seq = get_cds_sequence(genomic_sequence, feature)
    
    # Get codon information
    ref_codon, codon_num = get_codon_at_position(cds_seq, cds_pos)
    ref_aa = codon_to_aa(ref_codon)
    
    # Determine effects
    effects = []
    
    # Check for start codon disruption
    if codon_num == 1 and ref_codon == 'ATG':
        effects.append('disrupts_start_codon')
    
    if frameshift:
        effects.append('frameshift_insertion')
        cons_aa = ref_aa  # Keep ref_aa for display
        aa_change = f"frameshift_insertion_{insertion_length}bp"
    else:
        effects.append('in_frame_insertion')
        cons_aa = ref_aa  # In-frame insertion, same AA context
        aa_change = f"in_frame_insertion_{insertion_length}bp"
    
    effect_label = ';'.join(effects) if effects else 'unknown'
    
    return {
        'cds_position': cds_pos,
        'codon_position': codon_num,
        'insertion_length': insertion_length,
        'frameshift': frameshift,
        'ref_aa': ref_aa,
        'cons_aa': cons_aa,
        'aa_change': aa_change,
        'effect': effect_label
    }


def calculate_aa_change(genomic_sequence: str, feature: Feature,
                       mutation_type: str, genomic_pos: int,
                       ref_base: str, cons_base: str) -> Optional[Dict]:
    """
    Calculate amino acid change for any mutation type.
    
    Args:
        genomic_sequence: Full genomic sequence
        feature: CDS feature
        mutation_type: 'substitution', 'deletion', or 'insertion'
        genomic_pos: Genomic position (1-based)
        ref_base: Reference base(s) or sequence
        cons_base: Consensus (mutated) base(s) or sequence
    
    Returns:
        Dict with amino acid change information or None
    """
    if mutation_type == 'substitution':
        return calculate_aa_change_substitution(
            genomic_sequence, feature, genomic_pos, ref_base, cons_base
        )
    elif mutation_type == 'deletion':
        return calculate_aa_change_deletion(
            genomic_sequence, feature, genomic_pos, ref_base
        )
    elif mutation_type == 'insertion':
        return calculate_aa_change_insertion(
            genomic_sequence, feature, genomic_pos, cons_base
        )
    else:
        return None
