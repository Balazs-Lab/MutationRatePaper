#!/usr/bin/env python3
"""
Trim consensus sequences to match aligned regions only (remove soft-clipped bases)
"""

import sys
import re

def parse_cigar(cigar_string):
    """
    Parse CIGAR string and return left and right soft-clip lengths
    
    Args:
        cigar_string: CIGAR string (e.g., "33S2550M30S")
    
    Returns:
        tuple: (left_clip, right_clip)
    """
    left_clip = 0
    right_clip = 0
    
    # Match leading soft-clip (e.g., 33S at the start)
    match = re.match(r'^(\d+)S', cigar_string)
    if match:
        left_clip = int(match.group(1))
    
    # Match trailing soft-clip (e.g., 30S at the end)
    match = re.search(r'(\d+)S$', cigar_string)
    if match:
        right_clip = int(match.group(1))
    
    return left_clip, right_clip


def trim_sequence(seq, left_clip, right_clip):
    """
    Trim sequence by removing soft-clipped regions
    
    Args:
        seq: sequence string
        left_clip: number of bases to remove from 5' end
        right_clip: number of bases to remove from 3' end
    
    Returns:
        str: trimmed sequence
    """
    seq_len = len(seq)
    trim_len = seq_len - left_clip - right_clip
    
    if trim_len <= 0:
        return None
    
    if right_clip == 0:
        return seq[left_clip:]
    else:
        return seq[left_clip:-right_clip]


def main():
    """Process SAM input from stdin and output trimmed FASTA to stdout"""
    for line in sys.stdin:
        # Skip header lines
        if line.startswith('@'):
            continue
        
        fields = line.strip().split('\t')
        if len(fields) < 10:
            continue
        
        read_name = fields[0]
        cigar = fields[5]
        sequence = fields[9]
        
        # Parse CIGAR to get soft-clip lengths
        left_clip, right_clip = parse_cigar(cigar)
        
        # Trim the sequence
        trimmed_seq = trim_sequence(sequence, left_clip, right_clip)
        
        # Output FASTA
        if trimmed_seq:
            print(f">{read_name}")
            print(trimmed_seq)


if __name__ == "__main__":
    main()
