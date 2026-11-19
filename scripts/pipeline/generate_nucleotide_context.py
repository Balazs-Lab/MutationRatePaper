#!/usr/bin/env python3
"""
Generate nucleotide context CSV file from a reference FASTA
"""

import sys
import csv
from Bio import SeqIO

def generate_context(fasta_file, output_file):
    """Generate nucleotide context CSV from FASTA file"""
    
    # Load the reference
    seq_record = SeqIO.read(fasta_file, "fasta")
    sequence = str(seq_record.seq)
    
    print(f"Reference: {fasta_file}")
    print(f"Length: {len(sequence)} bases")
    
    # Generate nucleotide context
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['position', 'nucleotide', 'upstream_nucleotide', 'downstream_nucleotide', 
                      'second_downstream_nucleotide', 'dinucleotide_context', 'tri_nucleotide_context']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for i, nuc in enumerate(sequence):
            upstream = sequence[i-1] if i > 0 else 'N'
            downstream = sequence[i+1] if i < len(sequence) - 1 else 'N'
            second_downstream = sequence[i+2] if i < len(sequence) - 2 else 'N'
            
            # Dinucleotide context (current + downstream)
            dinuc = nuc + downstream
            
            # Trinucleotide context (upstream + current + downstream)
            trinuc = upstream + nuc + downstream
            
            writer.writerow({
                'position': i,
                'nucleotide': nuc,
                'upstream_nucleotide': upstream,
                'downstream_nucleotide': downstream,
                'second_downstream_nucleotide': second_downstream,
                'dinucleotide_context': dinuc,
                'tri_nucleotide_context': trinuc
            })
    
    print(f"Generated context file: {output_file}")
    print(f"Total positions: {len(sequence)}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: generate_nucleotide_context.py <input.fasta> <output.csv>")
        sys.exit(1)
    
    generate_context(sys.argv[1], sys.argv[2])
