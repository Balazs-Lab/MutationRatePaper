#!/usr/bin/env python3
"""
Edge-trim a single-record FASTA reference.

Removes N bases from each end (5' and 3') of the trimmed reference so that
downstream consensus compilation and mutation calling exclude PCR-induced edge
effects. The reference-map CSV (library -> biological coordinates) is left
untouched; the 5' shift introduced here is compensated in
remap_mutation_positions.py via --offset.
"""

import argparse


def read_single_fasta(path):
    """Read a single-record FASTA. Returns (header_line, sequence_string)."""
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    raise ValueError(
                        f"{path} contains more than one record; edge trimming "
                        "expects a single-record reference."
                    )
                header = line
            else:
                seq_parts.append(line.strip())
    if header is None:
        raise ValueError(f"{path} contains no FASTA header line.")
    return header, "".join(seq_parts)


def main():
    parser = argparse.ArgumentParser(description="Trim N bp from each end of a single-record FASTA")
    parser.add_argument("--input", required=True, help="Input FASTA (single record)")
    parser.add_argument("--output", required=True, help="Output edge-trimmed FASTA")
    parser.add_argument("--trim", type=int, required=True, help="Bases to remove from EACH end")
    args = parser.parse_args()

    if args.trim < 0:
        raise ValueError("--trim must be >= 0")

    header, seq = read_single_fasta(args.input)
    n = args.trim

    if len(seq) - 2 * n <= 0:
        raise ValueError(
            f"--trim {n} too large: sequence length {len(seq)} would be fully consumed."
        )

    trimmed = seq[n:len(seq) - n] if n > 0 else seq

    with open(args.output, "w") as f:
        f.write(f"{header}\n")
        f.write(f"{trimmed}\n")


if __name__ == "__main__":
    main()
