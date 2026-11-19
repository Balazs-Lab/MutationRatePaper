#!/usr/bin/env python3
"""Merge Stanford HIVDB mutation-prevalence downloads into variant_prevalence.xlsx.

Reads the Mutation Prevalence According to Subtype and Treatment exports in
data/mutation_supply/SDRD/ (not the archived copies in SDRD/V1/) and writes the
table used by the figure scripts:

    protein, Pos, Cons, Mut, Rx, DRM, all_seqs, all_muts

Only the pooled counts (all_seqs, all_muts) are kept. Subtype-specific counts
are dropped. Proteins are renamed to match the MRA tables (PR -> Pro, CA -> p24).

The NNRTI exports are not included. Their naive arm duplicates the RTI naive
arm, and the compiled table used by Figures 4-6 does not contain the NNRTI,
EFV, RPV, or DOR treatment bins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SDRD_DIR = PROJECT_ROOT / "data" / "mutation_supply" / "SDRD"
OUT_PATH = PROJECT_ROOT / "data" / "mutation_supply" / "variant_prevalence.xlsx"

# File-name prefix -> (protein label, treatment bins to keep).
# Prefixes are matched against the start of each CSV name so a change in the
# HIVDB filter suffix (PatientPerMutCount vs PatientCount, prevalence cutoff)
# does not require editing this table.
SOURCES = [
    ("PI.", "Pro", ("naive", "PI", "ATV", "LPV")),
    ("RTI.", "RT", ("naive", "NRTI", "TFV+XTC", "AZT+XTC", "ABC+XTC")),
    ("INI.", "IN", ("naive", "INI", "DTG", "CAB")),
    ("CAI.", "p24", ("naive", "CAI")),
]

KEEP_COLUMNS = ["protein", "Pos", "Cons", "Mut", "Rx", "DRM", "all_seqs", "all_muts"]


def _csvs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("*.csv") if p.is_file())


def _match(csvs: list[Path], prefix: str) -> Path:
    hits = [p for p in csvs if p.name.startswith(prefix)]
    if len(hits) != 1:
        names = [p.name for p in hits] or ["<none>"]
        raise SystemExit(f"Expected one CSV starting with {prefix!r} in {SDRD_DIR}, found: {names}")
    return hits[0]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]
    missing = [c for c in ("Pos", "Cons", "Mut", "Rx", "DRM", "all_seqs", "all_muts") if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns: {missing}")
    return df


def build(directory: Path = SDRD_DIR) -> pd.DataFrame:
    csvs = _csvs(directory)
    frames: list[pd.DataFrame] = []
    for prefix, protein, rx_keep in SOURCES:
        path = _match(csvs, prefix)
        df = _read(path)
        unknown = sorted(set(df["Rx"]) - set(rx_keep))
        if unknown:
            raise SystemExit(f"{path.name} has unexpected Rx values: {unknown}")
        df = df[df["Rx"].isin(rx_keep)].copy()
        df.insert(0, "protein", protein)
        frames.append(df[KEEP_COLUMNS])
        print(f"{protein:4s}  {len(df):6d} rows  <-  {path.name}")
    out = pd.concat(frames, ignore_index=True)
    for col in ("Pos", "DRM", "all_seqs", "all_muts"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype(int)
    return out


def main() -> None:
    if not SDRD_DIR.is_dir():
        raise SystemExit(f"Download directory not found: {SDRD_DIR}")
    out = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_excel(OUT_PATH, index=False)
    print(f"\nWrote {len(out)} rows -> {OUT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
