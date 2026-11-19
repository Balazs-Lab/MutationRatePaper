#!/usr/bin/env python3
"""Load hand-crafted priors from data/models/mra/mra_priors.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _param_name(prefix: str, *index: object) -> str:
    if not index:
        return prefix
    inner = ", ".join(str(x) for x in index)
    return f"{prefix}[{inner}]"


class FilePriors:
    """Lookup table for Normal / HalfNormal / Exponential priors by parameter name."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Prior file not found: {self.path}")
        df = pd.read_csv(self.path)
        required = {"parameter", "family", "location", "scale"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{self.path} is missing columns: {sorted(missing)}")
        df["parameter"] = df["parameter"].astype(str)
        df["family"] = df["family"].astype(str)
        self.table = df.set_index("parameter", drop=False)
        if self.table.index.has_duplicates:
            dups = self.table.index[self.table.index.duplicated()].unique().tolist()
            raise ValueError(f"Duplicate prior rows: {dups[:8]}")

    def __contains__(self, name: str) -> bool:
        return name in self.table.index

    def _row(self, name: str) -> pd.Series:
        if name not in self.table.index:
            raise KeyError(f"No prior for {name!r} in {self.path}")
        return self.table.loc[name]

    def require_family(self, name: str, family: str) -> pd.Series:
        row = self._row(name)
        if str(row["family"]) != family:
            raise ValueError(
                f"Prior for {name} is {row['family']}, expected {family}"
            )
        return row

    def normal(self, name: str) -> tuple[float, float]:
        row = self.require_family(name, "Normal")
        return float(row["location"]), float(row["scale"])

    def halfnormal(self, name: str) -> float:
        row = self.require_family(name, "HalfNormal")
        return float(row["scale"])

    def exponential(self, name: str) -> float:
        row = self.require_family(name, "Exponential")
        return float(row["scale"])

    def normal_vector(self, prefix: str, levels: list[object]) -> tuple[np.ndarray, np.ndarray]:
        names = [_param_name(prefix, level) for level in levels]
        mus = np.empty(len(names), dtype=np.float64)
        sigmas = np.empty(len(names), dtype=np.float64)
        for i, name in enumerate(names):
            mus[i], sigmas[i] = self.normal(name)
        return mus, sigmas

    def halfnormal_vector(self, prefix: str, levels: list[object]) -> np.ndarray:
        return np.array(
            [self.halfnormal(_param_name(prefix, level)) for level in levels],
            dtype=np.float64,
        )

    def exponential_vector(self, prefix: str, levels: list[object]) -> np.ndarray:
        return np.array(
            [self.exponential(_param_name(prefix, level)) for level in levels],
            dtype=np.float64,
        )

    def normal_matrix(
        self,
        prefix: str,
        row_levels: list[object],
        col_levels: list[object],
    ) -> tuple[np.ndarray, np.ndarray]:
        mus = np.empty((len(row_levels), len(col_levels)), dtype=np.float64)
        sigmas = np.empty_like(mus)
        for i, row in enumerate(row_levels):
            for j, col in enumerate(col_levels):
                mus[i, j], sigmas[i, j] = self.normal(_param_name(prefix, row, col))
        return mus, sigmas

    def unique_prior_summary(self) -> dict[str, dict]:
        if "unique_prior" not in self.table.columns:
            return {}

        def _opt_float(val: object) -> float | None:
            if val is None or (isinstance(val, str) and val.strip() == "") or pd.isna(val):
                return None
            return float(val)

        def _opt_str(val: object) -> str | None:
            if val is None or pd.isna(val):
                return None
            text = str(val).strip()
            return text or None

        out: dict[str, dict] = {}
        for name, sub in self.table.groupby("unique_prior", sort=False):
            row = sub.iloc[0]
            out[str(name)] = {
                "n": int(len(sub)),
                "family": str(row["family"]),
                "location": _opt_float(row["location"]),
                "scale": float(row["scale"]),
                "location_note": _opt_str(row["location_note"]) if "location_note" in row else None,
            }
        return out
