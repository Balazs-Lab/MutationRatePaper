#!/usr/bin/env python3
"""Load non-prior run settings from an mra_run_*.csv file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _parse(value: Any, kind: str) -> Any:
    if kind == "int":
        return int(value)
    if kind == "float":
        return float(value)
    if kind == "bool":
        text = str(value).strip().lower()
        if text in {"1", "true", "yes"}:
            return True
        if text in {"0", "false", "no"}:
            return False
        raise ValueError(f"Cannot parse bool from {value!r}")
    if kind == "str":
        return str(value)
    if kind == "list":
        if isinstance(value, list):
            return [str(x) for x in value]
        return [token.strip() for token in str(value).split(",") if token.strip()]
    raise ValueError(f"Unknown run-setting type {kind!r}")


class RunConfig:
    """Documented run settings from a CSV passed on the command line."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Run settings file not found: {self.path}")
        df = pd.read_csv(self.path)
        required = {"parameter", "value", "type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{self.path} is missing columns: {sorted(missing)}")
        if df["parameter"].duplicated().any():
            dups = df.loc[df["parameter"].duplicated(), "parameter"].tolist()
            raise ValueError(f"Duplicate run settings: {dups}")
        self.table = df.set_index("parameter", drop=False)

    def raw(self, name: str) -> pd.Series:
        if name not in self.table.index:
            raise KeyError(f"No run setting {name!r} in {self.path}")
        return self.table.loc[name]

    def get(self, name: str) -> Any:
        row = self.raw(name)
        return _parse(row["value"], str(row["type"]))

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name in self.table.index:
            value = self.get(name)
            if isinstance(value, list):
                out[name] = list(value)
            elif isinstance(value, bool):
                out[name] = bool(value)
            else:
                out[name] = value
        out["source"] = str(self.path)
        return out
