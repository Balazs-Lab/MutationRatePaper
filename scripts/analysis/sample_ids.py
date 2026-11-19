"""Pipeline vs analysis YAML sample ID spellings.

Pipeline configs often use ``LP###-NL43-…`` and ``LP###-ePCR-…`` while Step 09/10
outputs use ``NL43-LP###-…`` in CSV/JSON. Analysis YAML may list either form.
Downstream filters must accept both.
"""

from __future__ import annotations

from typing import Iterable, List, Set


def pipeline_output_sample_id(config_sample_id: str) -> str:
    """Map YAML / pipeline-config style IDs to mutation_analysis table names."""
    parts = config_sample_id.split("-")
    if len(parts) >= 3 and parts[0].startswith("LP") and parts[1] == "NL43":
        return "-".join(["NL43", parts[0]] + parts[2:])
    if len(parts) >= 3 and parts[0].startswith("LP") and parts[1] == "ePCR":
        return "-".join(["NL43", parts[0], "ePCR"] + parts[2:])
    return config_sample_id


def expand_sample_ids_for_matching_tables(sample_ids: Iterable[str]) -> List[str]:
    """Sorted unique IDs for ``.isin(...)`` against pipeline mutation tables."""
    out: Set[str] = set()
    for s in sample_ids:
        if not s:
            continue
        out.add(s)
        out.add(pipeline_output_sample_id(s))
    return sorted(out)


def canonical_pipeline_sample_ids(sample_ids: Iterable[str]) -> List[str]:
    """One pipeline-style ID per physical sample after ``expand_sample_ids_for_matching_tables``.

    ``expand`` can retain both ``LP249-NL43-…`` and ``NL43-LP249-…``; this collapses
    them so grouping, counts, and ``.isin`` lists are not duplicated.
    """
    return sorted({pipeline_output_sample_id(s) for s in sample_ids if s})
