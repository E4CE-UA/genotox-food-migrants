"""Accounting of where the structures came from.

EFSA publishes conclusions about substance NAMES and CAS numbers. A model needs
structures, and that gap is the most laborious and least visible part of the
project: it was closed by src/resolver.py against the PubChem PUG REST API.

This module does not call any API. It reads what the resolver left behind
(the CSV and the JSON caches in cache/) and reports, with numbers, how many
structures came from each route and how many were lost. The notebook shows
this instead of starting from a table of SMILES as if it had always existed.
"""

import json

import pandas as pd

from . import paths

# Routes tried per row, in the order resolver.py tries them. The keys are the
# literal values resolver.py writes into `fuente_cid` and are Spanish because
# that script predates the rest of the project; they are not renamed here.
ROUTE_LABELS = {
    "excel": "CID already in the EFSA file",
    "nombre": "name -> CID (PubChem name lookup)",
}
# There is no CAS -> CID route. resolver.py never queries PubChem by CAS, and
# the 3191 unresolved rows carry no CAS at all, so adding one would recover
# nothing: those rows are name-only and the name lookup already failed on them.

# Cache filenames written by resolver.py, same reasoning.
CACHES = {
    "cas2cid": "CAS -> CID",
    "nombre2cid": "name -> CID",
    "cid_propiedades": "CID -> SMILES/InChIKey (batches of 100)",
}


def resolution_routes(df: pd.DataFrame) -> pd.DataFrame:
    """How many rows got their CID from each route, and how many got none."""
    col = "fuente_cid"
    if col not in df:
        return pd.DataFrame(columns=["route", "n", "pct"])
    counts = df[col].value_counts(dropna=False)
    rows = []
    for key, label in ROUTE_LABELS.items():
        if key in counts:
            rows.append((label, int(counts[key])))
    unresolved = int(counts.get(float("nan"), 0)) if counts.index.isna().any() else 0
    if unresolved == 0:
        unresolved = int(df[col].isna().sum())
    rows.append(("no structure resolved", unresolved))
    out = pd.DataFrame(rows, columns=["route", "n"])
    out["pct"] = (100 * out["n"] / len(df)).round(1)
    return out


def api_effort() -> pd.DataFrame:
    """Reads the resolver caches: how many API lookups, how many came back empty.

    Each cache entry is one question asked of PubChem that never has to be
    asked again. Entries whose value is null are questions PubChem answered
    with "no match" -- they are cached too, so a rerun does not repeat them.
    """
    rows = []
    for name, description in CACHES.items():
        f = paths.CACHE / f"{name}.json"
        if not f.exists():
            rows.append((description, 0, 0, None))
            continue
        d = json.loads(f.read_text())
        empty = sum(1 for v in d.values() if v is None)
        hit_rate = round(100 * (len(d) - empty) / len(d), 1) if len(d) else None
        rows.append((description, len(d), empty, hit_rate))
    return pd.DataFrame(rows, columns=["lookup", "cached_queries", "no_match", "hit_rate_pct"])


def funnel(raw: pd.DataFrame, with_structure: pd.DataFrame,
           informative: int, substances: int, modelable: int) -> pd.DataFrame:
    """The full path from published conclusion to trainable example.

    Returned as a table so the figure and the prose cannot disagree: both
    read these numbers.
    """
    rows = [
        ("EFSA conclusions (rows)", len(raw),
         "one row = one output's conclusion about one substance"),
        ("with a resolved structure", len(with_structure),
         "the rest have no CID in the file and no PubChem match"),
        ("with an informative label", informative,
         "only Positive and Negative; the other five values are not evidence"),
        ("unique substances after aggregation", substances,
         "several rows per substance collapse under the aggregation rule"),
        ("modelable substances", modelable,
         "standardized, deduplicated by standardized InChIKey"),
    ]
    out = pd.DataFrame(rows, columns=["stage", "n", "what is lost here"])
    out["retained_pct"] = (100 * out["n"] / len(raw)).round(1)
    return out
