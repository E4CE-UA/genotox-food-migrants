"""Optional merge with external mutagenicity sources.

The usual sources (Hansen Ames benchmark, ISSSTY/ISSCAN, EURL ECVAM) are not
downloaded here: each one has its own license and endpoint. It is enough to leave
a CSV per source in data/external/ with these columns:

    smiles,y,source        y in {0,1}; source is free text (e.g. hansen_ames)

and the module reads them all, standardizes the structures with the same pipeline
used for EFSA and merges by standardized InChIKey. Without that folder the notebook
still works fine with only EFSA.

Important note about the endpoint: Ames is bacterial mutagenicity, while
EFSA's conclusion aggregates several in vitro and in vivo assays. Merging them
increases n at the cost of mixing endpoints, and that is why the source column
is kept and the merge is optional and reversible.
"""

import pandas as pd

from . import chemistry, paths

DIR_EXTERNAL = paths.DATA / "external"
COLUMNS = ("smiles", "y", "source")


def catalog() -> pd.DataFrame:
    """Which files are in data/external and how many rows each one brings."""
    if not DIR_EXTERNAL.exists():
        return pd.DataFrame(columns=["file", "rows", "status"])
    rows = []
    for f in sorted(DIR_EXTERNAL.glob("*.csv")):
        try:
            d = pd.read_csv(f)
            missing = [c for c in COLUMNS if c not in d.columns]
            status = "ok" if not missing else f"missing columns: {missing}"
            rows.append({"file": f.name, "rows": len(d), "status": status})
        except Exception as exc:
            rows.append({"file": f.name, "rows": 0, "status": f"unreadable: {exc}"})
    return pd.DataFrame(rows)


def load(cache_path=None) -> pd.DataFrame:
    """Concatenates and standardizes the readable external sources."""
    cat = catalog()
    valid = cat[cat["status"] == "ok"]["file"] if len(cat) else []
    chunks = []
    for name in valid:
        d = pd.read_csv(DIR_EXTERNAL / name)[list(COLUMNS)]
        chunks.append(d)
    if not chunks:
        return pd.DataFrame(columns=[*COLUMNS, "inchikey_std", "smiles_std"])
    ext = pd.concat(chunks, ignore_index=True)
    ext["y"] = pd.to_numeric(ext["y"], errors="coerce")
    ext = ext[ext["y"].isin([0, 1])]
    return chemistry.standardize_table(ext, cache_path=cache_path)


def merge(base: pd.DataFrame, ext: pd.DataFrame,
          key: str = "inchikey_std", base_priority: bool = True) -> tuple:
    """Joins EFSA with the external ones by standardized InChIKey.

    Returns (merged table, conflicts table). A conflict is an
    InChIKey with different labels between sources; if base_priority,
    EFSA wins and the conflict is recorded but does not alter y.
    """
    base = base.copy()
    base["source"] = base.get("source", "efsa")
    cols = [key, "y", "source", "smiles_std"]
    if ext.empty:
        return base, pd.DataFrame(columns=[key, "y_base", "y_external", "source"])

    e = ext[[c for c in cols if c in ext.columns]].dropna(subset=[key, "y"])
    e = e.drop_duplicates(subset=[key, "y", "source"])

    common = e.merge(base[[key, "y"]].dropna(), on=key, suffixes=("_external", "_base"))
    conflicts = common[common["y_external"] != common["y_base"]]

    new = e[~e[key].isin(base[key])].drop_duplicates(subset=key)
    merged = pd.concat([base, new], ignore_index=True)
    return merged, conflicts.reset_index(drop=True)
