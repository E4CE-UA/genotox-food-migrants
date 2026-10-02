"""Loading of the EFSA dataset and row count versus unique substances.

The working file is data/efsa_estructuras.csv, which src/resolver.py produces
from data/genotoxicity_cas_cid.xlsx by adding cid/smiles/inchikey.
A row is NOT a substance: it is a genotoxicity conclusion of an EFSA
output (OutputID) about a substance, so there are several rows per
substance and the same substance can appear with different conclusions.
"""

import pandas as pd

from . import paths

MINIMUM_COLUMNS = ["CleanSubstanceName", "Genotoxicity"]


def load_efsa(path=None) -> pd.DataFrame:
    """Reads efsa_estructuras.csv; if it does not exist, falls back to the original Excel without structure."""
    path = paths.EFSA_CSV if path is None else path
    if path.exists():
        df = pd.read_csv(path)
    elif paths.EFSA_XLSX.exists():
        df = pd.read_excel(paths.EFSA_XLSX)
        for c in ("cid", "smiles", "inchikey", "formula", "n_fragmentos"):
            df[c] = pd.NA
    else:
        raise FileNotFoundError(
            f"Cannot find either {paths.relative(paths.EFSA_CSV)} or "
            f"{paths.relative(paths.EFSA_XLSX)}. Run src/resolver.py first."
        )
    missing = [c for c in MINIMUM_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in the input file: {missing}")
    df["has_structure"] = df["smiles"].notna() if "smiles" in df else False
    return df


def load_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Rows versus unique substances, with and without resolved structure."""
    with_structure = df[df["has_structure"]]
    rows = [
        ("rows (study conclusions x substance)", len(df)),
        ("unique EFSA outputs (OutputID)", df["OutputID"].nunique() if "OutputID" in df else 0),
        ("unique substance names", df["CleanSubstanceName"].nunique()),
        ("rows with resolved structure", len(with_structure)),
        ("rows without structure", int((~df["has_structure"]).sum())),
        ("unique InChIKeys", with_structure["inchikey"].nunique() if "inchikey" in with_structure else 0),
        ("non-null CAS", int(df["CAS"].notna().sum()) if "CAS" in df else 0),
    ]
    return pd.DataFrame(rows, columns=["quantity", "n"])


def with_structure_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["has_structure"]].copy()
