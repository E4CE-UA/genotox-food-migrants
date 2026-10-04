"""Project paths. Everything relative to the repository root.

No module should write an absolute path: they are all derived from ROOT,
which is computed from the location of this file.
"""

import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = Path(__file__).resolve().parent
DATA = ROOT / "data" if (ROOT / "pyproject.toml").exists() else PACKAGE / "_data"
VALIDATION = ROOT / "docs" if (ROOT / "pyproject.toml").exists() else PACKAGE / "_validation"

# In a clone, ROOT is the repository (writable, carries pyproject.toml) and the
# cache lives there, gitignored. Installed into site-packages there is no repo
# and ROOT may be read-only, so the cache goes to a writable temp directory
# instead. The cache only ever holds recomputable RDKit standardisation and
# t-SNE results, so its location does not affect any reported number.
_IN_REPO = (ROOT / "pyproject.toml").exists()
CACHE = (ROOT / "cache") if _IN_REPO else Path(tempfile.gettempdir()) / "genotox_cache"
NOTEBOOKS = ROOT / "notebooks"

# Inputs
EFSA_XLSX = DATA / "genotoxicity_cas_cid.xlsx"   # EFSA OpenFoodTox + CAS/CID
EFSA_CSV = DATA / "efsa_estructuras.csv"         # output of src/resolver.py

# Exports from the GC-MS candidate database (generated with src/gcms.py)
GCMS_CANDIDATES = DATA / "gcms_candidates.parquet"
GCMS_FEATURES = DATA / "gcms_features.parquet"

# Public MetaboLights features and their candidate isomer sets, precomputed
# with src/public_features.py:bundle(). These ship with the repository because
# building them cold costs a PubChem formula search per feature - half an hour
# for the default study - and a reader should not have to wait for that to see
# the section run. Both files are derived from public data and the function
# that regenerates them is in the repository.
PUBLIC_CANDIDATES = DATA / "public_candidates.parquet"
PUBLIC_FEATURES = DATA / "public_features.parquet"

# Notebook computation caches
CACHE_STANDARDIZATION = CACHE / "standardization_rdkit.parquet"


def ensure_cache() -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE


def relative(p) -> str:
    """Represents a path relative to ROOT (to avoid printing absolute paths)."""
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return p.name
