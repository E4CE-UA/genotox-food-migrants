"""Read-only reading of the GC-MS candidate database.

The notebook does not rerun any identification: it opens the database in
ro mode, exports the two tables it needs to data/*.parquet and from there
it works only with the parquet files. The connection is always opened with mode=ro so that
a code error cannot write to the 700 MB database.

Expected path in the GCMS_DB environment variable. If it is not defined or the
file does not exist, available() returns False and the notebook continues without the
integration cell instead of failing.
"""

import os
import sqlite3
from pathlib import Path

import pandas as pd

from . import paths

# The table and column names on the left of each AS belong to the upstream
# database and must not be touched. The aliases are what the rest of this
# project sees, so they follow the English naming used everywhere else.
SQL_CANDIDATES = """
SELECT c.id_feature        AS feature_id,
       c.rank              AS rank,
       c.inchikey          AS inchikey,
       c.nombre            AS name,
       c.evidencia         AS evidence,
       c.n_muestras_cand   AS n_samples_cand,
       r.smiles            AS smiles,
       r.cas_oficial       AS cas_official,
       r.efsa_genotoxico   AS efsa_genotoxic,
       t.alerta_genotox    AS genotox_alert,
       t.ttc_clase         AS ttc_class,
       t.ttc_ug_dia        AS ttc_ug_day,
       t.cramer_clase      AS cramer_class
FROM feature_candidatos_v3 c
LEFT JOIN compuestos_ref r ON r.inchikey = c.inchikey
LEFT JOIN toxicidad_estructura_v3 t ON t.inchikey = c.inchikey
"""

SQL_FEATURES = """
SELECT id_feature       AS feature_id,
       n_muestras       AS n_samples,
       rt_medio         AS rt_mean,
       ri_medio         AS ri_mean,
       inchikey_id      AS inchikey_id,
       nombre_id        AS name_id,
       nivel_msi        AS msi_level,
       margen           AS margin,
       cobertura        AS coverage,
       score_espectral  AS spectral_score,
       delta_ri         AS delta_ri,
       delta_airi       AS delta_airi,
       mplus_confirma   AS mplus_confirms,
       frag_confirma    AS frag_confirms,
       frag_contradice  AS frag_contradicts,
       airi_confirma    AS airi_confirms,
       iso_confirma     AS iso_confirms,
       es_surrogate     AS is_surrogate,
       es_artefacto     AS is_artifact,
       es_fantasma      AS is_ghost
FROM features_v3
"""

EXCLUSION_FLAGS = ("is_surrogate", "is_artifact", "is_ghost")


def db_path():
    v = os.environ.get("GCMS_DB")
    return Path(v).expanduser() if v else None


def available() -> bool:
    r = db_path()
    return bool(r and r.exists())


def unavailable_reason() -> str:
    r = db_path()
    if r is None:
        return "GCMS_DB is not defined. export GCMS_DB=/path/to/your_export.sqlite"
    return f"GCMS_DB points to {r.name}, which does not exist."


def connect(path=None):
    path = path or db_path()
    return sqlite3.connect(f"file:{Path(path).resolve()}?mode=ro", uri=True)


def export(path=None) -> dict:
    """Exports candidates and features to data/*.parquet. Returns the shapes."""
    with connect(path) as con:
        cand = pd.read_sql_query(SQL_CANDIDATES, con)
        feat = pd.read_sql_query(SQL_FEATURES, con)
    paths.DATA.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(paths.GCMS_CANDIDATES, index=False)
    feat.to_parquet(paths.GCMS_FEATURES, index=False)
    return {"candidates": cand.shape, "features": feat.shape}


def load_export():
    """Reads the already exported parquet files and filters surrogates, artifacts and phantoms."""
    if not (paths.GCMS_CANDIDATES.exists() and paths.GCMS_FEATURES.exists()):
        raise FileNotFoundError(
            "Missing gcms parquet files. Run gcms.export() with GCMS_DB defined."
        )
    cand = pd.read_parquet(paths.GCMS_CANDIDATES)
    feat = pd.read_parquet(paths.GCMS_FEATURES)
    n0 = len(feat)
    for b in EXCLUSION_FLAGS:
        if b in feat.columns:
            feat = feat[feat[b] != 1]
    cand = cand[cand["feature_id"].isin(feat["feature_id"])]
    return cand, feat, {"features_dropped": n0 - len(feat)}


# Patterns for a column that would allow a batch / acquisition-campaign split.
# Spanish and English, because the database is Spanish and a future export
# may not be.
BATCH_PATTERNS = ("fecha", "date", "lote", "batch", "campana", "campaign",
                  "secuencia", "sequence", "run", "tanda", "adquisicion",
                  "acquisition", "muestra", "sample")


def batch_columns(path=None) -> pd.DataFrame:
    """Looks for a column that would support a split by acquisition batch.

    The two tables this project reads (feature_candidatos_v3, features_v3)
    carry no date: `n_muestras` counts samples but does not say when they ran.
    A split by acquisition campaign is the leakage control this notebook is
    missing -- features from the same sequence share column state, tuning and
    contamination, so a random split over features is optimistic in a way the
    scaffold split does not fix.

    Rather than guess the schema, this probes the live database and reports
    every table/column whose name could plausibly carry that information.
    Returns an empty frame if the database has nothing usable.
    """
    with connect(path) as con:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type IN ('table','view')", con
        )["name"].tolist()
        rows = []
        for t in tables:
            try:
                info = pd.read_sql_query(f'PRAGMA table_info("{t}")', con)
            except Exception:
                continue
            for c in info["name"]:
                if any(p in c.lower() for p in BATCH_PATTERNS):
                    rows.append((t, c))
    return pd.DataFrame(rows, columns=["table", "column"])
