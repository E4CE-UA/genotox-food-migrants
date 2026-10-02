"""Retention index as public evidence for ordering candidates.

A feature's measured retention is one number; ranking its candidate isomers
needs a number per candidate. That is what makes a structure-to-index
predictor the piece that turns public data into ordering evidence: predict an
index for every candidate, and rank by how far it falls from the index the
depositors measured.

Harvesting the training set has one trap that has to be handled rather than
assumed away. The MAF column `retention_time` holds minutes in most studies
and a Kovats index in some, under the same name, and magnitude alone does not
separate them: a forty-minute run reported in seconds lands squarely in the
range a Kovats index occupies. So studies are calibrated against those that
publish an explicit index column, using the structures they have in common -
a study whose shared structures agree one-to-one with the explicit-index
consensus is reporting an index, and a study that comes out at 0.4x is not,
whatever its column is called.

Which studies are allowed to *be* the reference needs its own test, because
the column name is not enough either. A column called `RI` is a retention
index in some studies and relative isotope abundance in others - MTBLS1866
carries `RI`, `RIA 1` and `RIA 2` side by side, and its `RI` spans 0 to 27117
with a median of 0.80, which no index does. Name-matching alone admitted that
column as a reference. So a reference column must additionally rise with
molecular size: retention on a nonpolar column is dominated by boiling point,
so a genuine index correlates strongly with molecular weight across a study's
structures (the four surviving references sit at Spearman 0.85-0.91), while an
abundance column carries no such association. This is a test of what the
column measures, not of its units, so it filters references without being
able to separate an index from a retention time - that is what the ratio
against the consensus is for.

The floor for any predictor here is not zero. The same structure measured in
two studies differs, because retention index depends on the stationary phase
and only approximately transfers between phases, and that between-study
dispersion is reported alongside the model error. A model whose error sits at
the dispersion has learned as much as this reference can teach; reporting it
against zero error would be the wrong comparison.
"""

import re
from functools import lru_cache
import statistics as st
from collections import defaultdict

import numpy as np
import pandas as pd

from . import chemistry, public_features, splitting

INDEX_COLUMNS = re.compile(r"(?i)^ri$|kovats|retention.?index")
PLAUSIBLE = (200.0, 5000.0)      # a Kovats index cannot be outside this
RATIO_TOLERANCE = 0.10           # how far from 1.00 a study may calibrate
MIN_OVERLAP = 3                  # shared structures needed to judge a study
REF_MIN_N = 8                    # structures a column needs before it can be a reference
REF_MIN_RHO = 0.70               # how strongly a reference column must rise with size


def harvest(accessions=None, cache_dir=None) -> pd.DataFrame:
    """Candidate structure-index pairs from public GC-MS MAFs.

    Returns every numeric retention value in the plausible index range, tagged
    with the column it came from. Nothing here is trusted yet: `calibrate`
    decides which of these are indices.
    """
    accs = accessions if accessions is not None else public_features.gc_studies()
    rows = []
    for acc in accs:
        try:
            files = public_features.maf_files(acc)
        except Exception:
            continue
        for maf in files:
            try:
                df = public_features.load_maf(acc, maf, cache_dir=cache_dir)
            except Exception:
                continue
            cols = [c for c in df.columns if INDEX_COLUMNS.search(c)]
            struct = df.get("inchi")
            if struct is None:
                struct = df.get("smiles")
            if struct is None:
                continue
            fallback = df["retention_time"] if "retention_time" in df else None
            for col in cols + (["retention_time"] if fallback is not None else []):
                vals = pd.to_numeric(df[col].astype(str).str.strip(), errors="coerce")
                ok = vals.between(*PLAUSIBLE) & struct.notna() & struct.astype(str).str.strip().ne("")
                for s, v in zip(struct[ok].astype(str).str.strip(), vals[ok]):
                    rows.append({"study": acc, "maf": maf, "column": col,
                                 "structure": s, "value": float(v)})
    out = pd.DataFrame(rows)
    if len(out):
        # One value per structure and column within a study: replicate rows in
        # a MAF are the same measurement reported per sample, not evidence.
        out = (out.groupby(["study", "column", "structure"], as_index=False)["value"]
                  .median())
    return out



@lru_cache(maxsize=None)
def _mol_weight(structure: str) -> float:
    """Molecular weight from an InChI, NaN when it does not parse."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    mol = Chem.MolFromInchi(structure) if isinstance(structure, str) else None
    return float(Descriptors.MolWt(mol)) if mol is not None else float("nan")

def _size_rho(grp: pd.DataFrame) -> float:
    """Spearman correlation between a column's values and molecular weight.

    The test that separates a retention column from an abundance column that
    happens to be called `RI`. Returns NaN when too few structures parse.
    """
    from scipy.stats import spearmanr

    w = [_mol_weight(s) for s in grp["structure"]]
    ok = [(a, b) for a, b in zip(grp["value"], w) if b == b and a == a]
    if len(ok) < REF_MIN_N:
        return float("nan")
    return float(spearmanr([a for a, _ in ok], [b for _, b in ok]).statistic)


def _is_reference(column: str, grp: pd.DataFrame) -> bool:
    """A reference column is named like an index AND rises with molecular size."""
    if not INDEX_COLUMNS.search(column):
        return False
    rho = _size_rho(grp)
    return rho == rho and rho >= REF_MIN_RHO


def calibrate(pairs: pd.DataFrame) -> pd.DataFrame:
    """Which studies report an index, judged on structures shared with those that say so."""
    is_ref = {(s, c): _is_reference(c, g) for (s, c), g in pairs.groupby(["study", "column"])}
    explicit = pairs[[is_ref[(s, c)] for s, c in zip(pairs["study"], pairs["column"])]]
    ref = explicit.groupby("structure")["value"].median().to_dict()
    rows = []
    for (study, column), grp in pairs.groupby(["study", "column"]):
        if is_ref[(study, column)]:
            rows.append({"study": study, "column": column, "n": len(grp), "overlap": 0,
                         "ratio": 1.0, "accepted": True,
                         "basis": f"explicit index column, rises with size "
                                  f"(rho {_size_rho(grp):.2f})"})
            continue
        if INDEX_COLUMNS.search(column):
            rows.append({"study": study, "column": column, "n": len(grp), "overlap": 0,
                         "ratio": float("nan"), "accepted": False,
                         "basis": "named like an index but does not rise with molecular "
                                  "size, so it is not a retention column"})
            continue
        ratios = [v / ref[s] for s, v in zip(grp["structure"], grp["value"])
                  if s in ref and ref[s] > 0]
        if len(ratios) < MIN_OVERLAP:
            rows.append({"study": study, "column": column, "n": len(grp),
                         "overlap": len(ratios), "ratio": float("nan"),
                         "accepted": False, "basis": "not enough shared structures to judge"})
            continue
        r = st.median(ratios)
        rows.append({"study": study, "column": column, "n": len(grp), "overlap": len(ratios),
                     "ratio": round(r, 3),
                     "accepted": abs(r - 1.0) <= RATIO_TOLERANCE,
                     "basis": "calibrated against explicit-index studies"})
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def reference(pairs: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    """Per-structure index from the accepted studies, with its between-study spread."""
    keep = {(r.study, r.column) for r in calibration.itertuples() if r.accepted}
    good = pairs[[(s, c) in keep for s, c in zip(pairs["study"], pairs["column"])]]
    grp = good.groupby("structure")["value"]
    out = pd.DataFrame({"ri": grp.median(), "n_studies": good.groupby("structure")["study"].nunique(),
                        "spread": grp.max() - grp.min(),
                        "sd": grp.std()}).reset_index()
    return out.sort_values("n_studies", ascending=False)


def between_study_floor(ref: pd.DataFrame) -> dict:
    """The dispersion a predictor cannot beat, from structures seen twice or more."""
    multi = ref[ref["n_studies"] >= 2]
    return {"n_structures": int(len(multi)),
            "median_spread": float(multi["spread"].median()),
            "p90_spread": float(multi["spread"].quantile(0.90)),
            "median_sd": float(multi["sd"].median())}


def _smiles_of(structure: str):
    """The reference keys are InChI in some studies and SMILES in others."""
    from rdkit import Chem
    s = structure.strip()
    mol = Chem.MolFromInchi(s) if s.startswith("InChI=") else Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(mol) if mol is not None else None


def prepare(ref: pd.DataFrame) -> pd.DataFrame:
    """Reference rows that RDKit can parse, with canonical SMILES and scaffold."""
    smi = [_smiles_of(s) for s in ref["structure"]]
    out = ref.assign(smiles=smi).dropna(subset=["smiles"]).copy()
    out["scaffold"] = [chemistry.scaffold(s) for s in out["smiles"]]
    return out


def fit(prepared: pd.DataFrame, seeds=(0, 1, 2), frac_test: float = 0.2):
    """Predict index from ECFP4, evaluated on scaffold-disjoint test sets.

    The split is the same randomized scaffold split the hazard model uses, for
    the same reason: with congener families a random split lets the model see a
    sibling of every test compound and the error comes out flattering.
    """
    from lightgbm import LGBMRegressor

    X = chemistry.matrix(chemistry.fingerprints(prepared["smiles"]))
    y = prepared["ri"].to_numpy(dtype=float)
    rows, models = [], []
    for seed in seeds:
        tr, te = splitting.split_scaffold(prepared["scaffold"].tolist(), seed, frac_test)
        model = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                              min_child_samples=5, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.6, random_state=seed, n_jobs=-1,
                              verbose=-1)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        rows.append({"seed": seed, "n_train": len(tr), "n_test": len(te),
                     "mae": float(np.mean(np.abs(pred - y[te]))),
                     "rmse": float(np.sqrt(np.mean((pred - y[te]) ** 2))),
                     "spearman": float(pd.Series(pred).corr(pd.Series(y[te]), method="spearman"))})
        models.append(model)
    return pd.DataFrame(rows), models


def fit_full(prepared: pd.DataFrame, seed: int = 0):
    """Refit on every reference structure, for predicting candidates."""
    from lightgbm import LGBMRegressor

    X = chemistry.matrix(chemistry.fingerprints(prepared["smiles"]))
    model = LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                          min_child_samples=5, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.6, random_state=seed, n_jobs=-1, verbose=-1)
    model.fit(X, prepared["ri"].to_numpy(dtype=float))
    return model


def predict_index(model, smiles) -> np.ndarray:
    return model.predict(chemistry.matrix(chemistry.fingerprints(list(smiles))))


def retention_weights(predicted, measured: float, sigma: float) -> np.ndarray:
    """Candidate weights from |predicted - measured|, as a Gaussian kernel.

    `sigma` is the predictor's own validation error, so the weights say how
    compatible each candidate is with the measurement given how well the
    predictor actually predicts - not how close it looks on an arbitrary
    scale. A predictor no better than the between-study dispersion produces a
    kernel wide enough to be close to uniform, which is the honest outcome
    rather than a hidden failure.
    """
    d = np.asarray(predicted, dtype=float) - float(measured)
    w = np.exp(-0.5 * (d / float(sigma)) ** 2)
    total = w.sum()
    return w / total if total > 0 else np.full(len(w), 1.0 / max(len(w), 1))
