"""Public GC-MS feature tables from MetaboLights.

A MetaboLights study publishes a metabolite assignment file (MAF) per assay:
one row per detected feature, with the identity the depositors assigned to it
and whatever evidence they reported for that assignment. That is the shape this
notebook needs downstream - a feature with a proposed identity is exactly the
object whose identity is uncertain - and it is public, which the instrument
tables this project started from are not.

Column names are kept as they arrive. `database_identifier`,
`metabolite_identification` and `mass_to_charge` are MAF vocabulary, not names
chosen here, and renaming them would hide where the numbers came from.

Two things a caller must not assume:

* `retention_time` is not in comparable units across studies. Some studies
  report minutes on their own column and program, some report a Kovats
  retention index, and the column name is the same in both cases. Deciding
  which is which is `retention.py`'s job, by calibrating against the studies
  that publish an explicit index column; do not infer it from the magnitude
  alone. Nor from the column name: MTBLS1866 carries a column called `RI`
  that is not a retention index at all - it sits beside `RIA 1` and `RIA 2`
  and spans 0 to 27117 with a median of 0.80, which is relative abundance.
  That study's index is in `retention_time` (300-1928).
* coverage varies per study. `mass_to_charge`, `smiles` and `reliability` are
  each empty in some studies, so which sections can run at all is a property
  of the chosen accession, not of the pipeline. `coverage` reports it.
"""

import io
import json
import re
import urllib.request

import pandas as pd

from . import chemistry as _chem

FTP = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"
WS = "https://www.ebi.ac.uk/metabolights/ws/studies"

# Screened from the 239 public GC-* studies: those whose first MAF carries
# m/z, retention and structure for essentially every feature. n_features is
# what the MAF had when screened; it is a label for the dropdown, not a
# contract - a depositor can revise a study.
# The note is what the study can support, measured rather than assumed: only
# two of the seven publish an identification level at all, and the one that
# publishes it for every feature gives no formula for the features it calls
# unknown - so a formula and a stated identity level are nearly complementary
# in public GC-MS deposits.
SCREENED = [
    ("FCM2018", 33, "food contact migrants: MSI 1 for 22 features confirmed with standards, MSI 2 for 11 library-only (Galman Graino et al. 2018)"),
    ("MTBLS1866", 251, "formula for every feature, no identification level published"),
    ("MTBLS627", 154, "MSI level for every feature: 105 at level 1, 49 unknowns with no formula"),
    ("MTBLS522", 65, "formula for every feature, no level"),
    ("MTBLS8966", 63, "formula for every feature, no level"),
    ("MTBLS519", 51, "level on 24 % of features, and its values include a 5, which is not MSI"),
    ("MTBLS4497", 44, "formula for every feature, no level"),
    ("MTBLS892", 37, "formula and structure for every feature, no level"),
]

MAF_COLUMNS = ["database_identifier", "chemical_formula", "smiles", "inchi",
               "metabolite_identification", "mass_to_charge", "retention_time",
               "reliability", "database", "search_engine_score"]


def _read(url: str, limit: int = 8_000_000, timeout: int = 60) -> str:
    with urllib.request.urlopen(urllib.request.Request(url), timeout=timeout) as r:
        return r.read(limit).decode("utf-8", "replace")


def gc_studies(cache_path=None) -> list:
    """Accessions of public studies whose technology string mentions GC."""
    if cache_path is not None and cache_path.exists():
        return json.loads(cache_path.read_text())
    rows = json.loads(_read(f"{WS}/technology?technology=GC-MS"))
    accs = sorted({d["accession"] for d in rows if "GC" in (d.get("technology") or "")})
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(accs))
    return accs


def maf_files(acc: str) -> list:
    """MAF filenames published by a study, from its public FTP directory."""
    return re.findall(r'href="(m_[^"]+\.tsv)"', _read(f"{FTP}/{acc}/", 200_000))


def _cache(cache_dir, name: str):
    """The default cache location, so a second pass does not re-download.

    `cache/` is not versioned, so nothing here travels with the repository;
    it only keeps a rerun on the same machine from paying the downloads twice.
    """
    from . import paths

    d = (paths.CACHE / name) if cache_dir is None else cache_dir
    d.mkdir(parents=True, exist_ok=True)
    return d


def _maf_frame(text: str, acc: str, maf: str) -> pd.DataFrame:
    """Parse a MAF's text into the frame the callers expect."""
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    df.insert(0, "study", acc)
    df.insert(1, "maf", maf)
    return df


def load_maf(acc: str, maf: str = None, cache_dir=None) -> pd.DataFrame:
    """One study's feature table, with upstream column names untouched.

    `maf=None` takes the study's first MAF, which is the only choice for the
    single-assay studies and an arbitrary one otherwise; pass the filename
    explicitly when a study has several assays.

    A copy of each screened study's first MAF ships under `data/metabolights`
    so the notebook runs with no network at all. That bundled copy is preferred
    over both the download and the writable `cache/` mirror, and when it answers
    the study's FTP directory is never listed - so an offline run makes no
    request even to resolve the filename.
    """
    from . import paths

    bundled = paths.DATA / "metabolights"
    if maf is None and bundled.exists():
        hits = sorted(bundled.glob(f"{acc}__*.tsv"))
        if hits:
            return _maf_frame(hits[0].read_text(), acc,
                              hits[0].name.split("__", 1)[1])

    if maf is None:
        files = maf_files(acc)
        if not files:
            raise ValueError(f"{acc} publishes no MAF")
        maf = files[0]

    shipped = bundled / f"{acc}__{maf}"
    if shipped.exists():
        return _maf_frame(shipped.read_text(), acc, maf)

    local = _cache(cache_dir, "metabolights") / f"{acc}__{maf}"
    if local.exists():
        text = local.read_text()
    else:
        text = _read(f"{FTP}/{acc}/{maf}")
        local.write_text(text)
    return _maf_frame(text, acc, maf)


def coverage(df: pd.DataFrame, columns=None) -> pd.DataFrame:
    """Populated fraction per column: what this study can and cannot support."""
    cols = [c for c in (columns or MAF_COLUMNS) if c in df.columns]
    n = max(len(df), 1)
    # A MAF read with dtype=str turns empty cells into NaN, and str(NaN) is
    # the non-empty string "nan": counting populated cells without the notna
    # guard reports full coverage for an empty column.
    def _filled(col):
        s = df[col]
        return int((s.notna() & s.astype(str).str.strip().ne("")).sum())
    out = [{"column": c, "populated": _filled(c),
            "fraction": round(_filled(c) / n, 3)} for c in cols]
    missing = [{"column": c, "populated": 0, "fraction": 0.0}
               for c in (columns or MAF_COLUMNS) if c not in df.columns]
    return pd.DataFrame(out + missing)


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    """A MAF numeric column as floats, with unparseable entries as NaN."""
    return pd.to_numeric(df[column].astype(str).str.strip(), errors="coerce")

# --- assembling the two tables the integration consumes -------------------

MSI_SCALE = (1, 2, 3, 4)   # Sumner et al. 2007; a value outside it is not MSI


def _msi_level(raw) -> float:
    """MSI level from a `reliability` cell, NaN when it is not an MSI level.

    Depositors write this column several ways: bare integers, and the prefixed
    form `MSI:1` used in MTBLS627. Both parse. What does not parse is a digit
    outside the MSI range - MTBLS519 reports 0, 3, 4 and *5* in this column,
    and a 5 does not exist on a four-tier scale, so that study is reporting
    some other scheme (Schymanski's five tiers are the likely candidate). Those
    cells are left as NaN rather than coerced, because reading a 5 as an MSI
    level would silently import a different scale's meaning.
    """
    m = re.search(r"\d+", str(raw))
    if not m:
        return float("nan")
    v = int(m.group())
    return float(v) if v in MSI_SCALE else float("nan")


def _inchikey(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) and smiles.strip() else None
    return Chem.MolToInchiKey(mol) if mol is not None else None


def build(accession: str, n_candidates: int = 20, seed: int = 0,
          cache_dir=None, labels: pd.DataFrame = None, bundled: bool = True):
    """A public MAF as the candidates/features pair the integration consumes.

    Returns (candidates, features, info), same column contract as the export,
    with `info["is_fixture"] = False`. Four decisions are worth stating because
    each one could have gone differently:

    **`evidence` is absent, not estimated.** A MAF publishes one assigned
    identity per feature and no score per alternative candidate, so there is
    nothing to rank the isomers of a formula by. The column is present and
    all-NaN so the contract holds, which leaves `uniform` as the only weighting
    scheme the public source can support. That is a property of what
    depositors publish, not a fallback chosen here, and section 13 says so on
    screen.

    **`rank` is enumeration order.** With no evidence there is no ranking; the
    column is filled so downstream code has it, and any use of it as a
    preference order would be reading structure into an arbitrary sequence.

    **The gate only fires on a parsed MSI level 1.** A feature the depositors
    confirmed against an authentic standard has its weight placed on the
    structure they assigned, exactly as the export path does. Features whose
    level is absent - which is most of them, since only 2 of the 7 screened
    studies publish the column at all - are marginalized, because an
    unconfirmed assignment is a hypothesis and exempting it from weighting
    would be assuming the answer.

    **The structural alerts are computed, not published.** No MAF carries a
    genotoxicity alert column, so `genotox_alert` is derived here from the
    candidate SMILES with `chemistry.dna_reactive_alerts`, a short and
    explicitly partial list of DNA-reactive classes. It is not the
    Benigni-Bossa rulebase and section 17 says so where it uses it.

    **The assigned structure is always in the candidate set.** Sampling 20 of
    a formula's isomers would otherwise drop the depositor's own identity most
    of the time, and the gate could not match it. It is included and flagged
    `is_assigned`, which also lets a caller measure where in the set it fell.

    When the listed PubChem formula pool exceeds `n_candidates`, alternatives
    are sampled and every row carries `n_total` and `sampled`. The published
    identity is forcibly included. Equal weights therefore define a retained
    structural stress-test scenario, NOT an unbiased Monte Carlo estimate of
    the whole pool. The listed pool is not all chemically possible identities.
    """
    from . import candidates as _cand

    pug_dir = _cache(cache_dir, "pubchem_formula")
    packed = _from_bundle(accession, n_candidates) if bundled else None
    if packed is not None:
        cand, feat = packed
        if labels is not None and len(cand):
            cand = _attach_labels(cand, labels)
        # Derived from the SMILES alone, so it is recomputed on read rather
        # than stored: a bundle built before the alerts existed still gets
        # them, and the rulebase can change without rebuilding the bundle.
        cand = _chem.attach_alerts(cand)
        return cand, feat, _info(accession, cand, feat, from_bundle=True)

    maf = load_maf(accession, cache_dir=cache_dir)
    crows, frows = [], []
    for i, row in enumerate(maf.itertuples(index=False), start=1):
        fid = f"{accession}-{i:04d}"
        formula = _text(getattr(row, "chemical_formula", ""))
        if formula and not FORMULA_RE.match(formula):
            formula = ""      # charges, adducts, free text: not a formula query
        assigned = _text(getattr(row, "smiles", ""))
        level = _msi_level(getattr(row, "reliability", None))
        key_assigned = _inchikey(assigned)

        cset = pd.DataFrame(columns=["cid", "smiles", "n_total", "sampled"])
        if formula:
            cset = _cand.candidate_set(formula, n=n_candidates, seed=seed,
                                       cache_dir=pug_dir)
        pool = [(None, assigned)] if key_assigned else []
        pool += [(c, s) for c, s in zip(cset.get("cid", []), cset.get("smiles", []))
                 if _inchikey(s) != key_assigned]
        pool = pool[:max(n_candidates, 1)]

        for r, (cid, smi) in enumerate(pool, start=1):
            crows.append(dict(
                feature_id=fid, rank=r, inchikey=_inchikey(smi), name=None,
                evidence=float("nan"), smiles=smi, cid=cid,
                is_assigned=bool(r == 1 and key_assigned),
                n_total=(int(cset["n_total"].iloc[0]) if len(cset) else len(pool)),
                sampled=(bool(cset["sampled"].iloc[0]) if len(cset) else False),
                cas_official=None, efsa_genotoxic=float("nan"),
            ))

        frows.append(dict(
            feature_id=fid,
            database_identifier=getattr(row, "database_identifier", None),
            name_id=(getattr(row, "metabolite_identification", None) if level == 1 else None),
            inchikey_id=(key_assigned if level == 1 else None),
            msi_level=level, msi_raw=getattr(row, "reliability", None),
            chemical_formula=formula or None,
            rt_mean=pd.to_numeric(getattr(row, "retention_time", None), errors="coerce"),
            ri_mean=float("nan"),
            mass_to_charge=pd.to_numeric(getattr(row, "mass_to_charge", None), errors="coerce"),
            n_candidates=len(pool),
            n_isomers_total=(int(cset["n_total"].iloc[0]) if len(cset) else 0),
            sampled=(bool(cset["sampled"].iloc[0]) if len(cset) else False),
            n_samples=float("nan"), coverage=float("nan"),
        ))

    cand = pd.DataFrame(crows)
    feat = pd.DataFrame(frows)
    if labels is not None and len(cand):
        cand = _attach_labels(cand, labels)
    cand = _chem.attach_alerts(cand)
    return cand, feat, _info(accession, cand, feat, from_bundle=False)


def _info(accession: str, cand: pd.DataFrame, feat: pd.DataFrame,
          from_bundle: bool) -> dict:
    """What the section has to be able to say about where its numbers come from."""
    return {
        "is_fixture": False, "accession": accession, "from_bundle": from_bundle,
        "n_features": len(feat), "n_candidates": len(cand),
        "distinct_structures": int(cand["smiles"].nunique()) if len(cand) else 0,
        "n_confirmed": int((feat["msi_level"] == 1).sum()),
        "n_no_formula": int(feat["chemical_formula"].isna().sum()),
        "median_isomer_space": float(feat.loc[feat.n_isomers_total > 0, "n_isomers_total"].median())
        if (feat.n_isomers_total > 0).any() else float("nan"),
        "sampled_features": int(feat["sampled"].sum()),
        "has_evidence": False,
    }


def bundle(accessions=None, n_candidates: int = 20, seed: int = 0, cache_dir=None):
    """Precomputes the candidate sets for the screened studies into `data/`.

    Building a study cold costs one PubChem formula search per distinct
    formula plus SMILES for every candidate: about half an hour for the 251
    features of MTBLS1866. A reader opening the notebook should not pay that,
    so the result ships as two parquet files keyed by `study` and this function
    is how they are regenerated. Delete them and the section falls back to
    fetching live, which produces the same tables.
    """
    from . import paths

    accs = list(accessions or [a for a, _, _ in SCREENED])
    cands, feats, failed = [], [], {}
    for acc in accs:
        try:
            c, f, _ = build(acc, n_candidates=n_candidates, seed=seed,
                            cache_dir=cache_dir, bundled=False)
        except Exception as exc:
            # One study losing its downloads should not discard the studies
            # already built; the bundle covers what it covers, and a study
            # missing from it is rebuilt live when asked for.
            failed[acc] = f"{type(exc).__name__}: {exc}"
            continue
        cands.append(c.assign(study=acc, n_requested=n_candidates))
        feats.append(f.assign(study=acc, n_requested=n_candidates))
    if not cands:
        raise RuntimeError(f"no study could be built: {failed}")
    if failed:
        print("studies left out of the bundle:", failed)
    cand = pd.concat(cands, ignore_index=True)
    feat = pd.concat(feats, ignore_index=True)
    paths.DATA.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(paths.PUBLIC_CANDIDATES, index=False)
    feat.to_parquet(paths.PUBLIC_FEATURES, index=False)
    return cand, feat


def _from_bundle(accession: str, n_candidates: int):
    """The bundled tables for one study, or None when they do not cover it."""
    from . import paths

    if not (paths.PUBLIC_CANDIDATES.exists() and paths.PUBLIC_FEATURES.exists()):
        return None
    cand = pd.read_parquet(paths.PUBLIC_CANDIDATES)
    feat = pd.read_parquet(paths.PUBLIC_FEATURES)
    if "study" not in cand.columns or accession not in set(cand["study"]):
        return None
    cand = cand[cand["study"] == accession].reset_index(drop=True)
    feat = feat[feat["study"] == accession].reset_index(drop=True)
    # A bundle built with fewer candidates per feature than asked for would
    # silently answer a different question, so it is refused rather than
    # trimmed up to.
    built = int(cand["n_requested"].iloc[0]) if "n_requested" in cand.columns else 0
    if built < n_candidates:
        return None
    if built > n_candidates:
        cand = cand[cand["rank"] <= n_candidates].reset_index(drop=True)
        feat = feat.assign(n_candidates=cand.groupby("feature_id")["rank"].max()
                           .reindex(feat["feature_id"]).to_numpy())
    return cand, feat


FORMULA_RE = re.compile(r"(?:[A-Z][a-z]?\d*)+\Z")
BLANK = {"", "nan", "none", "na", "n/a", "-", "<na>", "unknown", "null"}


def _text(v) -> str:
    """A MAF cell as text, with every spelling of a blank mapped to ''.

    Depositors leave a missing formula as an empty cell, as the word unknown,
    or as a float NaN that renders as the string 'nan'. Passing that last one
    through to a formula search is how it reaches PubChem as a query and comes
    back 400, so the blanks are collapsed here instead.
    """
    if v is None or (isinstance(v, float) and v != v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in BLANK else s


def _attach_labels(cand: pd.DataFrame, labels: pd.DataFrame,
                   key: str = "inchikey_std", value: str = "genotoxic"):
    """Joins the experimental label onto candidates that happen to have one.

    Most candidates will not: that is the experimental gap the model exists to
    fill, and it is measured rather than filled in.
    """
    if key not in labels.columns or value not in labels.columns:
        return cand
    m = labels.dropna(subset=[key]).drop_duplicates(subset=[key]).set_index(key)[value]
    return cand.assign(efsa_genotoxic=pd.to_numeric(cand["inchikey"].map(m), errors="coerce"))


def banner(info: dict) -> str:
    """What the reader has to be told about the source on screen."""
    if info.get("is_fixture"):
        return ""
    med = info.get("median_isomer_space")
    med_txt = f"{med:,.0f}" if med == med else "n/a"
    return (
        f"> **Real public measurements: MetaboLights {info['accession']}.** "
        f"{info['n_features']} features from the study's metabolite assignment file, "
        f"{info['n_candidates']} candidate structures over "
        f"{info['distinct_structures']} distinct compounds. The retention values, "
        "formulae and assigned identities are the depositors'; the candidate sets are "
        "the real isomers of each formula in PubChem (median isomer space "
        f"{med_txt}). No per-candidate evidence score exists in a MAF, so the "
        "weighting is uniform and no ranking is claimed. "
        f"{info['n_confirmed']} features carry a parsed MSI level 1 and bypass the "
        f"weighting; {info['n_no_formula']} publish no formula and so have no "
        "candidate set at all."
        + (" Candidate sets come from the precomputed bundle in `data/`; delete it "
           "and the same code rebuilds them from MetaboLights and PubChem."
           if info.get("from_bundle") else
           " Candidate sets were built live from MetaboLights and PubChem.")
    )


def complementarity(accessions=None, cache_dir=None) -> pd.DataFrame:
    """Where a formula exists and where a stated identity level does.

    The two things section 13 needs travel separately in public GC-MS
    deposits: a feature with a formula can have its identity marginalized over
    that formula's isomers, and a feature with a stated MSI level can have the
    gate applied to it. This counts how often a study gives both. It reads only
    the assignment files, so it costs nothing beyond the MAF downloads.
    """
    rows = []
    for acc, _, _ in (SCREENED if accessions is None else [(a, 0, "") for a in accessions]):
        maf = load_maf(acc, cache_dir=cache_dir)
        raw = maf["reliability"] if "reliability" in maf.columns else pd.Series(index=maf.index)
        level = raw.map(_msi_level)
        formula = (maf["chemical_formula"].map(_text).ne("")
                   if "chemical_formula" in maf.columns
                   else pd.Series(False, index=maf.index))
        rows.append({
            "study": acc, "features": len(maf),
            "with_formula": int(formula.sum()),
            "with_msi_level": int(level.notna().sum()),
            "gated (level 1)": int((level == 1).sum()),
            "marginalizable (formula, no level)": int((formula & level.isna()).sum()),
            "neither": int((~formula & level.isna()).sum()),
            "raw levels": "+".join(sorted(str(v).strip() for v in raw.dropna().unique()
                                          if str(v).strip())) or "-",
        })
    return pd.DataFrame(rows)


def acquisition_columns(accession: str, cache_dir=None) -> pd.DataFrame:
    """Columns in a deposited MAF that could support an acquisition-batch split.

    A scaffold split controls chemical leakage; it does nothing about features
    that share column state, tuning drift and carryover because they were
    measured in the same sequence. Controlling that needs a run date, a batch
    label or a run order, and this reports whether the deposit publishes one.

    `retention_time` is deliberately not counted as a hit. It is chromatographic
    position within a run, not position in the acquisition sequence, and using
    it as a batch proxy would be a different control wearing this one's name.
    """
    maf = load_maf(accession, cache_dir=cache_dir)
    pat = re.compile(r"date|batch|run[_ ]?order|sequence|campaign|acquisition|injection",
                     re.I)
    hits = [c for c in maf.columns if pat.search(c)]
    if not hits:
        return pd.DataFrame({"column": ["(none found)"], "non-null rows": [0],
                             "n columns searched": [len(maf.columns)]})
    return pd.DataFrame({"column": hits,
                         "non-null rows": [int(maf[c].notna().sum()) for c in hits],
                         "n columns searched": len(maf.columns)})
