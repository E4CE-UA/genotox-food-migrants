"""Standardization of structures, ECFP4 fingerprints, scaffolds and similarity.

Standardization uses an explicit scope check and RDKit transformations:
  Scope check -> Cleanup -> single carbon-containing fragment -> Uncharger
  -> TautomerEnumerator.Canonicalize
Each step can change the InChIKey, so the module records how many
structures change and at which step, instead of applying the cleanup silently.
The model represents a single organic parent. Inorganic substances, metal
complexes and inputs with multiple carbon-containing fragments are outside
that scope; they are recorded as excluded before any fragment is discarded.
Disconnected sodium/potassium counterions are allowed. This is a modelling
scope restriction, not a claim that excluded substances are non-genotoxic.
"""

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Tautomer canonicalization is combinatorial: above this number
# of heavy atoms that step is skipped and this is recorded in the note column.
MAX_TAUTOMER_ATOMS = 60

# Included in every cache row so pre-fix standardizations cannot be reused.
STANDARDIZATION_POLICY = "single-organic-parent-v2"
_ORGANIC_ELEMENTS = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
_COUNTERION_ELEMENTS = {11, 19}  # isolated Na+ and K+ only


def _scope_reason(mol) -> str:
    fragments = Chem.GetMolFrags(mol, asMols=True)
    organic = [f for f in fragments if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
    if not organic:
        return "outside model scope: no carbon-containing parent"
    if len(organic) > 1:
        return "outside model scope: multiple carbon-containing fragments"
    for atom in mol.GetAtoms():
        z = atom.GetAtomicNum()
        if z in _ORGANIC_ELEMENTS:
            continue
        if z in _COUNTERION_ELEMENTS and atom.GetDegree() == 0 and atom.GetFormalCharge() == 1:
            continue
        return "outside model scope: unsupported element or metal complex"
    return ""


def _inchikey(mol):
    try:
        return Chem.MolToInchiKey(mol) or None
    except Exception:
        return None


def standardize_mol(smi: str) -> dict:
    """Applies the standardization cascade to a SMILES.

    Returns a dict with the input and output SMILES and InChIKey, the number
    of original fragments, whether the InChIKey changed and an incident note.
    """
    result = {
        "smiles_std": None, "inchikey_std": None, "n_orig_fragments": np.nan,
        "inchikey_changed": False, "ok": False, "note": "",
        "standardization_policy": STANDARDIZATION_POLICY,
        "rdkit_version": rdBase.rdkitVersion,
    }
    if not isinstance(smi, str) or not smi.strip():
        result["note"] = "no smiles"
        return result
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        result["note"] = "not parseable"
        return result

    result["n_orig_fragments"] = len(Chem.GetMolFrags(mol))
    reason = _scope_reason(mol)
    if reason:
        result["note"] = reason
        return result
    ik_input = _inchikey(mol)
    try:
        mol = rdMolStandardize.Cleanup(mol)
        # Select the one organic fragment explicitly. LargestFragmentChooser
        # can otherwise choose a larger inorganic counterion.
        fragments = Chem.GetMolFrags(mol, asMols=True)
        organic = [f for f in fragments if any(a.GetAtomicNum() == 6 for a in f.GetAtoms())]
        if len(organic) != 1:
            result["note"] = "outside model scope after cleanup: not one organic parent"
            return result
        mol = organic[0]
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        if mol.GetNumHeavyAtoms() <= MAX_TAUTOMER_ATOMS:
            mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        else:
            result["note"] = "tautomers skipped (large molecule)"
    except Exception as exc:
        result["note"] = f"standardization failed: {type(exc).__name__}"
        return result

    ik_output = _inchikey(mol)
    result.update(
        smiles_std=Chem.MolToSmiles(mol),
        inchikey_std=ik_output,
        inchikey_changed=bool(ik_input and ik_output and ik_input != ik_output),
        ok=ik_output is not None,
    )
    return result


def standardize_table(df: pd.DataFrame, smiles_col: str = "smiles",
                       cache_path=None) -> pd.DataFrame:
    """Standardizes a table with an on-disk cache over the input SMILES."""
    input_values = df[smiles_col].astype("string")
    unique = pd.Index(input_values.dropna().unique(), name=smiles_col)

    cache = pd.DataFrame()
    if cache_path is not None and cache_path.exists():
        cached = pd.read_parquet(cache_path)
        required = {smiles_col, "standardization_policy", "rdkit_version"}
        if required <= set(cached.columns):
            cache = cached.loc[
                cached["standardization_policy"].eq(STANDARDIZATION_POLICY)
                & cached["rdkit_version"].eq(rdBase.rdkitVersion)
            ].set_index(smiles_col)
            cache = cache[~cache.index.duplicated()]

    pending = unique.difference(cache.index) if len(cache) else unique
    if len(pending):
        new_rows = pd.DataFrame([standardize_mol(s) for s in pending], index=pending)
        new_rows.index.name = smiles_col
        cache = pd.concat([cache, new_rows]) if len(cache) else new_rows
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache.reset_index().to_parquet(cache_path, index=False)

    if cache.empty:
        cache = pd.DataFrame(columns=standardize_mol("").keys(),
                             index=pd.Index([], name=smiles_col))
    return df.join(cache, on=smiles_col)


def standardization_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("input structures", len(df)),
        ("successfully standardized", int(df["ok"].sum())),
        ("outside model scope", int(df["note"].fillna("").str.startswith("outside model scope").sum())),
        ("not parseable, failed, or outside scope", int((~df["ok"].fillna(False)).sum())),
        ("with more than one fragment (salt or mixture)", int((df["n_orig_fragments"] > 1).sum())),
        ("change InChIKey upon standardization", int(df["inchikey_changed"].sum())),
        ("unique InChIKeys before", df["inchikey"].nunique() if "inchikey" in df else 0),
        ("unique InChIKeys after", df["inchikey_std"].nunique()),
    ]
    return pd.DataFrame(rows, columns=["quantity", "n"])


# ------------------------------------------------------------------ fingerprints
_GEN_ECFP4 = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprints(smiles, radius: int = 2, n_bits: int = 2048):
    """List of ExplicitBitVect (ECFP4 by default); None if the SMILES fails."""
    gen = (_GEN_ECFP4 if (radius, n_bits) == (2, 2048)
           else rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits))
    fps = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s) if isinstance(s, str) else None
        fps.append(gen.GetFingerprint(mol) if mol is not None else None)
    return fps


def matrix(fps, n_bits: int = 2048) -> np.ndarray:
    """Converts the fingerprints into the dense matrix consumed by the models."""
    X = np.zeros((len(fps), n_bits), dtype=np.uint8)
    for i, fp in enumerate(fps):
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, X[i])
    return X


def scaffold(smi: str, generic: bool = False) -> str:
    """Bemis-Murcko scaffold as SMILES; empty string for acyclic molecules."""
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return ""
    try:
        core = MurckoScaffold.GetScaffoldForMol(mol)
        if generic:
            core = MurckoScaffold.MakeScaffoldGeneric(core)
        return Chem.MolToSmiles(core)
    except Exception:
        return ""


def max_tanimoto(query_fps, reference_fps) -> np.ndarray:
    """Tanimoto similarity to the nearest neighbor in the reference set."""
    ref = [f for f in reference_fps if f is not None]
    out = np.full(len(query_fps), np.nan)
    if not ref:
        return out
    for i, fp in enumerate(query_fps):
        if fp is not None:
            out[i] = max(DataStructs.BulkTanimotoSimilarity(fp, ref))
    return out


def nearest_neighbor(query_fps, reference_fps):
    """Per query fingerprint, the nearest neighbour in the reference set.

    Returns ``(sim, idx)``: ``sim`` is the maximum Tanimoto similarity (same
    value as :func:`max_tanimoto`), ``idx`` is the position of that neighbour
    in the ORIGINAL ``reference_fps`` order. ``None``-valued references are
    skipped during the search but do not shift the returned index, so ``idx``
    can index straight back into the training table. ``sim`` is ``nan`` and
    ``idx`` is ``-1`` where the query is missing or no usable reference exists.
    """
    kept = [(j, f) for j, f in enumerate(reference_fps) if f is not None]
    sim = np.full(len(query_fps), np.nan)
    idx = np.full(len(query_fps), -1, dtype=int)
    if not kept:
        return sim, idx
    ref_idx = [j for j, _ in kept]
    ref_fps = [f for _, f in kept]
    for i, fp in enumerate(query_fps):
        if fp is not None:
            sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
            k = int(np.argmax(sims))
            sim[i] = sims[k]
            idx[i] = ref_idx[k]
    return sim, idx


def embed_tsne(X, keys=None, cache_path=None, seed: int = 0, perplexity: int = 30):
    """2-D map of the fingerprint matrix using Tanimoto (Jaccard) distance.

    Euclidean PCA on binary fingerprints captures under 10 % of the variance
    and produces a blob; Jaccard is the distance the rest of the notebook
    already uses for the applicability domain, so the map and the domain
    analysis speak about the same geometry.

    Cached on the identity of the compound set (`keys`), because the set only
    changes when the aggregation controls change, not on every re-render.
    """
    import hashlib
    from pathlib import Path

    import numpy as np
    from sklearn.manifold import TSNE

    X = np.asarray(X)
    tag = None
    if keys is not None and cache_path is not None:
        tag = hashlib.sha256("|".join(map(str, keys)).encode()).hexdigest()[:16]
        f = Path(cache_path).with_name(f"tsne_{tag}.npy")
        if f.exists():
            return np.load(f)

    z = TSNE(
        n_components=2, metric="jaccard", init="random",
        random_state=seed, perplexity=min(perplexity, max(5, (len(X) - 1) // 3)),
    ).fit_transform(X.astype(bool))

    if tag is not None:
        f.parent.mkdir(parents=True, exist_ok=True)
        np.save(f, z)
    return z


# A small, explicitly partial set of DNA-reactive structural classes.
#
# This is NOT the Benigni-Bossa rulebase. That rulebase is a curated expert
# system with dozens of alerts, exclusion rules and potency modulations, and
# this project has no licensed implementation of it. What follows is a short
# list of structural classes whose DNA-reactive mechanism is textbook
# chemistry, written as SMARTS here so that the agreement section has a
# second opinion computable from a structure alone.
#
# Consequences of that choice, which the notebook states where it uses them:
#   - agreement or kappa against this list is not comparable to an evaluation
#     against Benigni-Bossa, and must not be quoted as one;
#   - the list is deliberately under-inclusive. A structure with no hit is not
#     "no alert" in any regulatory sense, it is "none of these ten";
#   - a hit is a structural flag, not a genotoxicity conclusion. Many
#     substances carrying these groups are not genotoxic in vivo, which is
#     exactly why the agreement table is a disagreement count and not an
#     accuracy measure.
DNA_REACTIVE_ALERTS = (
    ("aromatic nitro", "[$([NX3](=O)=O),$([NX3+](=O)[O-])]c"),
    ("primary aromatic amine", "[NX3;H2;!$(N[C,S]=[O,S,N])]c"),
    ("epoxide", "[OX2r3]1[#6r3][#6r3]1"),
    ("aziridine", "[NX3r3]1[#6r3][#6r3]1"),
    ("N-nitroso", "[NX3][NX2]=O"),
    # Both nitrogens must be free of an acyl or sulfonyl neighbour, which is
    # what separates a hydrazine from a hydrazide, a semicarbazide or a
    # sulfonohydrazide. Those carry an N-N bond but not the reactivity the
    # label claims, and letting them match would make the class name a
    # misstatement of what the pattern expresses. N-aryl is deliberately
    # still allowed: phenylhydrazine is a hydrazine alert, not an exception
    # to one.
    #
    # Scope, so that the exclusion is not read as a claim: this class covers
    # hydrazines, not every N-N structure that might matter. The excluded
    # hydrazides are therefore not asserted to be inert; they fall outside
    # these ten classes, which is the same status as any other structure the
    # list does not cover. Adding a hydrazide class would widen the chemical
    # scope of the rule set rather than fix a label, and that is a separate
    # decision needing its own justification, not a corollary of this one.
    ("hydrazine", "[NX3;!$(N=*);!$(N#*);!$([NX3][CX3]=[OX1,SX1]);!$([NX3][SX4](=[OX1])=[OX1])][NX3;!$(N=*);!$(N#*);!$([NX3][CX3]=[OX1,SX1]);!$([NX3][SX4](=[OX1])=[OX1])]"),
    ("organic azide", "[#6][NX2]=[NX2+]=[NX1-]"),
    ("aromatic azo", "c[NX2]=[NX2]c"),
    ("alkyl halide", "[CX4;H1,H2,H3][F,Cl,Br,I]"),
    ("Michael acceptor", "[CX3]=[CX3][CX3]=[OX1]"),
)


def alert_patterns():
    """Compile the alert SMARTS once, dropping any that fail to compile."""
    out = []
    for name, sma in DNA_REACTIVE_ALERTS:
        patt = Chem.MolFromSmarts(sma)
        if patt is not None:
            out.append((name, patt))
    return out


def dna_reactive_alerts(smiles) -> pd.DataFrame:
    """Which of the DNA-reactive classes each structure carries.

    Returns one row per input in input order with `genotox_alert` (any class
    matched), `alert_names` (semicolon-separated, empty when none matched) and
    `alert_parsed` (whether the SMILES could be read at all). A structure that
    does not parse gets `genotox_alert = None` rather than False: the notebook
    distinguishes "no alert" from "not evaluable", and collapsing the two would
    turn a parsing failure into a clean bill of health.
    """
    patterns = alert_patterns()
    RDLogger.DisableLog("rdApp.*")
    rows = []
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
        if mol is None:
            rows.append({"genotox_alert": None, "alert_names": "", "alert_parsed": False})
            continue
        hits = [name for name, patt in patterns if mol.HasSubstructMatch(patt)]
        rows.append({"genotox_alert": bool(hits), "alert_names": ";".join(hits),
                     "alert_parsed": True})
    RDLogger.EnableLog("rdApp.*")
    return pd.DataFrame(rows, index=pd.RangeIndex(len(rows)))


def attach_alerts(cand: pd.DataFrame, smiles_col: str = "smiles") -> pd.DataFrame:
    """Add the alert columns to a candidate table, computing each SMILES once.

    Candidate tables repeat structures across features, so the substructure
    match is done on the distinct SMILES and joined back.
    """
    if smiles_col not in cand.columns:
        return cand
    uniq = pd.Series(cand[smiles_col].dropna().unique(), name=smiles_col)
    flags = dna_reactive_alerts(uniq.tolist())
    flags[smiles_col] = uniq.to_numpy()
    return cand.drop(columns=[c for c in ("genotox_alert", "alert_names", "alert_parsed")
                              if c in cand.columns]).merge(flags, on=smiles_col, how="left")
