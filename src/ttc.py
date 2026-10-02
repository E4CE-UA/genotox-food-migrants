"""Threshold of Toxicological Concern: what the genotoxicity call decides.

A probability is not a decision. In the Kroes decision tree a compound with a
structural alert for genotoxicity is held to 0.15 ug/person/day, while a
Cramer class III compound is allowed 90. The genotoxic/non-genotoxic call is
what selects between them, so a change in that call moves the admissible
exposure by a factor of six hundred.

This module does not re-derive the tree. The upstream export already assigns
`ttc_class`, `ttc_ug_day` and `cramer_class` per structure; what is added here
is the counterfactual: which features the model would move into the genotoxic
bin that the pipeline did not, and what that does to their limit.

Values are ug/person/day for a 60 kg adult, as published by Kroes et al.
(2004), Food Chem. Toxicol. 42(1), 65-83.
"""

import numpy as np
import pandas as pd

TTC_GENOTOXIC = 0.15
TTC_UG_DAY = {
    "I": 1800.0,
    "II": 540.0,
    "III": 90.0,
    "organophosphate": 18.0,
}
# Applied when the Cramer class is missing: the most restrictive non-genotoxic
# tier, so an unknown class never buys a compound a looser limit.
TTC_DEFAULT = TTC_UG_DAY["III"]


def _cramer_ttc(cramer_class) -> float:
    if isinstance(cramer_class, str):
        key = cramer_class.strip().upper().replace("CLASS", "").strip()
        if key in TTC_UG_DAY:
            return TTC_UG_DAY[key]
        if key.lower() in TTC_UG_DAY:
            return TTC_UG_DAY[key.lower()]
    return TTC_DEFAULT


def baseline_ttc(cand: pd.DataFrame, weight, weight_floor: float = 0.05,
                 col_group: str = "feature_id") -> pd.DataFrame:
    """The limit the pipeline already applies, per feature.

    Among the candidates carrying more than `weight_floor` of the identity
    mass, the most restrictive limit wins. Taking the minimum rather than a
    weighted mean is deliberate: a limit is a bound, and averaging a bound
    across candidates would produce a number that protects against none of
    them individually.
    """
    d = pd.DataFrame({
        "group": cand[col_group].to_numpy(),
        "w": np.asarray(weight, dtype=float),
    })
    if "ttc_ug_day" in cand.columns:
        d["ttc"] = pd.to_numeric(cand["ttc_ug_day"], errors="coerce")
    else:
        d["ttc"] = np.nan
    fallback = cand["cramer_class"].map(_cramer_ttc) if "cramer_class" in cand.columns \
        else pd.Series(TTC_DEFAULT, index=cand.index)
    d["ttc"] = d["ttc"].fillna(pd.Series(fallback.to_numpy(), index=d.index))

    keep = d[d["w"] >= weight_floor]
    keep = keep if len(keep) else d
    g = keep.groupby("group", sort=False)
    out = pd.DataFrame({
        "ttc_baseline": g["ttc"].min(),
        "n_weighted_candidates": g.size(),
    }).reset_index().rename(columns={"group": col_group})
    return out


def decide(feat_p: pd.DataFrame, base: pd.DataFrame, threshold: float = 0.5,
           p_col: str = "identity_weighted_genotoxicity", col_group: str = "feature_id") -> pd.DataFrame:
    """Applies the genotoxic branch when the model calls the feature positive.

    Returns one row per feature with the limit before and after, and the
    factor between them. `crosses` marks the features the model moves into
    the genotoxic bin.
    """
    out = feat_p.merge(base, on=col_group, how="left")
    out["ttc_baseline"] = out["ttc_baseline"].fillna(TTC_DEFAULT)
    p = pd.to_numeric(out[p_col], errors="coerce")
    out["calls_genotoxic"] = p >= threshold
    out["ttc_model"] = np.where(out["calls_genotoxic"],
                                np.minimum(out["ttc_baseline"], TTC_GENOTOXIC),
                                out["ttc_baseline"])
    out["crosses"] = out["ttc_model"] < out["ttc_baseline"]
    out["tightening_factor"] = out["ttc_baseline"] / out["ttc_model"]
    return out


def crossing_summary(dec: pd.DataFrame) -> pd.DataFrame:
    """One table a risk assessor can read without the notebook around it."""
    n = len(dec)
    cross = dec[dec["crosses"]]
    rows = [
        ("features evaluated", n, ""),
        ("already at the genotoxic limit", int((dec["ttc_baseline"] <= TTC_GENOTOXIC).sum()),
         "the pipeline had already flagged them; the model changes nothing"),
        ("moved into the genotoxic bin", len(cross),
         f"limit drops to {TTC_GENOTOXIC} ug/day"),
        ("median tightening among those", float(cross["tightening_factor"].median())
         if len(cross) else float("nan"), "times stricter than before"),
        ("unchanged", int(n - len(cross)), ""),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value", "meaning"])


def exposure_columns(cand: pd.DataFrame, feat: pd.DataFrame):
    """Names of columns that could carry a measured amount, or None.

    The half of the decision this project cannot close is `exposure > TTC`:
    neither exported table carries a concentration or an intake. Rather than
    invent one, this reports what is actually present so the check can be
    switched on the moment the export includes it.
    """
    pat = ("conc", "ug", "ng", "mg", "area", "amount", "cantidad", "exposicion",
           "exposure", "intake", "migra")
    # The limit columns match some of those patterns and are not measurements.
    skip = ("ttc", "cramer")
    found = []
    for name, df in (("candidates", cand), ("features", feat)):
        for c in df.columns:
            low = c.lower()
            if any(p in low for p in pat) and not any(s in low for s in skip):
                found.append((name, c))
    return pd.DataFrame(found, columns=["table", "column"])


def exceedance(dec: pd.DataFrame, exposure, col_group: str = "feature_id") -> pd.DataFrame:
    """Which features exceed their limit, before and after the model's call.

    `exposure` is ug/person/day per feature. Only callable once the export
    carries it; `exposure_columns` says whether that is the case.
    """
    out = dec.copy()
    out["exposure_ug_day"] = pd.to_numeric(pd.Series(exposure).to_numpy(), errors="coerce")
    out["exceeds_baseline"] = out["exposure_ug_day"] > out["ttc_baseline"]
    out["exceeds_model"] = out["exposure_ug_day"] > out["ttc_model"]
    out["newly_exceeds"] = out["exceeds_model"] & ~out["exceeds_baseline"]
    return out
