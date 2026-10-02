"""From candidates per feature to P(genotoxic | feature).

Each chromatographic feature has up to 5 identity candidates. The
evidence column is the sum of NIST scores re-ranked by AIRI: it is NOT a
probability and does not sum to 1 within a feature. Turning it into weights is an
explicit ASSUMPTION of the notebook, not a fact, and that is why there are three methods and the
result is always presented together with the weight coverage.

  P(genotoxic | feature) = sum_k  w_k * P(genotoxic | candidate_k)

If a candidate has no SMILES or falls outside the applicability domain, its
weight is not redistributed among the others: it is counted as probability mass
not covered. A feature with coverage 0.4 has 60% of its identity
unevaluated and its P should not be read as a firm prediction.
"""

import numpy as np
import pandas as pd

#: "uniform" is first because it is the default, and the default is a
#: methodological position rather than a fallback. An upstream match score is
#: a per-candidate ranking quantity, not P(identity), because nothing
#: normalizes it across a feature's candidate set; a softmax over it inherits
#: that objection
#: with less justification behind it, since it invents a temperature on top of
#: a sum of scores. What survives the objection is equal weight across the
#: candidate set: the position of maximum ignorance about identity, which
#: assumes nothing and needs no calibration to defend. Any departure from it
#: has to be earned against it -- which is exactly what the ablation in
#: section 15 measures.
METHODS = ("uniform", "softmax", "proportional", "top1_only")


def evidence_weights(cand: pd.DataFrame, method: str = "uniform",
                    temperature: float = 1.0,
                    col_group: str = "feature_id",
                    col_evidence: str = "evidence") -> pd.Series:
    """Weights per candidate within each feature. They sum to 1 per feature.

    Defaults to ``uniform``; see ``METHODS`` for why that is the position to
    depart from rather than the one to fall back to.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    e = pd.to_numeric(cand[col_evidence], errors="coerce")

    if method == "uniform":
        n = cand.groupby(col_group)[col_group].transform("size")
        return (1.0 / n).astype(float)

    if method == "top1_only":
        r = cand["rank"] if "rank" in cand else e.groupby(cand[col_group]).rank(ascending=False)
        return (r == r.groupby(cand[col_group]).transform("min")).astype(float)

    if method == "proportional":
        base = e.clip(lower=0)
    else:  # softmax with temperature over the evidence standardized per feature
        g = e.groupby(cand[col_group])
        z = (e - g.transform("mean")) / g.transform("std").replace(0, np.nan).fillna(1.0)
        base = np.exp(z / max(temperature, 1e-6))

    total = base.groupby(cand[col_group]).transform("sum")
    return (base / total.replace(0, np.nan)).fillna(0.0)


def probability_per_feature(cand: pd.DataFrame, p_candidate, weight,
                             col_group: str = "feature_id",
                             in_domain=None) -> pd.DataFrame:
    """Marginalizes toxicity over the candidates of each feature.

    p_candidate: probability predicted per candidate (NaN if there is no SMILES).
    in_domain: boolean per candidate; those outside are excluded from the sum
    but their weight counts as uncovered mass.
    """
    d = pd.DataFrame({
        "group": cand[col_group].to_numpy(),
        "p": pd.to_numeric(pd.Series(p_candidate).to_numpy(), errors="coerce"),
        "w": np.asarray(weight, dtype=float),
    })
    d["usable"] = d["p"].notna()
    if in_domain is not None:
        d["usable"] &= np.asarray(in_domain, dtype=bool)
    d["wp"] = np.where(d["usable"], d["w"] * d["p"], 0.0)
    d["w_ok"] = np.where(d["usable"], d["w"], 0.0)
    d["p_ok"] = np.where(d["usable"], d["p"], np.nan)

    g = d.groupby("group", sort=False)
    out = pd.DataFrame({
        "covered_weight": g["w_ok"].sum(),
        "identity_weighted_genotoxicity": g["wp"].sum(),
        "p_max_candidate": g["p_ok"].max(),
        "n_candidates": g.size(),
        "n_evaluable": g["usable"].sum(),
    })
    # identity_weighted_genotoxicity conditioned on the covered part: comparable across features
    out["identity_weighted_genotoxicity_norm"] = out["identity_weighted_genotoxicity"] / out["covered_weight"].replace(0, np.nan)
    out = out.reset_index().rename(columns={"group": col_group})
    return out


def coverage_summary(feat_p: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("features evaluated", len(feat_p)),
        ("with all candidates evaluable", int((feat_p["n_evaluable"] == feat_p["n_candidates"]).sum())),
        ("with no evaluable candidate", int((feat_p["n_evaluable"] == 0).sum())),
        ("median covered weight", round(float(feat_p["covered_weight"].median()), 3)),
        ("features with covered weight < 0.5", int((feat_p["covered_weight"] < 0.5).sum())),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


#: Levels accepted as the ablation's reference. Deliberately NOT the same as
#: CONFIRMED_MSI, which gates the weighting. Exempting a feature from
#: marginalization is a claim that its identity is settled, and only an
#: authentic standard (level 1) supports that. Scoring an estimator is a
#: weaker demand, so level 2 is admitted to buy sample size. Level 1 is
#: populated only by compounds an authentic standard was injected for, which in
#: an untargeted run is normally just the deuterated QC surrogates: too few to
#: conclude from, and not migrants either. Residual weakness, to be declared
#: wherever the result is shown: level 2 rests on spectral evidence, which the
#: candidate ranking also uses, so reference and estimator are not fully
#: independent. Only injecting a mixture of known standards removes that, and
#: no amount of data access substitutes for it.
REFERENCE_MSI = (1, 2)


def _restrict_msi(feat: pd.DataFrame, msi_levels) -> pd.DataFrame:
    """Keeps only features confirmed at the given MSI levels, if possible.

    Pass ``REFERENCE_MSI`` for the ablation. Level 1 alone is the rigorous
    choice but yields too few features to conclude from; see that constant
    for why level 2 is admitted and what it costs.
    """
    if msi_levels is None or "msi_level" not in feat.columns:
        return feat
    sub = feat[feat["msi_level"].isin(msi_levels)]
    return sub if len(sub) else feat.iloc[0:0]


def ablation_identity_evidence(cand: pd.DataFrame, feat: pd.DataFrame,
                               temperature: float = 1.0,
                               col_group: str = "feature_id",
                               col_p: str = "p_genotox",
                               msi_levels=None) -> pd.DataFrame:
    """Does the spectral identity evidence buy anything over ignoring it?

    The honest form of "is the GC-MS worth anything here". It needs a
    reference, and the only one available is the subset of features whose
    identity is confirmed: `features_v3.inchikey_id` names the accepted
    compound, and when that compound is among the candidates its predicted
    probability is what a perfect identification would have given.

    Three estimators that do NOT get to see the answer are scored against it:

      evidence_weighted  w_k from the spectral evidence (what the notebook does)
      uniform            w_k = 1/K, structure alone, identity evidence discarded
      top1_only          the highest-ranked candidate, identity treated as certain

    If `evidence_weighted` does not beat `uniform`, the weighting is decoration
    and should be dropped. The comparison is restricted to confirmed features,
    so it says nothing about the features that most need the marginalization --
    that limitation is real and is reported alongside the numbers.
    """
    if "inchikey_id" not in feat.columns or "inchikey" not in cand.columns:
        return pd.DataFrame(columns=["estimator", "n_features", "mae", "rmse", "spearman"])

    feat = _restrict_msi(feat, msi_levels)
    ref = feat.loc[feat["inchikey_id"].notna(), [col_group, "inchikey_id"]]
    d = cand.merge(ref, on=col_group, how="inner")
    d["_is_gold"] = d["inchikey"] == d["inchikey_id"]

    gold = (d[d["_is_gold"] & d[col_p].notna()]
            .groupby(col_group)[col_p].first().rename("p_gold"))
    if gold.empty:
        return pd.DataFrame(columns=["estimator", "n_features", "mae", "rmse", "spearman"])

    d = d[d[col_group].isin(gold.index)].copy()
    n_cand = d.groupby(col_group)[col_p].transform("size")
    schemes = {
        "evidence_weighted": evidence_weights(d, "softmax", temperature, col_group),
        "uniform": pd.Series(1.0 / n_cand.to_numpy(), index=d.index),
        "top1_only": evidence_weights(d, "top1_only", temperature, col_group),
    }

    rows = []
    for name, w in schemes.items():
        est = probability_per_feature(d, d[col_p], w, col_group=col_group)
        m = est.set_index(col_group)["identity_weighted_genotoxicity_norm"].reindex(gold.index)
        ok = m.notna() & gold.notna()
        if ok.sum() < 3:
            rows.append((name, int(ok.sum()), np.nan, np.nan, np.nan))
            continue
        err = (m[ok] - gold[ok]).to_numpy()
        rho = pd.Series(m[ok].to_numpy()).corr(pd.Series(gold[ok].to_numpy()), method="spearman")
        rows.append((name, int(ok.sum()), float(np.abs(err).mean()),
                     float(np.sqrt((err ** 2).mean())), float(rho)))

    out = pd.DataFrame(rows, columns=["estimator", "n_features", "mae", "rmse", "spearman"])
    return out.round(4)


def ablation_reference_check(cand: pd.DataFrame, feat: pd.DataFrame,
                             col_group: str = "feature_id",
                             msi_levels=None) -> pd.DataFrame:
    """Is the confirmed identity independent of the evidence ranking?

    The ablation above is only meaningful if `inchikey_id` was established by
    something other than "take the best-scoring candidate". If the export
    assigns it that way, then `top1_only` is being scored against a
    copy of itself and will win by construction -- the comparison would be
    circular and must be read as nothing at all.

    This measures the overlap. A value near 100 % means the reference is not
    independent: restrict the ablation to features confirmed against an
    authentic standard (MSI level 1) before believing any of it.
    """
    if "inchikey_id" not in feat.columns or "inchikey" not in cand.columns:
        return pd.DataFrame(columns=["quantity", "value"])

    feat = _restrict_msi(feat, msi_levels)
    ref = feat.loc[feat["inchikey_id"].notna(), [col_group, "inchikey_id"]]
    d = cand.merge(ref, on=col_group, how="inner")
    if d.empty:
        return pd.DataFrame([("features with a confirmed identity", 0)],
                            columns=["quantity", "value"])

    r = d["rank"] if "rank" in d else pd.Series(np.nan, index=d.index)
    top1 = d.loc[r.groupby(d[col_group]).transform("min") == r]
    agree = (top1["inchikey"] == top1["inchikey_id"]).groupby(top1[col_group]).any()

    rows = [
        ("features with a confirmed identity", int(d[col_group].nunique())),
        ("of those, confirmed == top-1 candidate", int(agree.sum())),
        ("overlap (%)", round(100.0 * float(agree.mean()), 1)),
    ]
    return pd.DataFrame(rows, columns=["quantity", "value"])


# --- Identity gate by MSI level -------------------------------------------
#
# MSI levels are categorical statements about what kind of evidence exists
# (Sumner 2007; Schymanski 2014), not probabilities, and the scale exists
# precisely to stop them being read as numbers. Turning `evidence` into a
# softmax would contradict that for every feature, including the ones whose
# identity is already settled.
#
# The gate keeps both things true: a feature confirmed against an authentic
# standard has one identity and nothing to marginalize, so it bypasses the
# weighting entirely. Marginalization is then restricted to the features
# where the identity really is ambiguous, which is the only place it was
# ever justified.

#: Levels whose identity is treated as settled, exempting the feature from
#: marginalization. Level 1 only: confirmation against an authentic standard.
#:
#: Caveat on the scale this gate reads. An MSI level is only as good as the
#: evidence branches feeding it, and on a GC-MS batch whose mass axis is not
#: calibrated to formula-level accuracy, every branch depending on exact mass
#: (formula confirmation, fragment consistency, the level-3 veto) is degraded,
#: leaving the level resting on its spectral and retention-index branches. That
#: does not invalidate this gate: level 1 is established by an authentic
#: standard, independent of all three. It does mean the boundary between the
#: lower levels can carry less information than the scale's definition implies,
#: which matters wherever those levels are used for anything but this gate --
#: see REFERENCE_MSI. Whether it applies to a given batch is a property of the
#: acquisition and is not knowable from the export alone.
CONFIRMED_MSI = (1,)


def gated_weights(cand: pd.DataFrame, feat: pd.DataFrame,
                  method: str = "uniform", temperature: float = 1.0,
                  confirmed_levels=CONFIRMED_MSI,
                  col_group: str = "feature_id") -> pd.Series:
    """Candidate weights with confirmed identities exempted from weighting.

    For a feature whose `msi_level` is in `confirmed_levels` and which carries
    an `inchikey_id`, all the weight goes to the candidate matching that
    InChIKey. Every other feature is weighted by `evidence_weights`.

    Returns a weight per row of `cand`, summing to 1 within each feature.
    """
    w = evidence_weights(cand, method=method, temperature=temperature,
                         col_group=col_group).astype(float)
    if not {"msi_level", "inchikey_id"} <= set(feat.columns):
        return w

    conf = feat[feat["msi_level"].isin(confirmed_levels) & feat["inchikey_id"].notna()]
    if conf.empty:
        return w

    key = cand[col_group].map(dict(zip(conf[col_group], conf["inchikey_id"])))
    hit = key.notna() & (cand["inchikey"] == key)
    # Only override features where the confirmed structure is actually among
    # the candidates; otherwise there is nothing to put the weight on.
    usable = hit.groupby(cand[col_group]).transform("any") & key.notna()
    w = w.where(~usable, hit.astype(float))
    return w


def gate_summary(cand: pd.DataFrame, feat: pd.DataFrame,
                 confirmed_levels=CONFIRMED_MSI,
                 col_group: str = "feature_id") -> pd.DataFrame:
    """How many features bypass the weighting, and why the rest do not."""
    if "msi_level" not in feat.columns:
        return pd.DataFrame([("msi_level absent from the export", len(feat))],
                            columns=["group", "n_features"])
    conf = feat["msi_level"].isin(confirmed_levels)
    has_id = feat.get("inchikey_id", pd.Series(index=feat.index, dtype=object)).notna()
    ids = dict(zip(feat[col_group], feat.get("inchikey_id", pd.Series(dtype=object))))
    key = cand[col_group].map(ids)
    present = ((cand["inchikey"] == key) & key.notna()).groupby(cand[col_group]).any()
    present = feat[col_group].map(present).fillna(False)

    rows = [
        ("confirmed identity, weighting bypassed", int((conf & has_id & present).sum()),
         "MSI level confirmed and the structure is among the candidates"),
        ("confirmed but structure absent", int((conf & has_id & ~present).sum()),
         "confirmed InChIKey is not in the candidate list; falls back to weighting"),
        ("confirmed level, no inchikey_id", int((conf & ~has_id).sum()),
         "level says confirmed but no identity recorded"),
        ("ambiguous, marginalized", int((~conf).sum()),
         "identity genuinely uncertain; this is what the weighting is for"),
    ]
    return pd.DataFrame(rows, columns=["group", "n_features", "meaning"])


# --- External check against the rule-based alerts --------------------------
#
# A common upstream choice is to fill missing experimental genotoxicity with
# structural alerts (Benigni-Bossa), an expert rule system. This project is a
# statistical model
# fitted on EFSA conclusions. The two are independent routes to the same call,
# so their agreement is evidence about both, and their disagreements are the
# cases worth a human look.

TRUTHY = {"1", "true", "yes", "y", "si", "sí", "positive", "alert", "alerta"}
FALSY = {"0", "false", "no", "n", "negative", "none", "sin alerta"}


def _as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    n = pd.to_numeric(s, errors="coerce")
    out = n.where(n.isna(), n > 0)
    t = s.astype(str).str.strip().str.lower()
    out = out.where(out.notna(), t.map(lambda v: True if v in TRUTHY
                                       else (False if v in FALSY else None)))
    return out.astype("boolean")


def experimental_gap(cand: pd.DataFrame, col: str = "efsa_genotoxic") -> pd.DataFrame:
    """How much of the candidate space has no experimental value at all.

    This is the hole the model is there to fill, stated as a number.
    """
    if col not in cand.columns:
        return pd.DataFrame([("column absent from the export", len(cand), 100.0)],
                            columns=["group", "n", "pct"])
    have = cand[col].notna()
    n = len(cand)
    rows = [("experimental EFSA value", int(have.sum()), 100 * have.mean()),
            ("estimated from structure only", int((~have).sum()), 100 * (~have).mean())]
    return pd.DataFrame(rows, columns=["group", "n", "pct"]).round({"pct": 1})


def alert_agreement(cand: pd.DataFrame, p_candidate, threshold: float = 0.5,
                    alert_col: str = "genotox_alert",
                    restrict_to_gap: bool = False,
                    gap_col: str = "efsa_genotoxic") -> pd.DataFrame:
    """Confusion between the structural alert and this model's call.

    Neither is ground truth. The table counts how often two independent
    methods reach the same conclusion on the same structure.

    restrict_to_gap: keep only candidates with no experimental EFSA value,
    which is where both methods are actually being relied on.
    """
    if alert_col not in cand.columns:
        return pd.DataFrame([("alert column absent from the export", 0)],
                            columns=["cell", "n"])
    d = pd.DataFrame({
        "alert": _as_bool(cand[alert_col]).to_numpy(),
        "p": pd.to_numeric(pd.Series(p_candidate).to_numpy(), errors="coerce"),
    })
    if restrict_to_gap and gap_col in cand.columns:
        d = d[cand[gap_col].isna().to_numpy()]
    d = d.dropna(subset=["p"])
    d = d[d["alert"].notna()]
    if not len(d):
        return pd.DataFrame([("no candidate has both an alert and a prediction", 0)],
                            columns=["cell", "n"])
    d["model"] = d["p"] >= threshold
    a, m = d["alert"].astype(bool), d["model"]
    both = int((a & m).sum())
    neither = int((~a & ~m).sum())
    alert_only = int((a & ~m).sum())
    model_only = int((~a & m).sum())
    n = len(d)
    agree = (both + neither) / n
    # Cohen's kappa: agreement above what the two marginals would give by chance
    pe = (a.mean() * m.mean()) + ((1 - a.mean()) * (1 - m.mean()))
    kappa = (agree - pe) / (1 - pe) if pe < 1 else float("nan")
    rows = [
        ("both call it genotoxic", both, "the two methods agree, positive"),
        ("neither does", neither, "the two methods agree, negative"),
        ("alert only", alert_only, "rules fire, model does not"),
        ("model only", model_only, "model fires, rules do not"),
        ("candidates compared", n, ""),
        ("agreement (%)", round(100 * agree, 1), ""),
        ("Cohen's kappa", round(float(kappa), 3), "agreement above chance"),
    ]
    return pd.DataFrame(rows, columns=["cell", "n", "meaning"])


def alert_disagreements(cand: pd.DataFrame, p_candidate, threshold: float = 0.5,
                        alert_col: str = "genotox_alert", n: int = 15) -> pd.DataFrame:
    """The concrete structures where the two methods part company."""
    if alert_col not in cand.columns:
        return pd.DataFrame(columns=["name", "inchikey", "alert", "p", "disagreement"])
    d = cand.copy()
    d["_alert"] = _as_bool(d[alert_col])
    d["_p"] = pd.to_numeric(pd.Series(p_candidate).to_numpy(), errors="coerce")
    d = d[d["_p"].notna() & d["_alert"].notna()]
    d["_model"] = d["_p"] >= threshold
    d = d[d["_alert"].astype(bool) != d["_model"]]
    d["disagreement"] = np.where(d["_model"], "model only", "alert only")
    # The sharpest cases first: the model furthest from the rules
    d["_gap"] = (d["_p"] - threshold).abs()
    cols = [c for c in ("name", "inchikey", "cas_official") if c in d.columns]
    return (d.sort_values("_gap", ascending=False)
             .head(n)[cols + ["_alert", "_p", "disagreement"]]
             .rename(columns={"_alert": "structural alert", "_p": "p(genotoxic)"})
             .reset_index(drop=True))


def reference_levels(feat: pd.DataFrame, col_group: str = "feature_id") -> pd.DataFrame:
    """How the ablation reference is composed, by identification level.

    REFERENCE_MSI admits level 2 to buy sample size over level 1 alone, at the
    cost of a reference that shares spectral evidence with the estimator. That
    trade is only real if level 2 actually contributes features, and whether it
    does is a property of the export, not of this code. So the composition is
    reported rather than assumed: if the level-2 row is empty, the comparison
    on screen is the level-1 one regardless of which constant was passed.
    """
    if "inchikey_id" not in feat.columns or "msi_level" not in feat.columns:
        return pd.DataFrame(columns=["msi_level", "n_confirmed"])
    confirmed = feat[feat["inchikey_id"].notna()]
    out = (confirmed.groupby("msi_level")[col_group].nunique()
           .rename("n_confirmed").reset_index()
           .sort_values("msi_level").reset_index(drop=True))
    return out


def decision_table(cand: pd.DataFrame, feat_p: pd.DataFrame, threshold: float,
                   col_group: str = "feature_id",
                   domain_bounds=(0.2, 0.8)) -> pd.DataFrame:
    """One row per feature: what is known, and what would resolve it.

    This is a prioritisation of features whose unresolved identity changes the
    toxicological reading, not a verdict on any feature. Nothing here states
    that a feature is safe or unsafe: that would need a measured concentration,
    which this notebook does not have. The `next_step` column says what action
    would reduce the uncertainty that dominates the row -- identity, domain, or
    nothing -- and is a rule over the columns beside it, not a recommendation
    with any regulatory standing.
    """
    low, high = domain_bounds
    w = cand["weight"].fillna(0.0)
    grp = cand.assign(_w=w).groupby(col_group)
    share_domain = (grp.apply(lambda d: float(
        (d["_w"] * d["in_domain"].astype(float)).sum() / d["_w"].sum())
        if d["_w"].sum() > 0 else float("nan"), include_groups=False)
        .rename("weight_in_domain"))
    share_alert = (grp.apply(lambda d: float(
        (d["_w"] * d["genotox_alert"].astype(float)).sum() / d["_w"].sum())
        if d["_w"].sum() > 0 and "genotox_alert" in d else float("nan"),
        include_groups=False).rename("weight_alerting"))

    out = feat_p.merge(share_domain, on=col_group, how="left")                 .merge(share_alert, on=col_group, how="left")

    def _domain(v):
        if not np.isfinite(v):
            return "unevaluable"
        return "in" if v >= high else ("out" if v <= low else "partial")

    def _alert(v):
        if not np.isfinite(v):
            return "unevaluable"
        return "+" if v >= high else ("-" if v <= low else "+/-")

    out["domain"] = [_domain(v) for v in out["weight_in_domain"]]
    out["alert"] = [_alert(v) for v in out["weight_alerting"]]

    def _next(row):
        if row["domain"] == "out":
            return "outside the domain: do not rank"
        if row["covered_weight"] < 0.5:
            return ("resolve identity first" if row["identity_weighted_genotoxicity"] >= threshold
                    else "identity unresolved: score not informative")
        if row["identity_weighted_genotoxicity"] >= threshold:
            return "confirm identity against a standard"
        return "low priority"

    out["next_step"] = out.apply(_next, axis=1)
    return out.sort_values("identity_weighted_genotoxicity", ascending=False)               .reset_index(drop=True)
