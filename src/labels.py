"""Auditing and aggregation of EFSA genotoxicity labels.

The seven labels in the Genotoxicity field are not a binary. Only Positive and
Negative are informative; Ambiguous is a real EFSA conclusion (discordant
evidence) and must not be confused with absence of data, which is what
Not determined, No data and Not applicable indicate.

Aggregation by substance is a DECISION, not a fact of the data: a
substance with 3 Negative and 1 Positive can be labeled as positive (conservative
rule, out of toxicological caution) or as negative (majority vote).
That is why the rule is a notebook parameter and not a constant in the code.
"""

import pandas as pd

INFORMATIVE = ("Positive", "Negative")
AMBIGUOUS = ("Ambiguous",)
NO_DATA = ("Not determined", "No data", "Not applicable", "Other")

ORDER = list(INFORMATIVE) + list(AMBIGUOUS) + list(NO_DATA)

RULES = ("any_positive", "majority", "consensus")
AMBIGUOUS_HANDLING = ("exclude", "positive", "negative")


def distribution(df: pd.DataFrame, col: str = "Genotoxicity") -> pd.DataFrame:
    """Breakdown of labels with their semantic category."""
    n = df[col].value_counts()
    table = pd.DataFrame({"label": n.index, "n": n.to_numpy()})
    table["pct"] = (100 * table["n"] / len(df)).round(2)
    table["category"] = table["label"].map(
        lambda e: "informative" if e in INFORMATIVE
        else "ambiguous" if e in AMBIGUOUS
        else "no data"
    )
    order = {e: i for i, e in enumerate(ORDER)}
    return table.sort_values("label", key=lambda s: s.map(order)).reset_index(drop=True)


def contradictions(df: pd.DataFrame, key: str = "inchikey") -> pd.DataFrame:
    """Substances with both Positive and Negative conclusions at once.

    Returns one row per contradictory substance, with how many studies
    support each conclusion and which EFSA outputs they come from.
    """
    bina = df[df["Genotoxicity"].isin(INFORMATIVE) & df[key].notna()]
    g = bina.groupby(key)
    res = pd.DataFrame({
        "n_studies": g.size(),
        "n_pos": g["Genotoxicity"].apply(lambda s: (s == "Positive").sum()),
        "n_neg": g["Genotoxicity"].apply(lambda s: (s == "Negative").sum()),
    })
    res["name"] = g["CleanSubstanceName"].first()
    if "OutputID" in bina.columns:
        res["outputs"] = g["OutputID"].apply(lambda s: ", ".join(map(str, sorted(set(s)))))
    res = res[(res["n_pos"] > 0) & (res["n_neg"] > 0)]
    return res.reset_index().sort_values("n_studies", ascending=False)


def aggregate_by_substance(
    df: pd.DataFrame,
    key: str = "inchikey",
    rule: str = "any_positive",
    ambiguous: str = "exclude",
    min_studies: int = 1,
) -> pd.DataFrame:
    """Collapses study rows into a binary label per substance.

    rule:
      any_positive  any Positive -> 1; if none, any Negative -> 0
                      (conservative: it is the one that does not discard an isolated positive)
      majority         1 if n_pos > n_neg, 0 if n_neg > n_pos, drops if tied
      consensus        1 only if all Positive, 0 only if all Negative
    ambiguous:
      exclude   Ambiguous does not count (default)
      positive  Ambiguous adds to n_pos
      negative  Ambiguous adds to n_neg
    min_studies: discards substances with fewer informative conclusions.

    Returns one row per substance with y in {0, 1} and the count columns
    that justify that y. Substances with no assignable label come out with y NaN.
    """
    if rule not in RULES:
        raise ValueError(f"rule debe ser una de {RULES}")
    if ambiguous not in AMBIGUOUS_HANDLING:
        raise ValueError(f"ambiguous debe ser uno de {AMBIGUOUS_HANDLING}")

    use = list(INFORMATIVE) + (list(AMBIGUOUS) if ambiguous != "exclude" else [])
    sub = df[df["Genotoxicity"].isin(use) & df[key].notna()].copy()

    is_pos = sub["Genotoxicity"] == "Positive"
    is_neg = sub["Genotoxicity"] == "Negative"
    is_amb = sub["Genotoxicity"].isin(AMBIGUOUS)
    if ambiguous == "positive":
        is_pos = is_pos | is_amb
    elif ambiguous == "negative":
        is_neg = is_neg | is_amb
    sub["_pos"], sub["_neg"], sub["_amb"] = is_pos.astype(int), is_neg.astype(int), is_amb.astype(int)

    g = sub.groupby(key)
    out = pd.DataFrame({
        "n_studies": g.size(),
        "n_pos": g["_pos"].sum(),
        "n_neg": g["_neg"].sum(),
        "n_amb": g["_amb"].sum(),
        "name": g["CleanSubstanceName"].first(),
    })
    if "smiles" in sub.columns:
        out["smiles"] = g["smiles"].first()
    if "CAS" in sub.columns:
        out["CAS"] = g["CAS"].first()

    if rule == "any_positive":
        y = pd.Series(pd.NA, index=out.index, dtype="Float64")
        y[out["n_pos"] > 0] = 1.0
        y[(out["n_pos"] == 0) & (out["n_neg"] > 0)] = 0.0
    elif rule == "majority":
        y = pd.Series(pd.NA, index=out.index, dtype="Float64")
        y[out["n_pos"] > out["n_neg"]] = 1.0
        y[out["n_neg"] > out["n_pos"]] = 0.0
    else:  # consensus
        y = pd.Series(pd.NA, index=out.index, dtype="Float64")
        y[(out["n_pos"] > 0) & (out["n_neg"] == 0)] = 1.0
        y[(out["n_neg"] > 0) & (out["n_pos"] == 0)] = 0.0

    out["y"] = pd.array(y, dtype="Float64")
    out["contradictory"] = (out["n_pos"] > 0) & (out["n_neg"] > 0)
    out.loc[out["n_studies"] < min_studies, "y"] = pd.NA
    return out.reset_index()


def aggregation_summary(agg: pd.DataFrame) -> pd.DataFrame:
    labeled = agg[agg["y"].notna()]
    n_pos = int((labeled["y"] == 1).sum())
    n_neg = int((labeled["y"] == 0).sum())
    rows = [
        ("substances considered", len(agg)),
        ("substances with binary label", len(labeled)),
        ("positive", n_pos),
        ("negative", n_neg),
        ("prevalence of positives (%)", round(100 * n_pos / max(n_pos + n_neg, 1), 2)),
        ("contradictory substances", int(agg["contradictory"].sum())),
        ("substances with no assignable label", int(agg["y"].isna().sum())),
    ]
    return pd.DataFrame(rows, columns=["quantity", "n"])


def structure_attrition(df: pd.DataFrame, key: str = "CleanSubstanceName",
                        structure_col: str = "has_structure", id_col: str = "CAS",
                        rule: str = "any_positive", ambiguous: str = "exclude"):
    """Compares the substances a structure requirement keeps against those it drops.

    Modelling needs a structure, so every substance whose structure could not
    be resolved leaves the dataset. That exclusion is only harmless if it is
    unrelated to the label, and this is testable without any new data: the
    Genotoxicity conclusions are in the source rows whether or not a structure
    was ever found, so both groups can be labelled with the same rule and
    compared.

    Aggregation is by name rather than by InChIKey precisely because the
    dropped group has no InChIKey -- keying on the structure identifier would
    make the excluded substances unobservable, which is the bias being tested
    for.

    Returns ``(table, summary)``. `table` is one row per group with the label
    rate, its exact interval and the identifier coverage; `summary` carries the
    conditional odds ratio with its exact interval, Fisher's p, and the
    study-count comparison. The odds ratio is reported with its interval
    because "no significant difference" on a few hundred positives is a
    statement about power as much as about bias.
    """
    from scipy.stats import binomtest, contingency, fisher_exact, mannwhitneyu

    agg = aggregate_by_substance(df, key=key, rule=rule, ambiguous=ambiguous)
    has_structure = df.groupby(key)[structure_col].max().astype(bool)
    has_id = df.groupby(key)[id_col].apply(lambda c: c.notna().any())
    d = (agg.join(has_structure, on=key)
            .join(has_id.rename("has_identifier"), on=key))
    d = d[d["y"].notna()].copy()
    d["resolved"] = d[structure_col].astype(bool)

    rows = []
    for resolved, part in d.groupby("resolved"):
        n, pos = len(part), int(part["y"].sum())
        interval = binomtest(pos, n).proportion_ci(0.95)
        rows.append({
            "group": "structure resolved" if resolved else "structure unresolved",
            "n_substances": n, "n_positive": pos, "positive_rate": pos / n,
            "ci_low": float(interval.low), "ci_high": float(interval.high),
            "identifier_coverage": float(part["has_identifier"].mean()),
            "mean_studies": float(part["n_studies"].mean()),
            "single_study_share": float((part["n_studies"] < 2).mean()),
        })
    table = pd.DataFrame(rows).sort_values("group", ascending=False, ignore_index=True)

    kept, dropped = d[d["resolved"]], d[~d["resolved"]]
    counts = [[int(kept["y"].sum()), int(len(kept) - kept["y"].sum())],
              [int(dropped["y"].sum()), int(len(dropped) - dropped["y"].sum())]]
    odds = contingency.odds_ratio(counts)
    interval = odds.confidence_interval(0.95)
    strata = (d.assign(multi_study=d["n_studies"] >= 2)
               .groupby(["multi_study", "resolved"])["y"]
               .agg(["size", "mean"]).reset_index())
    summary = {
        "n_labelled": int(len(d)), "n_dropped": int(len(dropped)),
        "odds_ratio": float(odds.statistic),
        "odds_ratio_ci": (float(interval.low), float(interval.high)),
        "fisher_p": float(fisher_exact(counts)[1]),
        "studies_p": float(mannwhitneyu(kept["n_studies"], dropped["n_studies"]).pvalue),
        "strata": strata,
    }
    return table, summary
