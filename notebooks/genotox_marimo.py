# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "genotox-food-migrants[notebook] @ git+https://github.com/E4CE-UA/genotox-food-migrants",
#     "marimo",
# ]
# ///
import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _candidates = [Path.cwd(), Path.cwd().parent]
    try:
        _nb = mo.notebook_dir()
        if _nb is not None:
            _candidates.insert(0, Path(_nb).parent)
    except Exception:
        pass
    ROOT = next((c for c in _candidates if (c / "src" / "paths.py").exists()), None)
    if ROOT is not None and str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    # If src/paths.py is not beside the notebook - molab running the notebook
    # from a pip install of the package - fall through to the installed `src`.

    from src import (
        ambiguity,
        candidate_view,
        chemistry,
        data,
        environment,
        external,
        figures,
        fixture,
        gcms,
        integration,
        labels,
        models,
        paths,
        provenance,
        public_features,
        retention,
        splitting,
        ttc,
    )  # noqa: F401

    if ROOT is None:
        ROOT = Path(paths.ROOT)
    environment.set_seeds()
    figures.apply_style()
    return (
        ROOT,
        ambiguity,
        candidate_view,
        chemistry,
        data,
        environment,
        external,
        figures,
        fixture,
        gcms,
        integration,
        labels,
        mo,
        models,
        np,
        paths,
        provenance,
        public_features,
        retention,
        splitting,
        ttc,
    )


@app.cell
def _(figures, gcms_features, gcms_info, mo):
    _feat = gcms_features
    _acc = gcms_info.get("accession")
    _real = ("n_isomers_total" in _feat.columns
             and bool(_feat["n_isomers_total"].gt(0).any()))
    if _real:
        _n = _feat.loc[_feat["n_isomers_total"] > 0, "n_isomers_total"]
        _med = int(_n.median())
        _le10 = 100.0 * (_n <= 10).mean()
        _head = f"# {_med:,} compounds could be this one peak"
        _size = (f"In the public study loaded below ({_acc}), the **median** "
                 f"feature\'s formula has **{_med:,} real isomers in PubChem**, "
                 f"and only {_le10:.1f}% of features have ten or fewer.")
        _hero = figures.figure_isomer_hero(_feat, _acc or "")
    else:
        _head = "# Which compound is this peak?"
        _size = ("The public study could not be read here, so the sections "
                 "below run on a synthetic stand-in and the isomer counts "
                 "that open this notebook are not available. With network "
                 "access, or with `data/public_candidates.parquet` in place, "
                 "the median feature\'s formula resolves to hundreds or "
                 "thousands of real isomers in PubChem.")
        _hero = None
    _md = mo.md(f"""
    {_head}

    #### Genotoxicity of migrants: from the EFSA label to P(genotoxic | feature)

    A GC-MS peak is not a molecule. It is a mass spectrum with a formula and a
    ranked list of candidate identities, none of them certain. {_size}
    No public retention index orders them — that was tested and it cannot
    (section 13).

    So the honest input to a hazard model is not one structure per peak. It is
    a distribution over hundreds to thousands of them, and the question becomes
    **"is this peak something that damages DNA?"** answered without knowing
    which candidate is the right one. This notebook carries that whole
    distribution through the model instead of collapsing it onto a name — and
    reports how much of each answer the identity actually supports.
    """)
    mo.vstack([_md, _hero] if _hero is not None else [_md])
    return


@app.cell
def _(ROOT, environment, mo, paths):
    mo.md(f"""
    ## How the notebook answers it, and where each move can fail

    > **Sections 13 to 18 run on a public MetaboLights study**, chosen in the
    > parameter bar: its deposited peak assignments, the real isomers of each
    > formula from PubChem, and the identities its depositors confirmed. With no
    > network and no shipped bundle they fall back to a synthetic stand-in and
    > say so on screen, so a number is never left ambiguous about where it came
    > from. What no public deposit provides is a per-candidate score, and
    > section 13 shows the audit behind that claim rather than asserting it.

    Three uncertainties stack on top of each other, and the notebook keeps them
    apart instead of collapsing them into one figure: **which compound is this**
    (identity), **is that compound genotoxic** (hazard), and **what limit would
    change if it were** (consequence). Each has its own section, its own control
    in the parameter bar, and its own way of failing.

    Getting there takes three moves:

    1. **Get structures.** EFSA publishes genotoxicity conclusions about
       substance *names*. A model needs *structures*. Closing that gap took
       10,827 rows through the PubChem API — section 1 shows exactly how many
       survived and where the rest were lost.
    2. **Learn from them.** Aggregate the conclusions into one label per
       substance, split by scaffold so congeners cannot leak between train and
       test, and check that the probabilities mean what they say.
    3. **Push it back onto the peaks.** Marginalize the prediction over each
       peak's candidate identities -- uniformly on a public source, because
       there is nothing published to weight them by, and under a weighting
       scheme you choose if you attach a local export -- and report how much
       of each feature's identity the result rests on.

    Every debatable step — the aggregation rule, what to do with *Ambiguous*,
    the applicability-domain cutoff, how spectral evidence becomes a weight —
    is a **control in the parameter bar**, not a constant buried in the code.
    Move one and only the cells downstream of it re-run.

    `{ROOT.name}` | data `{paths.relative(paths.EFSA_CSV)}` | seed `{environment.SEED}`
    """)
    return


@app.cell
def _(environment):
    environment.versions()
    return


@app.cell
def _(mo):
    mo.md("""
    ## 1. Where the data comes from

    This is the part that does not appear in the final table and took the
    longest. EFSA's OpenFoodTox export identifies substances the way a
    regulator does — by name, and sometimes by CAS. It carries no structures.
    Nothing here is modelable until every row has been turned into a molecule.

    `src/resolver.py` does that against the **PubChem PUG REST API**, trying
    three routes per row and stopping at the first that works:

    | order | route | why it is not first |
    |---|---|---|
    | 1 | `PubChem_CID` already in the EFSA file | free, but present for only part of the rows |
    | 2 | `CAS -> CID` via `xref/RN` | CAS is precise, but missing or malformed in many rows |
    | 3 | `name -> CID` via `compound/name` | last resort: chemical names are ambiguous and this is where most lookups fail |

    Then, in batches of 100 CIDs, it retrieves SMILES, InChIKey, formula and
    molecular weight. Four details that matter more than they look:

    - **Every answer is cached to `cache/*.json`, including the failures.** A
      failed lookup is an answer too; caching it means a rerun does not
      re-ask PubChem the same 1,971 hopeless questions.
    - **Retries with exponential backoff** on `429` and `5xx`, because the
      public endpoint throttles at roughly 5 requests/second.
    - **Excel escape decoding** (`_x0028_` back into `(`), otherwise a
      substantial number of names never match anything.
    - **No standardization at this stage.** The full SMILES is stored, salts
      and mixtures included, so that discarding a counter-ion stays a visible
      decision in section 6 rather than an invisible one here.
    """)
    return


@app.cell
def _(mo, provenance):
    api_table = provenance.api_effort()
    mo.vstack([
        mo.md("**API lookups now cached on disk.** Each row is a question that "
              "never has to be asked again; `no_match` are the ones PubChem "
              "answered with nothing, which is itself worth caching."),
        api_table,
    ])
    return


@app.cell
def _(efsa, efsa_with_struct, figures, labels, modelable, provenance):
    _informative = int(labels.distribution(efsa_with_struct, "Genotoxicity")
                       .query("category == 'informative'")["n"].sum())
    provenance_funnel = provenance.funnel(
        efsa,
        efsa_with_struct,
        informative=_informative,
        substances=efsa_with_struct["inchikey"].nunique(),
        modelable=len(modelable),
    )
    provenance_routes = provenance.resolution_routes(efsa)
    fig_provenance = figures.figure_provenance(provenance_funnel, provenance_routes)
    fig_provenance
    return (provenance_funnel,)


@app.cell
def _(mo, provenance_funnel):
    mo.vstack([
        mo.md("The attrition is not a bug to be fixed — it is what the dataset "
              "actually supports. Reporting a model trained on the last row of "
              "this table while quoting the size of the first would be the "
              "easiest lie available in this project."),
        provenance_funnel,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Loading: a row is not a substance

    Each row is the genotoxicity conclusion of an EFSA output about a
    substance, so there are several rows per substance and the same substance
    can appear with different conclusions in different years.
    """)
    return


@app.cell
def _(data):
    efsa = data.load_efsa()
    load_table = data.load_summary(efsa)
    load_table
    return (efsa,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Label audit

    Only `Positive` and `Negative` are informative. `Ambiguous` is a real
    EFSA conclusion (discordant evidence) and is not the same as the absence
    of data indicated by `Not determined`, `No data` and `Not applicable`: confusing
    the two turns a "we don't know" into a "it is not genotoxic".
    """)
    return


@app.cell
def _(data, efsa, labels):
    efsa_with_struct = data.with_structure_only(efsa)
    labels_table = labels.distribution(efsa, "Genotoxicity").merge(
        labels.distribution(efsa_with_struct, "Genotoxicity"),
        on=["label", "category"],
        how="outer",
        suffixes=("_all", "_with_structure"),
    )
    contradictions_table = labels.contradictions(efsa_with_struct)
    return contradictions_table, efsa_with_struct, labels_table


@app.cell
def _(contradictions_table, efsa, figures, labels):
    fig_labels = figures.figure_labels(
        labels.distribution(efsa, "Genotoxicity"), len(contradictions_table)
    )
    fig_labels
    return


@app.cell
def _(contradictions_table, labels_table, mo):
    mo.vstack(
        [
            mo.md("**Label distribution** (all rows versus those with resolved structure)"),
            labels_table,
            mo.md(
                f"**Substances with both `Positive` and `Negative`: {len(contradictions_table)}.** "
                "This is not a data error: these are substances reassessed with different criteria or assays. "
                "The aggregation rule decides what to do with them."
            ),
            contradictions_table.head(15),
        ]
    )
    return


@app.cell
def _(mo):
    sel_rule = mo.ui.dropdown(
        options=["any_positive", "majority", "consensus"],
        value="any_positive",
        label="Aggregation rule per substance",
    )
    sel_ambiguous = mo.ui.dropdown(
        options=["exclude", "positive", "negative"],
        value="exclude",
        label="Handling of Ambiguous",
    )
    sl_test_frac = mo.ui.slider(
        start=0.1, stop=0.4, step=0.05, value=0.2, label="Test fraction", show_value=True
    )
    sl_n_seeds = mo.ui.slider(
        start=1, stop=10, step=1, value=5, label="Number of seeds", show_value=True
    )
    sel_model = mo.ui.dropdown(
        options=["logistic", "lgbm", "majority_class"],
        value="lgbm",
        label="Model to inspect (calibration and domain)",
    )
    sl_ad_threshold = mo.ui.slider(
        start=0.1, stop=0.9, step=0.05, value=0.4,
        label="Applicability domain threshold (Tanimoto to nearest neighbour)",
        show_value=True,
    )
    sel_weight_method = mo.ui.dropdown(
        options=["uniform", "softmax", "proportional", "top1_only"],
        value="uniform",
        label="Normalization of identity evidence",
    )
    sl_temperature = mo.ui.slider(
        start=0.2, stop=3.0, step=0.1, value=1.0, label="Softmax temperature", show_value=True
    )
    return (
        sel_ambiguous,
        sel_model,
        sel_rule,
        sel_weight_method,
        sl_ad_threshold,
        sl_n_seeds,
        sl_temperature,
        sl_test_frac,
    )


@app.cell
def _(
    mo,
    sel_ambiguous,
    sel_model,
    sel_rule,
    sel_weight_method,
    sl_ad_threshold,
    sl_n_seeds,
    sl_temperature,
    sl_test_frac,
):
    mo.accordion(
        {
            "Analysis parameters": mo.vstack(
                [
                    mo.md("**Labels**"),
                    sel_rule,
                    sel_ambiguous,
                    mo.md("**Splitting and models**"),
                    sl_test_frac,
                    sl_n_seeds,
                    sel_model,
                    sl_ad_threshold,
                    mo.md("**Candidate weighting** — applies to a local export "
                          "only. A public deposit publishes no per-candidate "
                          "score, so on the default path these two are "
                          "overridden to `uniform` and the section that uses "
                          "them says so on screen."),
                    sel_weight_method,
                    sl_temperature,
                ]
            )
        },
        lazy=False,
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 4. Aggregation by substance

    Aggregation is a decision, not a fact of the data. A substance with three
    `Negative` and one `Positive` can be labeled as positive (conservative
    criterion, the usual one in risk assessment) or as negative (majority
    vote), and the number of positives changes with that choice.
    """)
    return


@app.cell
def _(efsa_with_struct, labels, sel_ambiguous, sel_rule):
    aggregated = labels.aggregate_by_substance(
        efsa_with_struct,
        key="inchikey",
        rule=sel_rule.value,
        ambiguous=sel_ambiguous.value,
    )
    aggregation_table = labels.aggregation_summary(aggregated)
    aggregation_table
    return (aggregated,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Who gets dropped when the structure cannot be resolved?

    A model needs a structure, so every substance whose structure could not be
    resolved leaves the dataset before any model sees it. That exclusion is
    usually reported as a coverage number and left there. It deserves better,
    because a model trained on the survivors inherits whatever made them
    survive.

    The question is testable here with no additional data. The Genotoxicity
    conclusion sits in the source rows whether or not a structure was ever
    found, so both groups can be labelled with the same rule and compared. One
    detail decides whether the test works at all: the aggregation below is
    keyed on the substance **name**, not the InChIKey. Keying on the structure
    identifier would make the dropped substances unobservable, which is
    precisely the bias being tested for.
    """)
    return


@app.cell
def _(efsa, labels):
    attrition_table, attrition_summary = labels.structure_attrition(efsa)
    attrition_table
    return attrition_summary, attrition_table


@app.cell
def _(attrition_summary, attrition_table, figures):
    fig_attrition = figures.figure_attrition(attrition_table, attrition_summary)
    fig_attrition
    return


@app.cell
def _(attrition_summary, attrition_table, mo):
    _d = attrition_table.set_index("group")
    _kept = _d.loc["structure resolved"]
    _dropped = _d.loc["structure unresolved"]
    _lo, _hi = attrition_summary["odds_ratio_ci"]
    _strata = attrition_summary["strata"]
    _single = _strata[~_strata["multi_study"]].set_index("resolved")["mean"]
    _multi = _strata[_strata["multi_study"]].set_index("resolved")["mean"]
    _flips = (_single[True] > _single[False]) != (_multi[True] > _multi[False])
    mo.md(
        f"""
        **The label.** {100 * _kept["positive_rate"]:.1f} % of the
        {_kept["n_substances"]:.0f} modelled substances are positive against
        {100 * _dropped["positive_rate"]:.1f} % of the
        {_dropped["n_substances"]:.0f} dropped ones -- an odds ratio of
        {attrition_summary["odds_ratio"]:.2f}, interval
        [{_lo:.2f}, {_hi:.2f}], p = {attrition_summary["fisher_p"]:.2f}. No
        detectable enrichment, and the interval is the honest part of that
        sentence: with {_dropped["n_positive"]:.0f} positives in the dropped
        group, odds up to roughly {_hi:.1f} times the modelled group's are
        still compatible with this data. "Not significant" here is partly a
        statement about power.

        **The mechanism.** Identifier coverage is
        {100 * _kept["identifier_coverage"]:.0f} % among the modelled
        substances and {100 * _dropped["identifier_coverage"]:.0f} % among the
        dropped ones. The missingness is not idiosyncratic: what leaves is
        essentially every entry that arrives as a name with no registry number
        -- trade names, mixtures, ill-defined materials. So the exclusion is
        strongly non-random in *kind* while being undetectable in the label.

        **The strata.** Dropped substances also carry fewer informative
        studies ({_dropped["mean_studies"]:.2f} against
        {_kept["mean_studies"]:.2f} on average, Mann-Whitney
        p = {attrition_summary["studies_p"]:.1e}), and the positive rate
        depends strongly on that count under the any-positive rule.
        """
        + (f"""
        Within single-study substances the modelled group is the *more*
        positive one ({100 * _single[True]:.1f} % against
        {100 * _single[False]:.1f} %); within multi-study substances the
        comparison reverses ({100 * _multi[True]:.1f} % against
        {100 * _multi[False]:.1f} %). The marginal null above is therefore a
        mix of two strata that disagree in sign, over a stratifying variable
        that itself differs between the groups -- which is a good reason to
        report the strata rather than the pooled p-value alone.
        """ if _flips else f"""
        The two strata agree in sign here ({100 * _single[True]:.1f} % against
        {100 * _single[False]:.1f} % for single-study substances,
        {100 * _multi[True]:.1f} % against {100 * _multi[False]:.1f} % for
        multi-study ones), so the pooled comparison is not hiding a
        cancellation.
        """)
        + """
        **What this licenses, and what it does not.** It does not license
        calling the exclusion harmless. It licenses a scope statement: the
        prevalence this notebook models is not detectably optimistic relative
        to the substances it had to drop, and the operating point derived later
        applies to registered single substances -- not to the unregistered
        entries, which are a different population and are absent from every
        metric that follows. Two things stay untested. Whether more curation
        effort would resolve those structures, which is a question about
        sourcing rather than about the data. And anything about substances with
        no informative conclusion at all: the labelling rule drops them from
        both groups, so this comparison is conditional on having a label.
        """
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 6. Standardization of structures

    `Cleanup` -> `FragmentParent` -> `Uncharger` -> `TautomerEnumerator`.
    Each step can change the InChIKey, so the number of structures that
    change is counted instead of silently cleaning them. `FragmentParent` keeps the
    largest fragment: correct for salts, but for a real mixture it discards
    information, which is why multi-fragment entries are counted separately.

    The result is cached in `cache/` and the cell only recalculates the new
    structures.
    """)
    return


@app.cell
def _(aggregated, chemistry, paths):
    standardized = chemistry.standardize_table(
        aggregated, smiles_col="smiles", cache_path=paths.CACHE_STANDARDIZATION
    )
    standardization_table = chemistry.standardization_summary(standardized)
    standardization_table
    return (standardized,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Optional merging with external sources

    Leave one CSV per source in `data/external/` with columns `smiles,y,source`
    (Hansen Ames, ISSSTY/ISSCAN, EURL ECVAM...). They are standardized with the same
    pipeline and merged by standardized InChIKey, keeping the column
    `source`.

    Endpoint warning: Ames is bacterial mutagenicity and EFSA's conclusion
    aggregates several in vitro and in vivo assays. Merging expands n at the cost of
    mixing endpoints; that is why it is optional and reversible.
    """)
    return


@app.cell
def _(external, mo, paths, standardized):
    external_catalog = external.catalog()
    external_loaded = external.load(cache_path=paths.CACHE_STANDARDIZATION)
    merged, source_conflicts = external.merge(standardized, external_loaded)
    _msg = (
        f"External sources found: {len(external_catalog)}. "
        f"Rows added: {len(merged) - len(standardized)}. "
        f"Label conflicts with EFSA: {len(source_conflicts)} (EFSA wins)."
        if len(external_catalog)
        else f"No external sources in `{paths.relative(external.DIR_EXTERNAL)}`: continuing with EFSA only."
    )
    mo.md(_msg)
    return (merged,)


@app.cell
def _(chemistry, merged, mo):
    modelable = merged[merged["y"].notna() & merged["ok"].fillna(False)].copy()
    modelable = modelable.drop_duplicates(subset="inchikey_std").reset_index(drop=True)
    modelable["scaffold"] = [chemistry.scaffold(s) for s in modelable["smiles_std"]]
    y = modelable["y"].astype(int).to_numpy()
    mo.md(
        f"**Modelable set: {len(modelable)} substances, {int(y.sum())} positive "
        f"({100 * y.mean():.1f} %), {modelable['scaffold'].nunique()} Bemis-Murcko scaffolds.**"
    )
    return modelable, y


@app.cell
def _(mo):
    mo.md("""
    ## 8. Splitting by scaffold

    A random split would inflate the metrics: there are entire families of
    congeners, and it is enough for one member to be in training to recognize
    its siblings. Here each scaffold group goes entirely to one side. The order of
    the groups is shuffled with the seed, so that different seeds give
    different splits and the metrics are summarized with median and IQR.
    """)
    return


@app.cell
def _(modelable, sl_n_seeds, sl_test_frac, splitting, y):
    seeds = tuple(range(sl_n_seeds.value))
    splits = splitting.multiseed_splits(
        modelable["scaffold"], seeds, frac_test=sl_test_frac.value
    )
    splits_table = splitting.split_summary(
        y, modelable["scaffold"], seeds, frac_test=sl_test_frac.value
    )
    splits_table
    return seeds, splits


@app.cell
def _(X, chemistry, figures, modelable, np, paths, splits, y):
    chem_embedding = chemistry.embed_tsne(
        X, keys=modelable["inchikey_std"], cache_path=paths.CACHE_STANDARDIZATION
    )
    _test_mask = np.zeros(len(modelable), dtype=bool)
    _test_mask[splits[0][2]] = True
    fig_chemspace = figures.figure_chemspace(
        chem_embedding, y, _test_mask, modelable["scaffold"].nunique()
    )
    fig_chemspace
    return


@app.cell
def _(mo):
    mo.md("""
    ## 9. Baselines and metrics suited to the imbalance

    With a prevalence of positives around 7 % in the modelable set, accuracy
    says nothing:
    always predicting "not genotoxic" is right 93 % of the time. The baseline
    for PR-AUC is the prevalence, not 0.5, and that is why we also report
    `pr_auc_lift` (PR-AUC divided by the prevalence).

    MCC and recall are given at two thresholds: 0.5 and the prevalence threshold,
    which flags as positive exactly as many compounds as there are positives in
    test. The second measures the ranking without giving the model the choice
    of threshold. The majority class baseline is selected by rank, not by
    comparison with the threshold, so that its recall does not come out as 1.0 due to ties.
    """)
    return


@app.cell
def _(chemistry, modelable, models, splits, y):
    ecfp4_fingerprints = chemistry.fingerprints(modelable["smiles_std"], radius=2, n_bits=2048)
    X = chemistry.matrix(ecfp4_fingerprints, n_bits=2048)
    results, predictions = models.evaluate(X, y, splits)
    model_summary = models.median_iqr(results)
    model_summary
    return X, ecfp4_fingerprints, predictions, results


@app.cell
def _(figures, modelable, models, results):
    fig_metrics = figures.figure_metrics(results, models.MODELS, len(modelable))
    fig_metrics
    return


@app.cell
def _(mo):
    mo.md("""
    ## 10. Calibration

    PR-AUC and MCC measure the ranking; they do not say whether a 0.8 means an 80 %
    probability. The reliability diagram groups the predictions from all
    seeds and compares the mean probability of each bin with the actual fraction of
    positives. ECE is the mean difference between the two, weighted by bin
    occupancy.
    """)
    return


@app.cell
def _(figures, models, np, predictions, seeds, sel_model):
    y_calibration = np.concatenate([predictions[(sel_model.value, s)][0] for s in seeds])
    p_calibration = np.concatenate([predictions[(sel_model.value, s)][1] for s in seeds])
    reliability_table = models.reliability_curve(y_calibration, p_calibration, n_bins=10)
    global_ece = models.ece(y_calibration, p_calibration, n_bins=10)
    fig_calibration = figures.figure_calibration(reliability_table, global_ece, sel_model.value)
    fig_calibration
    return p_calibration, y_calibration


@app.cell
def _(mo):
    mo.md("""
    ## 11. Domain of applicability

    Tanimoto similarity (ECFP4) of each test compound to its nearest
    neighbour in the training set. A compound below the threshold is outside the
    chemical space seen and its probability should not be reported as a
    prediction, even though the model produces it anyway.

    The curve on the right answers the useful question: if I only accept
    predictions above a threshold, how much coverage do I lose and how much does
    the PR-AUC of the accepted subset gain.
    """)
    return


@app.cell
def _(
    chemistry,
    ecfp4_fingerprints,
    figures,
    models,
    np,
    predictions,
    seeds,
    sel_model,
    sl_ad_threshold,
):
    _sim, _yv, _pv = [], [], []
    for _s in seeds:
        _y_te, _p_te, _idx_te = predictions[(sel_model.value, _s)]
        _idx_tr = np.setdiff1d(np.arange(len(ecfp4_fingerprints)), _idx_te)
        _sim.append(
            chemistry.max_tanimoto(
                [ecfp4_fingerprints[i] for i in _idx_te], [ecfp4_fingerprints[i] for i in _idx_tr]
            )
        )
        _yv.append(_y_te)
        _pv.append(_p_te)
    neighbour_similarity = np.concatenate(_sim)
    y_ad = np.concatenate(_yv)
    p_ad = np.concatenate(_pv)
    ad_curve = models.domain_curve(neighbour_similarity, y_ad, p_ad)
    fig_domain = figures.figure_domain(neighbour_similarity, ad_curve, sl_ad_threshold.value)
    fig_domain
    return (neighbour_similarity,)


@app.cell
def _(mo, neighbour_similarity, np, sl_ad_threshold):
    mo.md(f"""
    With threshold {sl_ad_threshold.value:.2f},
    **{100 * np.mean(neighbour_similarity >= sl_ad_threshold.value):.1f} %**
    of test compounds remain within domain.
    Median Tanimoto to nearest neighbour: {np.median(neighbour_similarity):.3f}.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 12. Phase 2 and 3 (not implemented yet)

    - **Phase 2. Chemprop in ensemble + conformal prediction.** Chemprop is not
      in the environment; adding it brings in torch. The conformal split gives intervals with
      guaranteed coverage, which is what's missing to be able to say "this
      compound is positive with 90% confidence".
    - **Phase 3. MEGAN.** Learned fragment-based explanation instead of SHAP
      over fingerprint bits.

    Before that investment it's worth looking at the number of positives in the
    modelable set: with just over a hundred positives and scaffold splitting, a
    graph network doesn't usually beat ECFP4 with gradient boosting, and the variance
    between seeds is greater than the difference between architectures.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 13. Identity uncertainty: an identity-weighted genotoxicity score

    The notebook does not identify compounds and does not rerun any
    identification. It reads a **public GC-MS study from MetaboLights** --
    every feature the depositors reported, with the formula and the identity
    they assigned to it -- builds each feature's candidate set from the **real
    isomers of its formula in PubChem**, and marginalizes the toxicity
    prediction over that set:

    $$S(\text{feature}) = \sum_k w_k \, P(\text{genotoxic} \mid \text{candidate}_k)$$

    The left-hand side is deliberately not written $P(\text{genotoxic} \mid
    \text{feature})$. That sum is a posterior probability only if two things
    hold: the $w_k$ are a probability distribution over the feature's identity,
    and each $P(\text{genotoxic} \mid \text{candidate}_k)$ is calibrated. The
    second is testable and section 10 tests it. The first is **not claimed
    anywhere in this notebook** -- the weights are a normalization assumption
    over a ranking, as the paragraph below sets out. So $S$ is an expectation
    under an assumed identity distribution: a score that orders features, not a
    probability that a feature is genotoxic.

    **There is no per-candidate evidence score to weight by, and that is a
    finding rather than a gap in the plumbing.** A metabolite assignment file
    publishes one identity per feature and no score for any alternative: the
    ranking a spectral library computes over candidates stays inside the
    vendor software and is not what depositors archive. So on public data the
    `evidence` column exists and is empty, and the only weighting the source
    supports is equal weight over the candidate set. The section still carries
    the coverage of the weights, because a feature whose candidates are partly
    outside the model's applicability domain has part of its identity
    unevaluated: coverage 0.4 means 60% of the identity mass was not scored.

    **Uniform is therefore not a fallback here; it is the only scheme the
    public source can support** -- and it is also the position this section
    would take on an export that did carry scores. The objection to adopting an
    upstream score as a weight -- that
    a quantity nobody is willing to call a probability does not become one by
    being consumed downstream -- applies with equal force to a softmax over
    `evidence`, which additionally invents a temperature on top of a sum of
    scores. What survives the objection is equal weight across a feature's
    candidate set -- not because equal weight is the position of maximum
    ignorance, which it is not in any probabilistic sense, but because it is
    the one option that declines to treat a ranking as a probability. A
    spectral match score orders candidates; it does not state how much more
    likely the first is than the second. Weighting by it imports a calibration
    that has not been demonstrated; equal weight imports none. Every other
    option in the dropdown is a departure from it, and section 15 is where
    that departure has to earn itself -- on the controlled bench, which is the
    only place a candidate ranking exists to be ablated at all.

    ### Why marginalize at all

    Because one identity per feature is not available to be had, and the reason
    lies in the measurement rather than in the processing.

    Electron-ionization GC-MS at nominal mass produces spectra dominated by
    ubiquitous fragment ions. Isomers and homologues share them, so a match
    against a spectral library ranks candidates without pinning one structure
    out of the list. Exact mass would narrow it -- a formula constrains the
    candidate set hard -- but that needs a mass axis calibrated well enough to
    separate formulae, which is a property of the acquisition and cannot be
    recovered afterwards by better code.

    Retention index is the usual second line of evidence, and it carries a trap
    worth stating explicitly: if the RI scale is anchored on peaks that the same
    procedure identified, those peaks recover their own reference value and the
    agreement is not independent evidence. Anchoring on injected n-alkanes, or
    comparing against a library measured elsewhere, is what makes it
    independent.

    **This notebook tested whether a public retention index can supply the
    missing ranking, and it cannot.** Harvesting every public GC study with an
    assignment file gives 1366 structure-index pairs over 632 distinct
    structures, once each study's retention column is calibrated against the
    studies that publish an explicit index (`src/retention.py`). A model
    trained on those pairs, evaluated on scaffold-disjoint test sets, reaches
    MAE 265 index units against 465 for predicting the mean -- so it does learn
    something -- but the between-study dispersion of the *same* structure is
    only 27 units, so it sits an order of magnitude above the floor. And the
    global figure is the wrong one to look at: retention index is dominated by
    molecular size, which is identical across the isomers of a formula by
    construction. Measured where it matters, on out-of-fold predictions inside
    a fixed formula, the median rank correlation with the measured index is
    **0.00** over the 16 formulas with at least three measured isomers, and the
    predictor's spread inside a formula (235 units) is wider than the real one
    (105). Section 19 carries this as a measured limitation rather than an
    assumed one.

    So the inference is not "the identifications are unreliable, work around
    them". It is: **a molecular formula is what this measurement constrains
    tightly, so a model consuming these features has to carry the identity
    uncertainty forward over the isomers of that formula instead of collapsing
    it onto the assigned structure.** That is what this section does, and the
    size of what gets carried forward is not a modelling choice: for the
    default study the median formula has **1392 real isomers in PubChem**, and
    only 0.4% of its formulae have ten or fewer. The candidate set is capped
    below that by the control above, so every capped feature carries the size
    of its whole isomer space and the score over it is a Monte Carlo estimate
    of the expectation, not a sum over an enumerated list.

    One honesty note on the argument above: it is a general property of the
    technique, not a measurement of any particular batch. The spectra are not
    available here, so this notebook cannot quantify it.

    ### The identity gate comes before the weighting

    Identification confidence arrives as a level on a published scale. The
    column this notebook reads reports **Metabolomics Standards Initiative
    levels** (Sumner et al., *Metabolomics* 3:211, 2007), and that is the scale
    used throughout, including in the identity gate below.

    It is worth separating it from a scale it is often merged with. Schymanski
    et al. (*Environ. Sci. Technol.* 48:2097, 2014) propose a **different**
    five-tier scheme for communicating identification confidence in
    high-resolution non-target work. It is not a refinement of the MSI levels
    and the tiers are not interchangeable: both call a match against an
    authentic standard level 1, but below that the criteria and the granularity
    diverge — Schymanski splits probable structures into library-match and
    diagnostic-evidence cases, where MSI has a single putative-annotation tier.
    Reporting a number without naming its scale is therefore ambiguous, and
    mapping one onto the other is a claim that needs stating rather than
    assuming. Nothing here maps them; the scheme in use is MSI.

    Either way the levels are categorical statements about what kind of
    evidence exists, not numbers, and treating them as numbers is the exact
    confusion both scales were written to prevent. Applying a softmax to every
    feature would manufacture uncertainty where the scale says there is none.

    So the weighting is gated. A feature at **MSI level 1** has its identity
    established against an authentic standard injected on the same instrument:
    there is one compound, not a distribution over five, and its weight goes
    entirely to the confirmed structure. Marginalization applies only to the
    features whose identity is genuinely ambiguous, which is the only place it
    was ever justified.

    This is not a softening of the method. It narrows it to its defensible
    range, and it has a second consequence that section 15 depends on: the
    level-1 subset is a reference established **independently of the candidate
    ranking**, which is exactly what the ablation needs.

    ### What the public deposits actually publish

    The gate and the marginalization each need a different column, and in
    public GC-MS deposits those two columns travel apart. Of the seven studies
    screened here, **only two put anything in the identification-level
    column**, and the five that give a formula for every single feature state
    no level at all -- so in those five the gate never fires and every feature
    is marginalized.

    The one study that does report a level for every feature inverts the
    problem: 105 of its 154 features are MSI:1, and 104 of those carry a
    formula, so the gate fires and the marginalization is skipped. The other
    49 it calls MSI:4, and **not one of those 49 publishes a formula** -- there
    is nothing to marginalize over exactly where the identity is least
    certain. The two arms of this section therefore cannot be exercised by one
    study, which is why the table above shows all seven.

    The second study also shows the scale confusion above in the wild: it
    fills the column for 12 of its 51 features, and only two of those values
    fall on the MSI scale. Nine say **5** and one says **0**, neither of which
    exists on a four-tier scale, so that study is reporting some other scheme
    and those cells are left unparsed rather than read as MSI levels.

    A local GC-MS export can still be attached, in which case it takes
    precedence over the public study; without one the section runs on public
    data, which is the default.
    """)
    return


@app.cell
def _(mo, public_features):
    sel_study = mo.ui.dropdown(
        options={f"{a} - {note}": a for a, _, note in public_features.SCREENED},
        value=f"{public_features.SCREENED[0][0]} - {public_features.SCREENED[0][2]}",
        label="Public GC-MS study",
    )
    sl_n_public = mo.ui.slider(
        5, 50, step=5, value=20, show_value=True,
        label="candidate isomers per feature (the isomer space is larger; see below)",
    )
    mo.vstack([sel_study, sl_n_public])
    return sel_study, sl_n_public


@app.cell
def _(mo, public_features):
    complementarity_table = public_features.complementarity()
    mo.vstack([
        mo.md("**What each screened study can support: a formula to marginalize "
              "over, an identification level to gate on, or neither**"),
        complementarity_table,
    ])
    return (complementarity_table,)


@app.cell
def _(gcms, mo, paths, sel_study):
    gcms_source = ("export" if (gcms.available() or paths.GCMS_CANDIDATES.exists())
                   else "public")
    mo.md(
        "A local GC-MS export is attached, so it takes precedence: candidates read "
        f"from `{paths.relative(paths.GCMS_CANDIDATES)}`."
        if gcms_source == "export"
        else "No local export is attached, so the section runs on **public "
             f"MetaboLights data** ({sel_study.value}). This is the default path."
    )
    return (gcms_source,)


@app.cell
def _(
    X,
    chemistry,
    ecfp4_fingerprints,
    fixture,
    gcms,
    gcms_source,
    integration,
    modelable,
    models,
    np,
    paths,
    public_features,
    sel_model,
    sel_study,
    sel_weight_method,
    sl_ad_threshold,
    sl_n_public,
    sl_temperature,
    y,
):
    if gcms_source == "export":
        if not paths.GCMS_CANDIDATES.exists():
            gcms.export()
        gcms_candidates, gcms_features, gcms_info = gcms.load_export()
        gcms_info = dict(gcms_info, is_fixture=False, has_evidence=True)
    else:
        try:
            # The label table is passed so a candidate that already has an
            # experimental answer can be counted as such. Almost none do, and
            # that count is the experimental gap this notebook exists to
            # measure rather than assert.
            gcms_candidates, gcms_features, gcms_info = public_features.build(
                sel_study.value, n_candidates=int(sl_n_public.value),
                labels=modelable.rename(columns={"y": "genotoxic"}),
            )
        except Exception as _err:   # no bundle and no network: say so, do not pretend
            gcms_candidates, gcms_features, gcms_info = fixture.build()
            gcms_info = dict(gcms_info, fetch_error=str(_err))

    # With no per-candidate score in the source there is nothing for a softmax
    # to act on, so the control is overridden rather than silently applied to
    # an empty column.
    _method = (sel_weight_method.value if gcms_info.get("has_evidence", True)
               else "uniform")

    # The model is the one trained above, applied to candidate structures.
    # This part is real regardless of where the candidate list came from.
    _clf = models.build(sel_model.value, 0).fit(X, y)
    _fps_cand = chemistry.fingerprints(gcms_candidates["smiles"])
    _val = [i for i, f in enumerate(_fps_cand) if f is not None]
    _p = np.full(len(_fps_cand), np.nan)
    if _val:
        _p[_val] = _clf.predict_proba(chemistry.matrix([_fps_cand[i] for i in _val]))[:, 1]
    _sim = np.full(len(_fps_cand), np.nan)
    if _val:
        _sim[_val] = chemistry.max_tanimoto([_fps_cand[i] for i in _val], ecfp4_fingerprints)

    gcms_candidates = gcms_candidates.assign(
        p_genotox=_p,
        tanimoto_neighbour=_sim,
        in_domain=_sim >= sl_ad_threshold.value,
        weight=integration.gated_weights(
            gcms_candidates, gcms_features,
            method=_method, temperature=sl_temperature.value,
        ),
    )
    feature_probabilities = integration.probability_per_feature(
        gcms_candidates,
        gcms_candidates["p_genotox"],
        gcms_candidates["weight"],
        in_domain=gcms_candidates["in_domain"],
    )
    coverage_table = integration.coverage_summary(feature_probabilities)
    gate_table = integration.gate_summary(gcms_candidates, gcms_features)
    return (
        coverage_table,
        feature_probabilities,
        gate_table,
        gcms_candidates,
        gcms_features,
        gcms_info,
    )


@app.cell
def _(
    coverage_table,
    feature_probabilities,
    fixture,
    gate_table,
    gcms_info,
    mo,
    public_features,
    sel_weight_method,
):
    if gcms_info.get("fetch_error"):
        _banner = (fixture.banner(gcms_info) + "\n\n**The public study could not be "
                   f"read** (`{gcms_info['fetch_error']}`), so the bench above is "
                   "standing in. Rerun with network access, or with "
                   "`data/public_candidates.parquet` in place, for the real study.")
    elif gcms_info.get("is_fixture"):
        _banner = fixture.banner(gcms_info)
    else:
        _banner = public_features.banner(gcms_info)
    if not gcms_info.get("has_evidence", True) and sel_weight_method.value != "uniform":
        _banner += (f"\n\n`{sel_weight_method.value}` has no per-candidate score to "
                    "act on in this source, so the weights shown are **uniform**.")
    mo.vstack(
        [
            mo.md(_banner),
            mo.md("**Which features were marginalized at all**"),
            gate_table,
            coverage_table,
            mo.md("**Features sorted by expected P(genotoxic)**"),
            feature_probabilities.sort_values("identity_weighted_genotoxicity", ascending=False).head(25),
        ]
    )
    return


@app.cell
def _(feature_probabilities, gcms_candidates, mo):
    mo.md(r"""
    ### See it on one feature: the candidates, the alert, and the limit

    Pick a feature and look at what its score is made of. Each panel is one real
    isomer of the feature's formula, drawn from its SMILES, with the
    **DNA-reactive substructure coloured** where a structural alert fires -- one
    hue per class. The two headline numbers are the whole argument: commit to
    candidate&nbsp;#1 as though the identity were settled, or marginalize over
    the candidate set the way section&nbsp;13 does, and read the TTC limit each
    route lands on off the log axis.
    """)
    sel_feature = mo.ui.dropdown(
        options=sorted(gcms_candidates["feature_id"].astype(str).unique()),
        value=(
            "MTBLS1866-0056"
            if "MTBLS1866-0056" in set(gcms_candidates["feature_id"].astype(str))
            else str(feature_probabilities.sort_values(
                "identity_weighted_genotoxicity", ascending=False)["feature_id"].iloc[0])
        ),
        label="Feature",
    )
    sel_feature
    return (sel_feature,)


@app.cell
def _(candidate_view, gcms_candidates, sel_feature, sl_call_threshold, ttc):
    candidate_view.feature_panel(
        gcms_candidates.assign(feature_id=gcms_candidates["feature_id"].astype(str)),
        sel_feature.value,
        threshold=float(sl_call_threshold.value),
        n_max=12,
        baseline_ttc=ttc.TTC_DEFAULT,
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 14. Does the propagation recover a known answer?

    Section 13 propagates a prediction over a feature's candidate identities.
    On measured data that step cannot be checked, and the reason is not a
    missing file: the true identity of a GC-MS feature is exactly what is
    unknown, so there is nothing to score the propagated number against. The
    published set is different. Every substance in it has one structure and
    one conclusion, so the check can be built there instead -- and it has to
    be built deliberately, because ambiguity is what the published set does
    not have.

    Four properties of the construction, each of which is also its limit:

    - **The candidate list is built, not measured.** Each substance is given
      its nearest structural neighbours from the same set as decoys.
      Look-alikes are the hard case on purpose: an isomer-rich spectrum
      produces a list of near neighbours, and a list of unrelated structures
      would be easy to average over.
    - **The ranking has a quality that is set, not measured.** Quality is
      P(the true identity is ranked first). At 1/k the ranking carries no
      information; at 1 the identity is solved. A real spectral match score
      sits somewhere on that axis, and nothing here says where -- that is the
      measurement this notebook does not have.
    - **No weighting scheme sees the answer.** Weights apply by rank position
      only, so a scheme cannot do well by recognising the true structure.
    - **The probabilities are out-of-fold.** Only substances held out by at
      least one seed are used, scored by models that never trained on them.

    So this bench tests the propagation, not the spectral evidence. What it
    can settle is the question section 13 leaves open: whether carrying the
    candidate distribution forward is better than committing to the top-ranked
    structure. That turns out to depend on which question you are asking, so
    both are shown.

    The bench is not here because the measured data is unavailable -- section
    13 runs on it. It is here because two things the comparison needs exist
    nowhere else in this notebook. **A known answer:** a propagated score can
    only be scored against a settled identity, and a feature's settled
    identity is the unknown. **A ranking to ablate:** the deposited tables
    publish no per-candidate score, so on measured data every candidate
    weighs the same and there is no ordering to remove and re-measure. The
    bench supplies both by construction, and pays for it by simulating the
    ranking quality it cannot observe.

    One thing section 13 does supply, and it changes the bench: the candidate
    counts are now measured rather than guessed. The sizes swept below run on
    a log grid out to the median isomer count of the selected study, which is
    three orders of magnitude past the handful a similarity radius suggests.
    The earlier grid of 2 to 10 answered a question the data does not ask.

    The candidate-count slider in the parameter bar still runs 2 to 10, and
    that is deliberate rather than an oversight: it caps how many isomers each
    feature actually carries downstream, where enumerating a thousand
    structures per peak would cost minutes per interaction. The log grid
    replaces the *sweep* below, not the cap above — section 13 explains what a
    capped feature does with the size of its whole isomer space. The call
    threshold used here is the one section 16 derives from the retained scores;
    its slider lives there because that is where the number comes from.
    """)
    return


@app.cell
def _(mo):
    sl_n_candidates = mo.ui.slider(
        2, 10, 1, value=5,
        label="Candidates per substance (k)", show_value=True,
    )
    sl_rank_quality = mo.ui.slider(
        0.05, 1.0, 0.05, value=0.6,
        label="Ranking quality tested against the null: P(true identity ranked first)",
        show_value=True,
    )
    mo.vstack([sl_n_candidates, sl_rank_quality])
    return sl_n_candidates, sl_rank_quality


@app.cell
def _(
    X,
    ambiguity,
    np,
    predictions,
    sel_model,
    sl_n_candidates,
    sl_rank_quality,
    y,
):
    _p_oof, _n_folds = ambiguity.out_of_fold(predictions, sel_model.value, len(y))
    amb_pool = np.where(_n_folds > 0)[0]
    amb_k = int(sl_n_candidates.value)
    amb_similarity = ambiguity.tanimoto(X)
    amb_p_out_of_fold = _p_oof
    _candidates = ambiguity.candidate_sets(amb_similarity, amb_pool, k=amb_k)
    amb_p_candidates = _p_oof[_candidates]
    amb_p_known = _p_oof[amb_pool]
    amb_sweep = ambiguity.quality_sweep(
        amb_p_candidates, amb_p_known, y[amb_pool], (0.2, 0.4, 0.6, 0.8, 1.0)
    )
    amb_perm, amb_null = ambiguity.permutation_test(
        amb_p_candidates, amb_p_known,
        quality=float(sl_rank_quality.value), n_permutations=500,
    )
    return (
        amb_k,
        amb_null,
        amb_p_candidates,
        amb_p_known,
        amb_p_out_of_fold,
        amb_perm,
        amb_pool,
        amb_similarity,
        amb_sweep,
    )


@app.cell
def _(amb_k, amb_null, amb_perm, amb_pool, amb_sweep, figures):
    fig_ambiguity = figures.figure_ambiguity(
        amb_sweep, amb_null, amb_perm, len(amb_pool), amb_k
    )
    fig_ambiguity
    return


@app.cell
def _(amb_k, amb_perm, amb_sweep, ambiguity, mo):
    _mae = amb_sweep.pivot_table(index=["quality", "seed"], columns="scheme",
                                 values="mae_vs_known")
    _pr = amb_sweep.pivot_table(index=["quality", "seed"], columns="scheme",
                                values="pr_auc_vs_label")
    _grid = sorted(set(_mae.index.get_level_values("quality")))
    _chance = min(_grid, key=lambda q: abs(q - 1.0 / amb_k))
    _at_chance = _mae.loc[_chance]
    mo.vstack([
        mo.md(
            "**Question one: what probability would I report if the identity were "
            "settled?** Committing to the top-ranked candidate wins this one as soon "
            "as the ranking beats chance -- it is exact when the ranking is perfect, "
            f"and it beats equal weight in {int((_mae['top1'] < _mae['uniform']).sum())} of "
            f"{len(_mae)} quality-by-seed runs. The advantage is not free, though: it "
            f"is bought from the ranking. At the point on the grid closest to a ranking "
            f"made of noise (quality {_chance:.2f}, against 1/k = {1.0 / amb_k:.2f}) it "
            f"comes down to "
            f"{int((_at_chance['top1'] < _at_chance['uniform']).sum())} of "
            f"{len(_at_chance)} seeds, which is a coin toss. Equal weight is flat "
            "across the whole axis by construction: it is the one scheme whose error "
            "does not depend on a ranking quality nobody has measured."
        ),
        mo.md(
            "**Question two: which features should someone look at first?** The "
            "opposite answer, and this is the question the notebook actually asks. "
            f"Rank-decay weighting beats committing to one candidate in "
            f"{int((_pr['rank_decay'] > _pr['top1']).sum())} of {len(_pr)} runs and "
            f"equal weight beats it in {int((_pr['uniform'] > _pr['top1']).sum())}, at "
            "every ranking quality including a perfect one. Collapsing a feature onto "
            "its top candidate throws away information that helps the ordering, even "
            "when the top candidate is right: the spread over look-alikes is itself a "
            "signal about the neighbourhood."
        ),
        mo.md(
            f"**Does the ranking carry information at all?** At quality "
            f"{amb_perm['statistic']:.3f} Spearman against the settled-identity score, "
            f"against a null median of {amb_perm['null_median']:.3f} built by shuffling "
            f"the true identity's rank uniformly over the {amb_k} slots "
            f"({amb_perm['n_permutations']} permutations, p = {amb_perm['p_value']:.3f}). "
            "So a ranking of the declared quality does reach the propagated number. "
            "What the test cannot do is tell you the declared quality is the real one."
        ),
        mo.md("**Median across the five simulated rankings per quality**"),
        ambiguity.sweep_median(amb_sweep, "mae_vs_known"),
        ambiguity.sweep_median(amb_sweep, "pr_auc_vs_label"),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### How much ambiguity can it absorb, and does it change the conclusion?

    Two follow-up questions the sweep above cannot answer. The first is how the
    propagation behaves as the candidate list grows: the sweep varies how good
    the ranking is, at a fixed five candidates. The second is whether the score
    differences reach a decision at all -- a mean absolute error of 0.02 is
    either negligible or decisive depending on where it sits relative to the
    calling threshold, so the same comparison is repeated on the calls
    themselves, at the threshold section 16 derives from the held-out scores
    of section 10.

    The right-hand panel counts **missed calls**: the settled identity would
    have called the substance and the scheme does not. That is the error the
    TTC asymmetry exists to avoid, and it is reported as a share of the calls a
    settled identity makes, not of all substances, because the latter is
    dominated by the substances nobody would call either way.

    Three things to read off the panels. `uniform` degrades without bound,
    which is the expected cost of spreading identity mass over more
    look-alikes -- and the log grid shows that the degradation does not level
    off anywhere before the sizes the data actually has. `rank_decay`
    saturates instead: past a few dozen candidates the far tail of the ranking
    carries almost no weight, so adding more of it changes little. `top1` is
    exactly flat, and that is a property of the construction rather than a
    result: committing to the top slot returns the true probability when the
    ranking puts it first and the nearest neighbour's otherwise, whatever the
    candidate-set size. Its error therefore cannot register that a feature
    became more ambiguous. Flatness here is blindness, not robustness.

    What this does **not** measure: the *composition* of a real candidate
    list, which is set by the library and the mass accuracy rather than by a
    similarity radius -- its *size* is no longer a guess, since the grid now
    runs to the count measured upstream. And the decision consequence stops at
    the call, because
    in this pipeline the limit that gets applied is a deterministic function of
    it -- the genotoxic TTC replaces the baseline exactly when the call flips
    and the baseline was the looser of the two. How far the limit then tightens
    needs a Cramer class per substance, which the EFSA rows do not carry and
    this notebook does not assign.
    """)
    return


@app.cell
def _(
    amb_p_candidates,
    amb_p_known,
    amb_p_out_of_fold,
    amb_pool,
    amb_similarity,
    ambiguity,
    gcms_info,
    sl_call_threshold,
    sl_rank_quality,
    y,
):
    amb_anchor = gcms_info.get("median_isomer_space") or 0
    amb_ks = ambiguity.log_sizes(len(amb_pool), anchor=amb_anchor)
    amb_size = ambiguity.size_sweep(
        amb_similarity, amb_pool, amb_p_out_of_fold, y[amb_pool],
        threshold=float(sl_call_threshold.value),
        quality=float(sl_rank_quality.value), sizes=amb_ks,
    )
    amb_decision = ambiguity.decision_change(
        amb_p_candidates, amb_p_known,
        threshold=float(sl_call_threshold.value),
        quality=float(sl_rank_quality.value),
    )
    return amb_anchor, amb_decision, amb_size


@app.cell
def _(amb_anchor, amb_size, figures, sl_call_threshold, sl_rank_quality):
    fig_identity_size = figures.figure_identity_size(
        amb_size, quality=float(sl_rank_quality.value),
        threshold=float(sl_call_threshold.value), anchor=amb_anchor,
    )
    fig_identity_size
    return


@app.cell
def _(amb_decision, amb_size, mo):
    _d = amb_decision.groupby("scheme")[["changed", "missed", "added"]].median()
    _calls = float(amb_decision["reference_calls"].iloc[0])
    _n = int(amb_decision["n"].iloc[0])
    _missed_share = (100.0 * amb_decision.groupby("scheme")["missed"].median() * _n / _calls)
    _s = amb_size.assign(
        missed_pct=100.0 * amb_size["missed"] * amb_size["n"] / amb_size["reference_calls"])
    _by_k = _s.groupby(["k", "scheme"])["missed_pct"].median().unstack()
    _kmin, _kmax = int(_by_k.index.min()), int(_by_k.index.max())
    _crossed = [s for s in _by_k.columns
                if s != "top1"
                and _by_k.loc[_kmin, s] < _by_k.loc[_kmin, "top1"]
                and _by_k.loc[_kmax, s] > _by_k.loc[_kmax, "top1"]]
    mo.md(
        f"""
        At the notebook's settings, a settled identity calls
        **{_calls:.0f} of {_n} substances**. Committing to the top candidate
        changes the call on {100 * _d.loc["top1", "changed"]:.1f} % of
        substances and misses **{_missed_share["top1"]:.0f} % of those calls**;
        carrying the distribution with rank decay changes
        {100 * _d.loc["rank_decay", "changed"]:.1f} % and misses
        {_missed_share["rank_decay"]:.0f} %. So the headline is not that the
        scores differ by a little, it is that roughly a fifth of the calls a
        resolved identity would make are lost either way, and which scheme
        loses fewer of them depends on how ambiguous the features are.

        Across candidate-set sizes, `top1` misses
        {_by_k.loc[_kmin, "top1"]:.0f} % at k = {_kmin} and the same
        {_by_k.loc[_kmax, "top1"]:.0f} % at k = {_kmax}, by construction.
        `rank_decay` starts at {_by_k.loc[_kmin, "rank_decay"]:.0f} % and
        reaches {_by_k.loc[_kmax, "rank_decay"]:.0f} %; `uniform` starts at
        {_by_k.loc[_kmin, "uniform"]:.0f} % and reaches
        {_by_k.loc[_kmax, "uniform"]:.0f} %.
        """
        + (f"""
        {" and ".join(f"`{s}`" for s in _crossed)} crosses `top1` between
        those sizes: propagation misses fewer calls while the candidate list is
        short and more once it is long. That is the useful form of the answer
        -- not whether to carry the distribution, but up to how much ambiguity
        it pays to.
        """ if _crossed else """
        No scheme crosses `top1` over this range of sizes, so at these settings
        the ordering is stable in the candidate-set size and the choice does
        not hinge on it.
        """)
        + (f"""
        One number deserves to be read on its own. At the largest size swept,
        k = {_kmax} -- the median isomer count measured in section 13, not a
        round number chosen here -- `uniform` misses
        **{_by_k.loc[_kmax, "uniform"]:.0f} % of the calls** a settled
        identity makes. That is the cost of the only weighting the deposited
        data supports: with no per-candidate score there is no ranking to
        decay, identity mass spreads evenly across every isomer of the
        formula, and the propagated score lands below the threshold for
        essentially every substance that should have been called. Section 13's
        uniform weights are honest about the evidence and blunt about the
        decision, and this is the measurement of how blunt. It is also the
        strongest argument in the notebook for spending effort on the ranking
        rather than on the toxicity model.
        """ if _by_k.loc[_kmax, "uniform"] >= 90 else "")
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## 15. Does the identity evidence actually earn its place?

    Everything above assumes that weighting candidates by spectral evidence is
    better than not doing it. That is an assumption, it is testable, and **on a
    public source it cannot be tested** — which is the result of this section
    rather than an excuse for not having one. What follows measures the two
    distinct reasons why, because they are not the same failure and only one of
    them more data would fix.

    The reference, where it exists, is the subset of features whose identity the
    depositors **confirmed** (the MSI level of the assignment). For those, the
    probability computed from the confirmed structure is what a perfect
    identification would have produced, and three estimators that do not get to
    see the answer are scored against it:

    | estimator | what it knows | on a public source |
    |---|---|---|
    | `evidence_weighted` | the spectral evidence — what a local export supports | **not computable**: no per-candidate score exists to weight by |
    | `uniform` | structure alone; identity evidence thrown away | runnable, and the only one that is |
    | `top1_only` | the best-ranked candidate, identity assumed certain | runnable but **circular**: the deposited assignment *is* the top-ranked candidate |

    So the comparison this section was written to make — does the evidence
    weighting earn its place — is unavailable here, and the honest report is
    which of its three legs broke and how. What survives is narrower and still
    worth having: `uniform` against the confirmed structure says whether
    marginalizing over isomers recovers the right *magnitude* and the right
    *ordering*, and those two can come apart.

    **The conditional stands for a local export: if `evidence_weighted` does
    not beat `uniform` there, the weighting is decoration and should be removed
    from the notebook.** It is stated here and not quietly dropped because an
    untestable assumption is not a validated one.

    ### What counts as the reference, and why it is not MSI-1 alone

    Stated before the result rather than after it, because the choice is
    arguable and the reader should see it made.

    MSI level 1 means the identity was confirmed against an authentic standard
    injected on the same instrument. It is the only tier established fully
    independently of the candidate ranking, so it is the ideal reference. The
    problem is how few features reach it, and why.

    Level 1 exists only for compounds someone chose to inject a standard for,
    and in an untargeted run that is normally the QC set: deuterated internal
    standards, added deliberately at known concentration precisely because they
    are clean and well-behaved. They are not migrants. A reference built from
    them would measure how well the estimators recover the chemistry of spiked
    standards, which is not the chemistry they are being asked to identify. The
    small sample size is the smaller of the two problems.

    The reference used here is therefore **MSI levels 1 and 2**. What that buys
    is sample size, and the cell below reports how much of it there is. What it
    costs is stated plainly: level 2 rests on spectral evidence, and the
    candidate ranking uses spectral evidence too, so reference and estimator are
    **not fully independent**. Any margin `evidence_weighted` shows over
    `uniform` has to be read with that in mind, and the overlap check below is
    the guard that stops the weaker dependence from passing as independence.

    **This is not a limitation that data access removes.** A real export would
    make this comparison runnable rather than illustrative, but it would not add
    one authentic standard: level 1 is bounded by what was injected into the
    instrument, not by what is in the file. Genuine ground truth needs a mixture
    of known standards run through the same method. That is laboratory work, and
    it is the honest answer to anyone who asks why this notebook does not simply
    report identification accuracy.

    One trap, checked first: if the source assigns the confirmed identity by
    taking the top-scoring candidate, then `top1_only` is being compared
    against a copy of itself and wins by construction. The overlap check below
    is what says whether the ablation means anything. Near 100 % overlap
    invalidates it, and the comparison then has to be restricted to identities
    confirmed against an authentic standard (MSI level 1), which is independent
    of the ranking.

    ### What the public deposits do to this comparison

    On the public source the guard fires, and it is worth seeing why rather
    than being told the section is unavailable.

    The confirmed identity **is** the top-ranked candidate, in every confirmed
    feature, without exception. Not because a ranking put it there: a MAF
    publishes no candidate scores, so the candidate set is built by enumerating
    a formula's isomers with the depositor's own assignment placed first, and
    `rank` is that enumeration order. The overlap is therefore 100 % by
    construction, `top1_only` scores a perfect zero against itself, and that
    zero is an artefact of how the list was built and means nothing.

    `evidence_weighted` has the opposite problem: with no per-candidate score
    there is nothing to weight by, so the estimator this notebook is built
    around cannot be evaluated here at all.

    What survives is the one comparison that does not need a ranking:
    `uniform` against the confirmed structure. That is the cost of
    marginalizing over a formula's isomers when someone else already knew the
    answer, and it is the honest version of what this section set out to
    measure. The absolute error is small, because most isomers of these
    formulas are ordinary and the model says so about all of them. The rank
    correlation is the number to look at, and it is roughly a coin's worth of
    ordering: the features a risk assessor would look at first are not the same
    features in the two views.

    **This is the limitation that data access does not remove, stated again
    with a measurement behind it.** A private export with candidate scores
    would fill `evidence_weighted` in. It would still not give an independent
    reference, because level 1 is bounded by which authentic standards were
    injected, not by which file is read.
    """)
    return


@app.cell
def _(
    fixture,
    gcms_candidates,
    gcms_features,
    gcms_info,
    integration,
    mo,
    sl_temperature,
):
    # integration.REFERENCE_MSI, not CONFIRMED_MSI: the gate needs an
    # authentic standard before it will collapse a feature's uncertainty,
    # and level 1 alone leaves too few features to score against. See that
    # constant for the trade and the residual dependence it admits.
    reference_check = integration.ablation_reference_check(
        gcms_candidates, gcms_features, msi_levels=integration.REFERENCE_MSI
    )
    ablation_table = integration.ablation_identity_evidence(
        gcms_candidates, gcms_features, temperature=sl_temperature.value,
        msi_levels=integration.REFERENCE_MSI,
    )
    _overlap = reference_check.set_index("quantity")["value"].get("overlap (%)")
    _warn = (
        mo.md("> **The reference is not independent.** `inchikey_id` coincides with the "
              f"top-ranked candidate in {_overlap} % of confirmed features, so `top1_only` "
              "is scoring against itself. Read nothing into the table below until the "
              "comparison is restricted to MSI level 1 confirmations.")
        if _overlap is not None and _overlap >= 90
        else mo.md("The confirmed identity disagrees with the top-ranked candidate often "
                   f"enough ({100 - _overlap:.0f} % of the time) for the comparison to carry "
                   "information." if _overlap is not None else "")
    )
    reference_by_level = integration.reference_levels(gcms_features)
    ablation_confirmed = integration.ablation_identity_evidence(
        gcms_candidates, gcms_features, temperature=sl_temperature.value,
        msi_levels=integration.CONFIRMED_MSI,
    )
    _n_ref = int(ablation_table["n_features"].max()) if len(ablation_table) else 0
    _n_l1 = int(ablation_confirmed["n_features"].max()) if len(ablation_confirmed) else 0
    _coincide = _n_ref == _n_l1
    _levels = mo.md(
        f"**What the reference is actually made of.** Level 1 alone gives "
        f"{_n_l1} features; levels 1 and 2 give {_n_ref}. "
        + ("They coincide: every confirmed identity in this export sits at level 1, "
           "so the table below *is* the level-1 comparison and the shared-evidence "
           "caveat above has nothing to attach to. Admitting level 2 buys sample "
           "size only where level 2 confirmations exist, and here none do."
           if _coincide else
           f"Admitting level 2 adds {_n_ref - _n_l1} features, and those are exactly "
           "the rows whose reference shares spectral evidence with the estimator. "
           "Both tables are shown, so any margin can be read against the "
           "independent subset as well as the larger one.")
    )
    # Two ways this ablation can have nothing to say, and both are properties
    # of the source rather than bugs. They are reported instead of the table
    # because an empty table invites the reader to squint at it.
    _no_ref = _n_ref == 0
    _no_evidence = not gcms_info.get("has_evidence", True)
    _diag = mo.md(
        ("> **No reference in this study.** Not one feature here carries a parsed "
         "MSI level, so there is no confirmed identity to score the estimators "
         "against and the table below is empty. This is the asymmetry section 13 "
         "measured: the study with the isomer space is not the study that declares "
         "identity confidence. Switch the selector to a study that fills the "
         "column and the comparison runs."
         if _no_ref else
         "> **`evidence_weighted` cannot be computed on this source.** A public "
         "deposit publishes one assigned identity per feature and no score per "
         "candidate, so the estimator this notebook actually uses has no input "
         "and its row is blank rather than bad. What remains comparable is "
         "`uniform` against the confirmed structure, and that margin is the cost "
         "of marginalizing instead of identifying."
         if _no_evidence else "")
    )
    _out = mo.vstack([mo.md(fixture.banner(gcms_info)),
                      reference_check, reference_by_level, _levels, _diag, _warn]
                     + ([] if _no_ref else
                        [mo.md("**Error against the confirmed "
                               "structure** (lower MAE/RMSE is better, higher Spearman "
                               "is better)"),
                         ablation_table])
                     + ([] if _coincide or _no_ref else
                        [mo.md("**Restricted to level-1 confirmations** — independent of "
                               "the ranking, and smaller"), ablation_confirmed]))
    _out
    return


@app.cell
def _(gcms_source, mo, public_features, sel_study):
    _probe = (public_features.acquisition_columns(sel_study.value)
              if gcms_source != "export" else None)
    mo.vstack([
        mo.md("""
        ### The leakage control this notebook does not have

        The scaffold split stops congeners leaking between train and test on the
        **chemistry** side. It does nothing about the **acquisition** side:
        features measured in the same sequence share column state, tuning drift
        and carryover contamination, so treating them as independent is
        optimistic in a way no chemical split can fix.

        A split by acquisition batch would control it, and it is **not
        implemented** because the data does not carry what it needs. The probe
        below searches the deposited assignment file for a run date, a batch
        label or an injection order. The chromatographic retention time is not
        counted: it says where a peak sits inside a run, not which run it was,
        and using it here would be a different control wearing this one's name.
        """),
        _probe if _probe is not None and len(_probe)
        else mo.md("_Select the public source to probe a deposited file._"),
        mo.md("""
        The deposit publishes an assignment per feature and an abundance per
        sample, and nothing about when any of it was measured. That is not a
        gap in this study in particular: the MAF format has no field for it,
        and the acquisition metadata, where it exists at all, lives in the
        study's ISA files rather than the assignment table. So the control
        stays unimplemented and is declared rather than approximated.
        """),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ### Why the coverage number is the explainable part

    The output of this notebook is not one number per feature. It is a number
    **and the decomposition that produced it**: which candidate contributed
    which share of the risk, and how much of the identity was never evaluated
    at all.

    That distinction matters for how the result can be used. Post-hoc
    attribution methods explain a prediction after the fact, by perturbing
    inputs or fitting a surrogate, and the explanation is a second model that
    can disagree with the first. Here the weights are not a reconstruction —
    they *are* the computation. `identity_weighted_genotoxicity` cannot be produced without them.

    This follows the argument for self-explaining models made by Teufel,
    Torresi, Reiser and Friederich in MEGAN, where explanations are generated
    by the model along with the prediction rather than recovered afterwards by
    a separate procedure. Phase 3 of this project is MEGAN for exactly that
    reason; the marginalization here is the same idea applied one level up, at
    the identity rather than the atom.

    The practical consequence for a risk assessor is the part worth stating
    plainly: a feature with `identity_weighted_genotoxicity = 0.7` and `covered_weight = 0.35` is
    not a 70 % risk. It is a **claim about a third of an unknown compound**, and
    the honest action is to go back and identify it rather than to report the
    number.

    > Teufel, J., Torresi, L., Reiser, P., Friederich, P. (2023). *MEGAN:
    > Multi-explanation Graph Attention Network.* In Longo, L. (ed.)
    > Explainable Artificial Intelligence, xAI 2023. Communications in Computer
    > and Information Science, vol. 1902. Springer, Cham.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 16. What the call is for: prioritising against the TTC limit

    A probability is not a decision, and neither is what follows. A decision in
    food-contact safety compares an estimated exposure against a limit, and this
    notebook has no exposure estimate: no concentrations, no consumption figures,
    no migration model. What the threshold of toxicological concern supplies is
    the limit a migrant would be held to, which turns a genotoxicity call into
    something statable in micrograms -- enough to rank features by how much
    their limit would tighten, and not enough to clear or condemn any of them.

    Without a genotoxicity flag, a substance is held to its Cramer-class limit:
    **1800, 540 or 90 ug/day**. If it is treated as potentially genotoxic, the
    applicable value is **0.15 ug/day**. Between the Cramer III limit and the
    genotoxic one there is a factor of **600**.

    That factor is the reason this notebook exists, and also the reason to be
    careful with it. A false positive costs a 600-fold tightening on a compound
    that did not need it; a false negative leaves a genotoxic migrant held to a
    limit six hundred times too loose. The threshold below is therefore not a
    tuning knob — it is where that trade-off is set, and it belongs to the
    assessor, not to the model.

    Two limits are computed per feature: the one the pipeline already applies,
    and the one after the model's call. Among candidates carrying real weight,
    **the most restrictive limit wins** — a limit is a bound, and averaging
    bounds across candidate identities would produce a number that protects
    against none of them.

    ### This is hazard prioritization, not risk assessment

    Stated flatly, because the distinction is the one most easily lost in a
    table of micrograms. A TTC value is a limit; a risk assessment compares an
    **exposure** against a limit. This notebook has no exposure: it never sees
    a measured concentration, a migration test, or a consumption figure, so it
    cannot say that anything exceeds anything. What it produces is an ordering
    of which features would matter most if concentrations were known.

    This is the standard regulatory distinction rather than a nicety:
    intrinsic hazard weighted by a concentration in an extract is a hazard
    ranking, never a risk assessment. The per-feature probability computed above
    has that same shape — a probability weighted by an identity assumption — so
    treating either as a measurement of risk would be a category error, and
    neither this section nor section 17 should be quoted as one.
    """)
    return


@app.cell
def _(mo, models, np, p_calibration, y, y_calibration):
    call_threshold_matched = models.prevalence_threshold(
        p_calibration, int(y_calibration.sum())
    )
    call_threshold_curve = models.call_threshold_curve(p_calibration, y_calibration)
    sl_call_threshold = mo.ui.slider(
        0.01, 0.90, 0.01,
        value=round(float(np.clip(call_threshold_matched, 0.01, 0.90)), 2),
        label="Threshold for calling a feature genotoxic", show_value=True,
    )
    mo.vstack([
        sl_call_threshold,
        mo.md(
            f"**Why the default is {call_threshold_matched:.3f} and not 0.5.** Positives "
            f"are {100 * y.mean():.0f} % of the labelled set, and out of sample the "
            f"scores are compressed hard toward zero: the median held-out score is "
            f"{np.median(p_calibration):.4f} and only "
            f"{100 * (p_calibration > 0.5).mean():.1f} % of held-out substances exceed "
            f"0.5, against a positive rate of {100 * y_calibration.mean():.1f} % in the "
            "pooled held-out folds — pooled over every seed those folds cover almost "
            "the whole set, so that rate tracks the overall one rather than telling you "
            "anything extra. At 0.5 this layer would never fire — not "
            "because nothing is genotoxic, but because the threshold was imported from a "
            "balanced problem this is not. **The default is not a number typed in here:** "
            "it is `models.prevalence_threshold` on the held-out scores, the value whose "
            "count of positive calls matches the observed positive rate. That is a "
            "starting point and not a recommendation — section 10 is where it should "
            "actually be set, the asymmetry of the 600-fold consequence argues for "
            "putting it lower still, and the curve below is what the choice costs in "
            "either direction."
        ),
    ])
    return call_threshold_curve, call_threshold_matched, sl_call_threshold


@app.cell
def _(call_threshold_curve, call_threshold_matched, figures, p_calibration, y_calibration):
    fig_call_threshold = figures.figure_call_threshold(
        call_threshold_curve, call_threshold_matched,
        float(y_calibration.mean()), len(p_calibration),
    )
    fig_call_threshold
    return


@app.cell
def _(
    feature_probabilities,
    fixture,
    gcms_candidates,
    gcms_features,
    gcms_info,
    mo,
    sl_call_threshold,
    ttc,
):
    ttc_baseline = ttc.baseline_ttc(gcms_candidates, gcms_candidates["weight"])
    ttc_priority = ttc.decide(feature_probabilities, ttc_baseline,
                              threshold=sl_call_threshold.value)
    ttc_summary = ttc.crossing_summary(ttc_priority)
    _exposure = ttc.exposure_columns(gcms_candidates, gcms_features)
    mo.vstack([
        mo.md(fixture.banner(gcms_info)),
        ttc_summary,
        mo.md(
            "**Features whose limit the model tightens**"
            if bool(ttc_priority["crosses"].any()) else
            "**No feature crosses.** That is an outcome, not a failure: the highest "
            f"expected probability across all features is "
            f"{ttc_priority['identity_weighted_genotoxicity'].max():.3f}, below the threshold. A pipeline that "
            "produced alarms on a set of ordinary plasticizers and antioxidants would be "
            "the thing to distrust. The limits each feature is held to are shown anyway, "
            "so the mechanism is visible."
        ),
        (ttc_priority[ttc_priority["crosses"]] if bool(ttc_priority["crosses"].any())
         else ttc_priority)
        .sort_values("identity_weighted_genotoxicity", ascending=False)
        [["feature_id", "identity_weighted_genotoxicity", "ttc_baseline", "ttc_model", "tightening_factor"]]
        .head(20),
        mo.md(
            "**The half that is missing.** An exceedance needs a measured amount, and the "
            "probe above found no concentration column: "
            f"`{list(_exposure)}`. Without it the notebook can say a limit should be "
            "600 times stricter, but not whether anything actually crosses it. That is a "
            "gap in the data, and it is stated rather than filled with an assumption."
            if not len(_exposure) else
            f"Candidate exposure columns found: `{list(_exposure)}`. An exceedance "
            "calculation becomes possible; check units before using them."
        ),
    ])
    return (ttc_priority,)


@app.cell
def _(figures, gcms_candidates, gcms_info, sl_ad_threshold, ttc_priority):
    fig_decision_chain = figures.figure_decision_chain(
        gcms_candidates, ttc_priority,
        ad_threshold=float(sl_ad_threshold.value),
        anchor=gcms_info.get("median_isomer_space") or 0,
    )
    fig_decision_chain
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Reading the chain end to end

    The left panel is the applicability domain doing its job and the cost of
    it in the same picture. A candidate whose nearest training neighbour is
    below the Tanimoto cutoff is dropped from the weighted sum rather than
    given a prediction, so **covered weight and in-domain weight are the same
    number** -- the two are not independent checks, and the panel is labelled
    to say so. Around a third of the identity weight of a typical feature
    survives that filter.

    The middle panel is the answer to the question the isomer counts raised.
    Rank 1 of each candidate list is the identity the depositors assigned, and
    those assignments are mostly inside the domain -- these are the ordinary
    plasticizers and antioxidants the model was trained on. Every isomer added
    after it is a structure nobody vouched for, and the in-domain share falls
    steeply for the first few and then **flattens**: the isomers of a formula
    are not systematically more exotic the deeper you go into the list, so
    past roughly fifteen candidates the share stops moving. That flattening is
    measured out to the depth that was built, and the study's median isomer
    count is two orders of magnitude beyond it; reading the plateau as holding
    all the way there is an extrapolation, and it is labelled as one on the
    figure rather than drawn as a line.

    Two consequences worth stating plainly. The applicability domain does not
    collapse with depth -- it converges to a roughly constant share, which is
    the better of the two outcomes it could have had. But a constant share is
    still a diluted one: the deposited assignment's coverage is halved by the
    first few isomers, and since no per-candidate score exists to concentrate
    the weight back onto the plausible ones, that dilution is not recoverable
    within this notebook. Section 14 measures what it costs at the decision.

    The right panel is the end of the chain, and it is quiet on purpose. The
    limit either stays at the Cramer-class value or drops to the genotoxic
    one; there is nothing in between, because the genotoxic TTC replaces the
    baseline outright rather than scaling it. A handful of features cross, and
    the features with nothing evaluable are counted separately rather than
    folded into the unchanged ones -- an absent decision is not a negative
    one, and section 18 carries that distinction into the handover table.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 17. A second opinion: the structural alerts

    Everything so far is one method. A structural rulebase is another, built
    the opposite way: alerts read off the structure, no training set, no
    statistics. Where the two agree, a call rests on two independent grounds.
    Where they disagree, the disagreement itself is the finding.

    **What the rulebase here is, and what it is not.** No MetaboLights deposit
    publishes a genotoxicity alert, and this project has no licensed
    implementation of the Benigni-Bossa rulebase, so the alerts below are
    computed from the candidate structures against a short list of
    DNA-reactive classes written out in `chemistry.DNA_REACTIVE_ALERTS`:
    aromatic nitro, primary aromatic amine, epoxide, aziridine, N-nitroso,
    hydrazine, organic azide, aromatic azo, alkyl halide and Michael acceptor.
    Three things follow and none of them are hedging:

    - Agreement and kappa against this list are **not comparable** to an
      evaluation against Benigni-Bossa, and quoting them as one would be
      wrong. That rulebase has dozens of alerts, exclusion rules and potency
      modulations; this has ten SMARTS.
    - The list is under-inclusive on purpose. A structure with no hit is not
      "no alert" in a regulatory sense, it is "none of these ten".
    - A hit is a structural flag, not a conclusion. Plenty of substances
      carrying these groups are not genotoxic in vivo, which is exactly why
      the table below counts disagreements instead of scoring accuracy.

    Neither method is ground truth, so the table is not an accuracy measure.
    It is a count of how often two methods reach the same conclusion about the
    same structure — and the interesting cell is always the one where they do
    not.

    The comparison is restricted to candidates with **no experimental value in
    the labelled set**, since that is precisely where either method is being
    relied on rather than checked against a known answer. On the public source
    that restriction removes almost nothing, and the reason is worth reading
    off the row count: the isomer candidates a formula generates are, with a
    handful of exceptions, substances nobody has tested.

    **Read kappa, not the agreement percentage.** Both methods say "not
    genotoxic" about the overwhelming majority of these structures, so a high
    agreement figure is mostly the two of them agreeing about the easy
    negatives. Cohen's kappa strips that out, and it is the number that says
    whether the second opinion is independent evidence or a coin that happens
    to land the same way.
    """)
    return


@app.cell
def _(
    fixture,
    gcms_candidates,
    gcms_info,
    integration,
    mo,
    sl_call_threshold,
):
    alert_table = integration.alert_agreement(
        gcms_candidates, gcms_candidates["p_genotox"],
        threshold=sl_call_threshold.value, restrict_to_gap=True,
    )
    alert_cases = integration.alert_disagreements(
        gcms_candidates, gcms_candidates["p_genotox"], threshold=sl_call_threshold.value,
    )
    mo.vstack([
        mo.md(fixture.banner(gcms_info)),
        alert_table,
        mo.md("**Where the two methods part company**"),
        alert_cases,
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 18. What to hand over: one row per feature

    Everything above argues about method. This is the artefact an analyst would
    actually receive: one row per feature, with the quantity, the two things
    that qualify it, and the step that would reduce whichever uncertainty
    dominates the row.

    **It is a prioritisation, not a verdict.** No row says a feature is safe or
    unsafe. Saying that needs a measured concentration to compare against the
    TTC limit, and this notebook has none — it computes limits, not exceedances.
    What the table ranks is where unresolved identity changes the toxicological
    reading enough to be worth an analyst's time.

    Three columns qualify the score and none of them are decoration:

    - `covered_weight` — how much of the feature's identity mass was evaluable
      at all. A feature at 0.4 has 60 % of its identity unscored, and its number
      is an average over the part that happened to have a structure.
    - `weight_in_domain` — how much of that mass sits near the training
      chemistry. On this source the two columns carry the **same number**, and
      that is not redundancy to be cleaned up: a candidate outside the domain
      is dropped from the weighted sum rather than scored, so the domain filter
      is what defines coverage. They would diverge on a source where a
      structure can be in-domain and still unscored — a missing SMILES, a
      fingerprint that fails — and both are kept so that divergence would be
      visible rather than invisible.
    - The renormalised score is deliberately *not* what the table shows:
      dropping the out-of-domain candidates and rescaling the rest can turn a
      feature whose dominant identity the model cannot judge into a
      confident-looking number. The confidence would come from the deletion,
      not from the evidence.
    - `alert` — whether the DNA-reactive alert list of section 17 fires, as a
      weighted share of the candidate set. `+/-` means the candidates disagree
      with each other, which is itself a reason to resolve the identity, and
      it is the modal outcome for a feature with real isomers: the same
      formula covers structures that carry a reactive group and structures
      that do not.

    `next_step` is a rule over the columns beside it — stated in the function
    that computes it — and not a recommendation with any standing.
    """)
    return


@app.cell
def _(fixture, gcms_candidates, gcms_info, integration, mo, sl_call_threshold, ttc_priority):
    handover = integration.decision_table(
        gcms_candidates, ttc_priority, threshold=sl_call_threshold.value,
    )
    _show = ["feature_id", "n_candidates", "identity_weighted_genotoxicity",
             "covered_weight", "weight_in_domain", "domain", "alert",
             "ttc_baseline", "ttc_model", "tightening_factor", "next_step"]
    _n = len(handover)
    _counts = handover["next_step"].value_counts()
    _thin = int((handover["covered_weight"] < 0.5).sum())
    _odd = int((handover["domain"] != "in").sum())
    _split = int((handover["alert"] == "+/-").sum())
    mo.vstack([
        mo.md(fixture.banner(gcms_info)),
        handover[_show],
        mo.md(
            f"**{_n} features.** " + "; ".join(
                f"{int(v)} {k}" for k, v in _counts.items()) + ". "
            f"{_thin} carry less than half their identity mass as evaluable "
            f"structures, {_odd} have identity mass outside the training "
            f"chemistry, and in {_split} the candidates disagree with the "
            "rulebase among themselves. Those three counts are the ones worth "
            "acting on: they say where the number on the row is thin, and a "
            "thin number is not a low-hazard finding."
        ),
    ])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 19. What this notebook does not do

    Stated plainly, because a limitation found by a reviewer costs more than
    one declared by the author.

    - **The integration has never run on a spectrum.** It runs on deposited
      peak assignments, which are somebody else's identification already made:
      this notebook reads formulae, structures and MSI levels out of a public
      file and never sees a mass spectrum, a chromatogram or a spectral match
      score. Two consequences follow, and they are different in kind. The
      identity uncertainty it propagates is the isomer space of a formula,
      which is real and measured here; the *ranking* inside that space is not,
      because no public deposit publishes a per-candidate score. So
      `evidence_weighted` and the temperature control are implemented against a
      schema no public source fills, and section 15 measures what that costs
      instead of assuming it is small.
    - **The labels arrive as a file, not as a query.** The compound list and
      its genotoxicity conclusions come from a spreadsheet already carrying
      PubChem identifiers for most rows. EFSA's own service is never called,
      so the first step of the provenance chain cannot be re-run or audited here.
    - **1383 labelled rows -- 979 distinct substances -- are lost to structure
      resolution.** They carry no CAS at all, only a name, and the name lookup
      failed. Recovering them needs a source this notebook does not have.
      Section 5 measures what that loss does to the dataset rather than
      assuming it is harmless: no detectable difference in the label rate, a
      complete difference in identifier coverage, so the exclusion is a scope
      limit on what the model applies to, not a bias in its prevalence.
    - **No acquisition-batch split.** The scaffold split controls chemical
      leakage; nothing controls leakage through shared column state, tuning
      drift or carryover. The probe in section 15 looks for the column that
      would make the control possible.
    - **No exposure data.** The TTC layer computes limits, not exceedances.
      Whether anything crosses those limits is unanswerable without measured
      concentrations.
    - **The public retention index cannot rank isomers, and this is measured
      rather than assumed.** It is the obvious second line of evidence and the
      notebook tried it: 1366 structure-index pairs over 632 structures
      harvested from public GC studies, calibrated per study, trained and
      evaluated on scaffold-disjoint sets. Globally it learns something -- MAE
      265 index units against 465 for predicting the mean -- but the
      between-study dispersion of the *same* structure is 27 units, an order of
      magnitude below that error. And globally is the wrong place to look:
      inside a fixed formula, where the ranking would actually be used, the
      median rank correlation with the measured index is **0.00** over the 16
      formulas with at least three measured isomers, and the predictor spreads
      the isomers wider (235 units) than they really are (105). Section 13
      carries the full argument. The branch is kept in the notebook because a
      negative result that cost a module is worth more on screen than deleted.
    - **Identity confidence is not calibrated.** The weights are a
      normalization assumption, not identity probabilities. The route to
      earning the word 'probability' is measurable: on features confirmed
      against an authentic standard, check how often the top-ranked candidate
      is right as a function of the vote margin. That turns identity confidence
      from an assumption into a curve — and it is the single most valuable
      thing the real export would unlock.
    """)
    return


if __name__ == "__main__":
    app.run()