"""Notebook figures. No function saves to disk: they return the figure.

Rules that apply here and should not be broken when editing them:
  - with 5 seeds, each seed's point is drawn plus a median mark,
    not a box: a boxplot with n=5 suggests a distribution that isn't there.
  - no red versus green for binary distinctions.
  - the direction of "better" is shown in the figure, not only in the text.
  - the number of observations appears in the figure.
"""

import matplotlib as mpl
import numpy as np
import pandas as pd

FOCAL = "#1f4e79"
SECONDARY = "#c47e0a"
NEUTRAL = "#8a8a8a"
ALERT = "#b02418"
SIZE = (9, 8, 7)


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": SIZE[0], "axes.titlesize": SIZE[0], "axes.labelsize": SIZE[0],
        "legend.fontsize": SIZE[1], "xtick.labelsize": SIZE[2], "ytick.labelsize": SIZE[2],
        "axes.titlelocation": "left", "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "legend.frameon": False, "lines.markersize": 5,
    })


def _model_scatter(ax, res, metric, order, rng):
    for i, name in enumerate(order):
        v = res.loc[res["model"] == name, metric].dropna().to_numpy()
        if not len(v):
            continue
        x = i + rng.uniform(-0.09, 0.09, len(v))
        ax.plot(x, v, "o", color=FOCAL, alpha=0.65, mec="none")
        ax.hlines(np.median(v), i - 0.22, i + 0.22, color=FOCAL, lw=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.margins(x=0.15, y=0.12)


def figure_metrics(results: pd.DataFrame, order, n_substances: int, seed: int = 0):
    """One point per seed and a median mark, for the three key metrics."""
    import matplotlib.pyplot as plt

    apply_style()
    rng = np.random.default_rng(seed)
    metrics = [("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("mcc_prev", "MCC")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), constrained_layout=True)
    for ax, (col, label) in zip(axes, metrics):
        _model_scatter(ax, results, col, order, rng)
        ax.set_ylabel(label)
    prev = float(results["prevalence"].median())
    axes[0].axhline(prev, ls="--", lw=1, color=ALERT)
    axes[0].annotate(f"prevalence {prev:.2f}\n(baseline for PR-AUC)",
                     xy=(0.02, prev), xycoords=("axes fraction", "data"),
                     xytext=(0, 6), textcoords="offset points",
                     fontsize=SIZE[2], color=ALERT, va="bottom")
    axes[1].axhline(0.5, ls="--", lw=1, color=NEUTRAL)
    axes[1].annotate("chance", xy=(0.02, 0.5), xycoords=("axes fraction", "data"),
                     xytext=(0, 4), textcoords="offset points",
                     fontsize=SIZE[2], color=NEUTRAL, va="bottom")
    n_seeds = int(results.groupby("model").size().max())
    fig.suptitle(
        f"One point per seed ({n_seeds} scaffold-based splits, {n_substances} substances); "
        "the bar is the median. Higher = better",
        fontsize=SIZE[0], x=0.0, ha="left",
    )
    return fig


def figure_calibration(table: pd.DataFrame, ece: float, model_name: str):
    """Reliability diagram. The point size is the bin occupancy."""
    import matplotlib.pyplot as plt

    apply_style()
    d = table[table["n"] > 0]
    fig, ax = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)
    ax.plot([0, 1], [0, 1], ls="--", lw=1, color=NEUTRAL)
    ax.annotate("perfect calibration", xy=(0.62, 0.62), rotation=38,
                fontsize=SIZE[2], color=NEUTRAL, ha="center", va="bottom",
                rotation_mode="anchor")
    point_sizes = 18 + 220 * d["n"] / d["n"].max()
    ax.scatter(d["p_mean"], d["frac_positives"], s=point_sizes, color=FOCAL, alpha=0.8,
               zorder=3, ec="white", lw=0.6)
    ax.plot(d["p_mean"], d["frac_positives"], "-", color=FOCAL, lw=1, zorder=2)
    _largest = d.loc[d["n"].idxmax()]
    ax.annotate(f"n = {int(_largest['n'])}", (_largest["p_mean"], _largest["frac_positives"]),
                xytext=(8, -10), textcoords="offset points", fontsize=SIZE[2], color=FOCAL)
    ax.set_xlabel("predicted probability (bin mean)")
    ax.set_ylabel("actual fraction of positives")
    ax.set_title(f"{model_name}: ECE = {ece:.3f}\npoint area is the number of compounds in the bin")
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.03, 1.03)
    return fig


def figure_domain(similarities, curve: pd.DataFrame, threshold: float):
    """Distance to training and the coverage cost of requiring closeness."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)

    axes[0].hist(similarities, bins=30, color=FOCAL, alpha=0.85)
    axes[0].axvline(threshold, color=ALERT, ls="--", lw=1.2)
    axes[0].annotate(f"threshold {threshold:.2f}", xy=(threshold, 0.96), xycoords=("data", "axes fraction"),
                     xytext=(4, 0), textcoords="offset points",
                     fontsize=SIZE[2], color=ALERT, va="top")
    axes[0].set_xlabel("Tanimoto to nearest neighbor in training set")
    axes[0].set_ylabel(f"test compounds (n = {len(similarities)})")
    axes[0].set_title("Far from training = prediction not reportable")
    axes[0].margins(x=0.03)

    axes[1].plot(curve["threshold"], curve["coverage"], "o-", color=NEUTRAL, lw=1.4)
    axes[1].plot(curve["threshold"], curve["pr_auc"], "s-", color=SECONDARY, lw=1.4)
    _x = curve["threshold"].iloc[-1]
    axes[1].annotate("coverage", (_x, curve["coverage"].iloc[-1]), xytext=(5, 0),
                     textcoords="offset points", fontsize=SIZE[1], color=NEUTRAL, va="center")
    _last = curve["pr_auc"].dropna()
    if len(_last):
        axes[1].annotate("PR-AUC within\ndomain",
                         (curve.loc[_last.index[-1], "threshold"], _last.iloc[-1]),
                         xytext=(5, 0), textcoords="offset points",
                         fontsize=SIZE[1], color=SECONDARY, va="center")
    axes[1].axvline(threshold, color=ALERT, ls="--", lw=1.2)
    axes[1].set_xlabel("required Tanimoto threshold")
    axes[1].set_ylabel("fraction / PR-AUC")
    axes[1].set_title("How much coverage costs to gain reliability")
    axes[1].set_xlim(curve["threshold"].min() - 0.02, curve["threshold"].max() + 0.16)
    axes[1].margins(y=0.12)
    return fig


def figure_provenance(funnel: pd.DataFrame, routes: pd.DataFrame):
    """Where the data came from: the acquisition funnel and the API routes.

    Left: how many rows survive each stage between EFSA's published
    conclusion and a trainable example. Right: which route resolved the
    structure of each row.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4), constrained_layout=True)

    n0 = funnel["n"].iloc[0]
    ypos = np.arange(len(funnel))[::-1]
    axes[0].barh(ypos, funnel["n"], color=FOCAL, alpha=0.88, height=0.62)
    for yp, (_, r) in zip(ypos, funnel.iterrows()):
        axes[0].annotate(f"{r['n']:,}  ({r['retained_pct']:.0f} %)",
                         xy=(r["n"], yp), xytext=(5, 0), textcoords="offset points",
                         va="center", fontsize=SIZE[2], color=FOCAL)
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(funnel["stage"], fontsize=SIZE[2])
    axes[0].set_xlabel("rows / substances")
    axes[0].set_xlim(0, n0 * 1.34)
    axes[0].set_title("From published conclusion to trainable example")
    axes[0].grid(axis="y", visible=False)

    keep = routes[routes["n"] > 0]
    colors = [NEUTRAL if "no structure" in s else FOCAL for s in keep["route"]]
    xpos = np.arange(len(keep))
    axes[1].bar(xpos, keep["n"], color=colors, alpha=0.88, width=0.6)
    for xp, (_, r) in zip(xpos, keep.iterrows()):
        axes[1].annotate(f"{r['n']:,}", xy=(xp, r["n"]), xytext=(0, 3),
                         textcoords="offset points", ha="center",
                         fontsize=SIZE[2], color="#333333")
    axes[1].set_xticks(xpos)
    axes[1].set_xticklabels([s.replace(" (", "\n(").replace(" -> ", "\n-> ")
                             for s in keep["route"]], fontsize=SIZE[2])
    axes[1].set_ylabel("EFSA rows")
    axes[1].set_title("How each structure was obtained")
    axes[1].margins(y=0.16)
    axes[1].grid(axis="x", visible=False)
    return fig


def figure_labels(dist: pd.DataFrame, n_contradictions: int):
    """The seven EFSA conclusion values, grouped by what they are worth.

    Two of the seven are evidence. One is real discordant evidence. Four mean
    the question was not answered. Collapsing the last five into "negative"
    is the single most damaging mistake available in this dataset, so the
    grouping is drawn rather than described.
    """
    import matplotlib.pyplot as plt

    order = {"informative": 0, "ambiguous": 1, "no data": 2}
    d = dist.copy()
    d["_k"] = d["category"].map(lambda c: order.get(c, 3))
    d = d.sort_values(["_k", "n"], ascending=[True, False]).reset_index(drop=True)
    palette = {"informative": FOCAL, "ambiguous": SECONDARY, "no data": NEUTRAL}

    fig, ax = plt.subplots(figsize=(7.4, 3.2), constrained_layout=True)
    xpos = np.arange(len(d))
    ax.bar(xpos, d["n"], color=[palette.get(c, NEUTRAL) for c in d["category"]],
           alpha=0.9, width=0.62)
    for xp, (_, r) in zip(xpos, d.iterrows()):
        ax.annotate(f"{int(r['n']):,}", xy=(xp, r["n"]), xytext=(0, 3),
                    textcoords="offset points", ha="center",
                    fontsize=SIZE[2], color="#333333")
    ax.set_xticks(xpos)
    ax.set_xticklabels(d["label"], rotation=20, ha="right", fontsize=SIZE[2])
    ax.set_ylabel("EFSA rows")
    ax.set_title("Only the dark bars are evidence of an outcome")
    ax.margins(y=0.18)
    ax.grid(axis="x", visible=False)

    handles = [mpl.patches.Patch(color=palette[k], label=k) for k in order if k in set(d["category"])]
    ax.legend(handles=handles, loc="upper right", fontsize=SIZE[1])
    ax.annotate(f"{n_contradictions} substances carry both Positive and Negative\n"
                "across different assessments - the aggregation rule decides",
                xy=(0.5, -0.46), xycoords="axes fraction", ha="center",
                fontsize=SIZE[2], color="#555555")
    return fig


def figure_chemspace(embedding, y, test_mask, n_scaffolds: int):
    """Chemical space of the modelable set, and what the scaffold split does to it.

    Left: t-SNE on Tanimoto distance between ECFP4 fingerprints, positives on
    top. Right: the same map colored by which side of one scaffold split each
    compound landed on -- neighbourhoods move as blocks, which is the whole
    reason a random split would be dishonest here.
    """
    import matplotlib.pyplot as plt

    z = np.asarray(embedding)
    y = np.asarray(y).astype(bool)
    test_mask = np.asarray(test_mask).astype(bool)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)

    axes[0].scatter(z[~y, 0], z[~y, 1], s=7, color=NEUTRAL, alpha=0.45,
                    linewidths=0, label=f"negative (n = {int((~y).sum())})")
    axes[0].scatter(z[y, 0], z[y, 1], s=17, color=ALERT, alpha=0.9,
                    linewidths=0, label=f"positive (n = {int(y.sum())})")
    axes[0].set_title(f"{len(z):,} substances, {n_scaffolds:,} Bemis-Murcko scaffolds")
    axes[0].legend(loc="best", fontsize=SIZE[1], markerscale=1.6)

    axes[1].scatter(z[~test_mask, 0], z[~test_mask, 1], s=7, color=FOCAL, alpha=0.4,
                    linewidths=0, label=f"train (n = {int((~test_mask).sum())})")
    axes[1].scatter(z[test_mask, 0], z[test_mask, 1], s=7, color=SECONDARY, alpha=0.8,
                    linewidths=0, label=f"test (n = {int(test_mask.sum())})")
    axes[1].set_title("One scaffold split: whole neighbourhoods move together")
    axes[1].legend(loc="best", fontsize=SIZE[1], markerscale=1.6)

    for a in axes:
        a.set_xticks([])
        a.set_yticks([])
        a.grid(visible=False)
        a.set_xlabel("t-SNE 1 (Tanimoto)")
        a.set_ylabel("t-SNE 2")
    return fig

def figure_ambiguity(sweep: pd.DataFrame, null, summary: dict,
                     n_substances: int, k: int):
    """Two questions the bench separates, and whether the ranking is noise."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)
    colour = {"uniform": NEUTRAL, "rank_decay": FOCAL, "top1": SECONDARY}
    rng = np.random.default_rng(0)
    chance = summary["uninformative_quality"]

    panels = ((axes[0], "mae_vs_known", "MAE vs settled-identity score", "lower is better"),
              (axes[1], "pr_auc_vs_label", "PR-AUC vs the published label", "higher is better"))
    for ax, metric, ylabel, better in panels:
        for scheme, c in colour.items():
            d = sweep[sweep["scheme"] == scheme]
            qs = np.sort(d["quality"].unique())
            for q in qs:
                v = d.loc[d["quality"] == q, metric].to_numpy()
                ax.plot(q + rng.uniform(-0.011, 0.011, len(v)), v, "o",
                        color=c, alpha=0.5, mec="none", ms=3)
            med = [float(np.median(d.loc[d["quality"] == q, metric])) for q in qs]
            ax.plot(qs, med, "-", color=c, lw=1.5)
            ax.annotate(scheme, (qs[-1], med[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=SIZE[1], color=c, va="center")
        ax.axvline(chance, color=NEUTRAL, ls=":", lw=1.0)
        ax.annotate("ranking\nis noise", xy=(chance, 0.02), xycoords=("data", "axes fraction"),
                    xytext=(3, 0), textcoords="offset points", fontsize=SIZE[2], color=NEUTRAL)
        ax.set_xlabel("ranking quality: P(true identity ranked first)")
        ax.set_ylabel(ylabel)
        ax.set_title(better)
        ax.margins(x=0.18)

    axes[2].hist(null, bins=30, color=NEUTRAL, alpha=0.85)
    axes[2].axvline(summary["statistic"], color=ALERT, lw=1.4)
    axes[2].annotate(f"observed\np = {summary['p_value']:.3f}",
                     xy=(summary["statistic"], 0.96), xycoords=("data", "axes fraction"),
                     xytext=(-4, 0), textcoords="offset points",
                     fontsize=SIZE[2], color=ALERT, va="top", ha="right")
    axes[2].set_xlabel("Spearman vs settled-identity score")
    axes[2].set_ylabel(f"permutations (n = {summary['n_permutations']})")
    axes[2].set_title("Null: the rank of the true identity is uniform")
    axes[2].margins(x=0.04)

    axes[2].annotate(f"n = {n_substances} substances\n{k} candidates each\n"
                     f"5 simulated rankings per quality",
                     xy=(0.02, 0.96), xycoords="axes fraction", fontsize=SIZE[2],
                     color=NEUTRAL, ha="left", va="top")
    return fig


def figure_call_threshold(curve: pd.DataFrame, matched: float, prevalence: float,
                          n_scores: int):
    """Where the prevalence-matched threshold falls, and what it costs."""
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2), constrained_layout=True)

    axes[0].plot(curve["threshold"], curve["call_rate"], "-", color=FOCAL, lw=1.5)
    axes[0].axhline(prevalence, color=NEUTRAL, ls="--", lw=1.0)
    axes[0].annotate(f"positive rate {prevalence:.3f}", xy=(0.02, prevalence),
                     xycoords=("axes fraction", "data"), xytext=(0, 4),
                     textcoords="offset points", fontsize=SIZE[2], color=NEUTRAL)
    axes[0].axvline(matched, color=ALERT, ls="--", lw=1.2)
    axes[0].annotate(f"matched\n{matched:.3f}", xy=(matched, 0.96),
                     xycoords=("data", "axes fraction"), xytext=(4, 0),
                     textcoords="offset points", fontsize=SIZE[2], color=ALERT, va="top")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("calling threshold")
    axes[0].set_ylabel(f"share of scores called (n = {n_scores})")
    axes[0].set_title("The default is where these two cross")

    axes[1].plot(curve["threshold"], curve["precision"], "-", color=FOCAL, lw=1.5)
    axes[1].plot(curve["threshold"], curve["recall"], "-", color=SECONDARY, lw=1.5)
    for col, c in (("precision", FOCAL), ("recall", SECONDARY)):
        _v = curve[col].dropna()
        if len(_v):
            axes[1].annotate(col, (curve.loc[_v.index[-1], "threshold"], _v.iloc[-1]),
                             xytext=(4, 0), textcoords="offset points",
                             fontsize=SIZE[1], color=c, va="center")
    axes[1].axvline(matched, color=ALERT, ls="--", lw=1.2)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("calling threshold")
    axes[1].set_ylabel("held-out precision and recall")
    axes[1].set_title("higher is better on both; they trade")
    axes[1].margins(x=0.14)
    return fig


def figure_identity_size(sweep: pd.DataFrame, quality: float, threshold: float,
                         anchor: float = 0):
    """How the propagation degrades as more look-alikes compete for the identity.

    Right-hand panel is in calls rather than in score error, because a score
    only reaches a decision through a call. `top1` is flat by construction, and
    that flatness is the point: committing to the top slot returns either the
    true probability or the nearest neighbour's, whatever the candidate-set
    size, so its error cannot register that the feature became more ambiguous.

    The x axis is logarithmic and `anchor` marks the median isomer count
    measured on the public study, so the reader can see the bench at the size
    the data has instead of interpolating from a grid of round numbers.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), constrained_layout=True)
    colour = {"uniform": NEUTRAL, "rank_decay": FOCAL, "top1": SECONDARY}
    rng = np.random.default_rng(0)
    ks = np.sort(sweep["k"].unique())
    calls = float(sweep["reference_calls"].iloc[0])
    n = int(sweep["n"].iloc[0])

    d_all = sweep.assign(
        missed_pct=100.0 * sweep["missed"] * sweep["n"] / sweep["reference_calls"])
    panels = ((axes[0], "mae_vs_known", "MAE vs settled-identity score", "lower is better"),
              (axes[1], "missed_pct", "calls missed (% of settled-identity calls)",
               "lower is better"))
    for ax, metric, ylabel, better in panels:
        ax.set_xscale("log")
        if anchor and min(ks) <= anchor <= max(ks):
            ax.axvline(anchor, color=NEUTRAL, lw=0.8, ls=":", zorder=0)
        for scheme, c in colour.items():
            d = d_all[d_all["scheme"] == scheme]
            for k in ks:
                v = d.loc[d["k"] == k, metric].to_numpy()
                # Jitter multiplicatively: on a log axis an additive offset is
                # invisible at k = 1000 and enormous at k = 2.
                ax.plot(k * rng.uniform(0.94, 1.06, len(v)), v, "o",
                        color=c, alpha=0.5, mec="none", ms=3)
            med = [float(np.median(d.loc[d["k"] == k, metric])) for k in ks]
            ax.plot(ks, med, "-", color=c, lw=1.5)
            ax.annotate(scheme, (ks[-1], med[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=SIZE[1], color=c, va="center")
        ax.set_xlabel("candidates competing for the identity (k, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(better)
        ax.margins(x=0.10)

    axes[0].annotate(f"n = {n} substances\n{calls:.0f} settled-identity calls\n"
                     f"ranking quality {quality:.2f}, threshold {threshold:.3f}\n"
                     "5 simulated rankings per size",
                     xy=(0.03, 0.96), xycoords="axes fraction", fontsize=SIZE[2],
                     color=NEUTRAL, ha="left", va="top")
    if anchor and min(ks) <= anchor <= max(ks):
        axes[1].annotate(f"median isomer count\nmeasured upstream: {anchor:.0f}",
                         xy=(anchor, 0.04), xycoords=("data", "axes fraction"),
                         xytext=(-6, 0), textcoords="offset points",
                         fontsize=SIZE[2], color=NEUTRAL, ha="right", va="bottom")
    return fig


def figure_attrition(table: pd.DataFrame, summary: dict):
    """What the structure requirement removes from the dataset.

    Three panels because the answer has three parts: the label rate is what a
    selection-bias worry is about, the identifier coverage is the mechanism,
    and the study-count strata are why the marginal comparison is not the whole
    story. Intervals are exact (Clopper-Pearson) rather than normal-
    approximation: the dropped group has 55 positives.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4), constrained_layout=True)
    colour = {"structure resolved": FOCAL, "structure unresolved": SECONDARY}
    order = ["structure resolved", "structure unresolved"]
    d = table.set_index("group").loc[order]
    x = np.arange(len(order))

    for xi, g in zip(x, order):
        axes[0].errorbar([xi], [100 * d.loc[g, "positive_rate"]],
                         yerr=[[100 * (d.loc[g, "positive_rate"] - d.loc[g, "ci_low"])],
                               [100 * (d.loc[g, "ci_high"] - d.loc[g, "positive_rate"])]],
                         fmt="o", ms=6, lw=1.2, capsize=4,
                         color=colour[g], ecolor=NEUTRAL, mec="none")
    for xi, g in zip(x, order):
        axes[0].annotate(f"{100 * d.loc[g, 'positive_rate']:.1f} %\nn = {d.loc[g, 'n_substances']}",
                         (xi, d.loc[g, "ci_high"] * 100), xytext=(0, 6),
                         textcoords="offset points", ha="center",
                         fontsize=SIZE[2], color=colour[g])
    lo, hi = summary["odds_ratio_ci"]
    axes[0].set_title(f"odds ratio {summary['odds_ratio']:.2f} "
                      f"[{lo:.2f}, {hi:.2f}], p = {summary['fisher_p']:.2f}")
    axes[0].set_ylabel("positive substances (%)")
    axes[0].set_ylim(0, 12)

    axes[1].bar(x, 100 * d["identifier_coverage"],
                color=[colour[g] for g in order], width=0.55)
    for xi, g in zip(x, order):
        axes[1].annotate(f"{100 * d.loc[g, 'identifier_coverage']:.0f} %",
                         (xi, 100 * d.loc[g, "identifier_coverage"]), xytext=(0, 4),
                         textcoords="offset points", ha="center",
                         fontsize=SIZE[2], color=colour[g])
    axes[1].set_title("the mechanism: no registry identifier, no structure")
    axes[1].set_ylabel("substances with a CAS number (%)")
    axes[1].set_ylim(0, 108)

    strata = summary["strata"]
    labels_x = {False: "1 study", True: "2 or more"}
    for resolved, c in ((True, FOCAL), (False, SECONDARY)):
        part = strata[strata["resolved"] == resolved].sort_values("multi_study")
        xs = np.arange(len(part))
        axes[2].plot(xs, 100 * part["mean"], "o-", color=c, lw=1.5, ms=5, mec="none")
        axes[2].annotate("resolved" if resolved else "unresolved",
                         (xs[-1], 100 * part["mean"].to_numpy()[-1]),
                         xytext=(4, 0), textcoords="offset points",
                         fontsize=SIZE[1], color=c, va="center")
        axes[2].set_xticks(xs, [labels_x[bool(v)] for v in part["multi_study"]])
    axes[2].set_title("the strata disagree in sign")
    axes[2].set_ylabel("positive substances (%)")
    axes[2].set_xlabel("informative studies per substance")
    axes[2].margins(x=0.35)
    axes[2].set_ylim(0, 12)

    for ax in (axes[0], axes[1]):
        ax.set_xticks(x, ["resolved\n(modelled)", "unresolved\n(dropped)"])
    axes[0].annotate(f"{summary['n_labelled']} labelled substances\n"
                     f"{summary['n_dropped']} dropped for structure",
                     xy=(0.03, 0.04), xycoords="axes fraction", fontsize=SIZE[2],
                     color=NEUTRAL, ha="left", va="bottom")
    return fig

def figure_retention(oof: pd.DataFrame, within: pd.DataFrame, scales: dict):
    """Whether a public retention index can order candidate isomers.

    Three panels because the answer flips between them: globally the predictor
    works (left), but the scale panel shows its error is wider than the spread
    the index has inside a fixed formula (middle), and the within-formula rank
    correlations confirm that what it learned is molecular size, which is
    constant among isomers by definition (right). The scale panel is the one
    that decides the question, so the comparison there is drawn against the
    between-study dispersion of the same structure, which is the floor no
    predictor of this reference can beat.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5), constrained_layout=True)

    ax = axes[0]
    lim = [min(oof["ri"].min(), oof["pred"].min()) - 80,
           max(oof["ri"].max(), oof["pred"].max()) + 80]
    ax.plot(lim, lim, color=NEUTRAL, lw=0.9, zorder=1)
    ax.fill_between(lim, [v - scales["floor"] for v in lim], [v + scales["floor"] for v in lim],
                    color=NEUTRAL, alpha=0.30, lw=0, zorder=1,
                    label=f"between-study floor (\u00b1{scales['floor']:.0f})")
    ax.scatter(oof["ri"], oof["pred"], s=11, color=FOCAL, alpha=0.55, lw=0, zorder=2)
    ax.set(xlabel="measured retention index", ylabel="predicted (out of fold)",
           xlim=lim, ylim=lim, title="A  globally it predicts")
    ax.annotate(f"n = {len(oof)} structures\nMAE = {scales['mae']:.0f}\n"
                f"Spearman = {scales['spearman']:.2f}",
                (0.04, 0.96), xycoords="axes fraction", va="top", fontsize=SIZE[2])
    ax.legend(loc="lower right")

    ax = axes[1]
    bars = [("between-study floor", scales["floor"], NEUTRAL),
            ("index spread within\na formula (median)", scales["within_range"], SECONDARY),
            ("predictor error (MAE)", scales["mae"], ALERT),
            ("predicting the mean", scales["mean_mae"], NEUTRAL)]
    ypos = np.arange(len(bars))[::-1]
    ax.barh(ypos, [b[1] for b in bars], color=[b[2] for b in bars], height=0.6)
    for yi, (_, v, _) in zip(ypos, bars):
        ax.annotate(f"{v:.0f}", (v, yi), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=SIZE[2])
    ax.set_yticks(ypos, [b[0] for b in bars], fontsize=SIZE[2])
    ax.set(xlabel="retention index units", xlim=(0, max(b[1] for b in bars) * 1.22),
           ylim=(-1.25, len(bars) - 0.45),
           title="B  the error is wider than the signal")
    ax.annotate("an error above the within-formula spread cannot order isomers",
                (max(b[1] for b in bars) * 1.20, -0.95), ha="right", fontsize=SIZE[2],
                color=ALERT)
    ax.grid(axis="y", visible=False)

    ax = axes[2]
    rng = np.random.default_rng(0)
    v = within["rho_dentro"].to_numpy(dtype=float)
    ax.axhline(0.0, color=NEUTRAL, lw=0.9)
    ax.scatter(rng.normal(0, 0.045, len(v)), v, s=26, color=FOCAL, alpha=0.75, lw=0)
    ax.plot([-0.16, 0.16], [np.median(v)] * 2, color=ALERT, lw=1.6)
    ax.annotate(f"median {np.median(v):+.2f}", (0.17, np.median(v)), color=ALERT,
                va="center", fontsize=SIZE[2])
    ax.annotate(f"{int((v > 0).sum())} of {len(v)} formulas\nordered better than chance",
                (0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=SIZE[2])
    ax.annotate("orders correctly \u2191", (-0.42, 0.93), xycoords=("data", "axes fraction"),
                fontsize=SIZE[2], color=NEUTRAL)
    ax.set(xlim=(-0.45, 0.62), ylim=(-1.15, 1.15), xticks=[],
           ylabel="Spearman within formula",
           title="C  within a formula it does not")
    return fig

def figure_isomer_space(sizes: pd.DataFrame, bench_range=(2, 10), median_formula: str = ""):
    """How large a feature's candidate set really is, against the bench's k.

    An ECDF rather than a histogram: the question is what fraction of features
    carry at least a given ambiguity, and the axis is logarithmic because the
    sizes span four orders of magnitude. The bench range is drawn as a band so
    the comparison is read off the figure instead of the caption.
    """
    import matplotlib.pyplot as plt

    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 3.8), constrained_layout=True)
    v = np.sort(sizes["n_isomers"].dropna().to_numpy(dtype=float))
    frac = np.arange(1, len(v) + 1) / len(v)
    ax.axvspan(bench_range[0], bench_range[1], color=SECONDARY, alpha=0.22, lw=0,
               label=f"bench sweep (k = {bench_range[0]}\u2013{bench_range[1]})")
    ax.step(v, 100 * frac, where="post", color=FOCAL, lw=1.8)
    med = float(np.median(v))
    ax.plot([med], [50], "o", color=ALERT, ms=6, zorder=3)
    ax.annotate(f"median {med:,.0f} isomers"
                + (f"\n({median_formula})" if median_formula else ""),
                (med, 50), xytext=(-14, 16), textcoords="offset points",
                ha="right", color=ALERT, fontsize=SIZE[2],
                arrowprops=dict(arrowstyle="-", color=ALERT, lw=0.8))
    ax.annotate(f"n = {len(v)} formulas\n"
                f"{100 * float((v <= bench_range[1]).mean()):.1f} % inside the bench range",
                (0.03, 0.95), xycoords="axes fraction", va="top", fontsize=SIZE[2])
    ax.set(xscale="log", xlabel="isomers of the feature's formula in PubChem",
           ylabel="% of features at or below", ylim=(0, 100),
           title="Real candidate-set size against the bench's k")
    ax.legend(loc="lower right")
    return fig


def figure_decision_chain(candidates: pd.DataFrame, priority: pd.DataFrame,
                          ad_threshold: float, anchor: float = 0):
    """The whole chain on measured features: coverage, depth, decisions.

    Three things the reader has to see together. How much of each feature's
    identity weight survives the applicability domain; how that survival
    behaves as the candidate list deepens, which is the question the isomer
    counts raise; and what the decisions look like at the end of it.

    Coverage and in-domain weight are the same quantity here, not two: a
    candidate outside the domain is dropped from the weighted sum rather than
    given a prediction, so `covered_weight` *is* the in-domain share. The
    panels do not pretend to be independent checks of each other.
    """
    apply_style()
    import matplotlib.pyplot as plt

    d = candidates.copy()
    d["in_domain"] = d["in_domain"].astype(bool)
    n_feat = int(priority["feature_id"].nunique())
    depth = (d.groupby("feature_id")
              .apply(lambda f: pd.Series({
                  int(k): float(f.loc[f["rank"] <= k, "in_domain"].mean())
                  for k in range(1, int(d["rank"].max()) + 1)}),
                     include_groups=False)
              .mean())
    assigned = d.loc[d.get("is_assigned", pd.Series(False, index=d.index)).astype(bool)]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.3), constrained_layout=True)

    ax = axes[0]
    cw = priority["covered_weight"].to_numpy(dtype=float)
    ax.hist(cw, bins=np.linspace(0, 1, 21), color=FOCAL, alpha=0.85)
    ax.axvline(float(np.median(cw)), color=SECONDARY, lw=1.5)
    ax.annotate(f"median {np.median(cw):.2f}", xy=(float(np.median(cw)), 0.02),
                xycoords=("data", "axes fraction"), xytext=(5, 0),
                textcoords="offset points", fontsize=SIZE[2], color=SECONDARY, va="bottom")
    ax.set_xlabel("identity weight inside the applicability domain")
    ax.set_ylabel("features")
    ax.set_title("how much of a feature is evaluable")
    ax.annotate(f"{n_feat} features, {len(d)} candidates\n"
                f"Tanimoto cutoff {ad_threshold:.2f}\n"
                f"{int((cw == 0).sum())} features with nothing evaluable",
                xy=(0.97, 0.98), xycoords="axes fraction", fontsize=SIZE[2],
                color=NEUTRAL, ha="right", va="top")

    ax = axes[1]
    ax.plot(depth.index, depth.to_numpy(), "-o", color=FOCAL, lw=1.5, ms=3.5)
    if len(assigned):
        ax.plot([1], [float(assigned["in_domain"].mean())], "o", color=SECONDARY, ms=7, zorder=3)
        ax.annotate(f"rank 1 is the deposited\nassignment: {100 * assigned['in_domain'].mean():.0f} % in domain\n"
                    f"({len(assigned)} of {n_feat} features have one)",
                    xy=(1, float(assigned["in_domain"].mean())), xytext=(14, -6),
                    textcoords="offset points", fontsize=SIZE[2], color=SECONDARY, va="top")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    ax.set_xlabel("candidates kept per feature (by rank)")
    ax.set_ylabel("mean in-domain share of weight")
    ax.set_title("lower is worse, and it is the depth that lowers it")
    if anchor:
        ax.annotate(f"the study's median isomer count is {anchor:.0f};\n"
                    f"this axis stops at {int(d['rank'].max())}, which is what was built",
                    xy=(0.97, 0.06), xycoords="axes fraction", fontsize=SIZE[2],
                    color=NEUTRAL, ha="right", va="bottom")

    ax = axes[2]
    tight = priority["tightening_factor"].to_numpy(dtype=float)
    groups = [("limit unchanged", int((tight <= 1.0001).sum())),
              ("limit tightened", int((tight > 1.0001).sum())),
              ("nothing evaluable", int((priority["covered_weight"] == 0).sum()))]
    ax.barh([g[0] for g in groups], [g[1] for g in groups],
            color=[NEUTRAL, ALERT, SECONDARY], alpha=0.9)
    for i, (_, v) in enumerate(groups):
        ax.annotate(f"{v}", xy=(v, i), xytext=(4, 0), textcoords="offset points",
                    fontsize=SIZE[2], va="center", color="black")
    ax.invert_yaxis()
    ax.set_xlabel("features")
    ax.set_title("what the chain decides")
    _tf = tight[tight > 1.0001]
    ax.annotate(("no feature crosses the calling threshold"
                 if not len(_tf) else
                 f"all {len(_tf)} land on the genotoxic limit: {np.min(_tf):.0f}x stricter"
                 if np.isclose(np.min(_tf), np.max(_tf)) else
                 f"tightening among those that cross: "
                 f"{np.min(_tf):.0f}x to {np.max(_tf):.0f}x"),
                xy=(0.98, 0.04), xycoords="axes fraction", fontsize=SIZE[2],
                color=NEUTRAL, ha="right", va="bottom")
    ax.margins(x=0.22)
    return fig


def figure_isomer_hero(features: pd.DataFrame, accession: str = ""):
    """The one number the notebook is built around: how many compounds a single
    named peak could actually be.

    A banner, not an analysis figure -- it is the first thing on screen. Every
    resolved feature of the public study is one point on a log axis of its
    isomer count; the median is the headline. The band at ten and below is
    shaded because that is the size a naive candidate list assumes, and almost
    nothing lives there.
    """
    import matplotlib.pyplot as plt

    apply_style()
    n = np.sort(features.loc[features["n_isomers_total"] > 0,
                             "n_isomers_total"].to_numpy(dtype=float))
    med = float(np.median(n))
    pct_le10 = 100.0 * np.mean(n <= 10)

    fig, ax = plt.subplots(figsize=(8.6, 2.4), constrained_layout=True)
    ax.set_xscale("log")
    lo, hi = max(1.0, n.min() * 0.8), n.max() * 1.25
    ax.axvspan(lo, 10, color=NEUTRAL, alpha=0.12, lw=0, zorder=0)
    rng = np.random.default_rng(0)
    ax.scatter(n, rng.uniform(0.12, 0.88, len(n)), s=10, color=FOCAL,
               alpha=0.45, edgecolors="none", zorder=2)
    ax.axvline(med, color=ALERT, lw=1.6, zorder=3)
    ax.annotate(f"median {med:.0f}", xy=(med, 1.0), xycoords=("data", "axes fraction"),
                xytext=(6, -2), textcoords="offset points", fontsize=SIZE[1],
                color=ALERT, ha="left", va="top", fontweight="bold")
    ax.annotate(f"{pct_le10:.1f}% of features\nhave 10 or fewer",
                xy=(10, 0.02), xycoords=("data", "axes fraction"),
                xytext=(-6, 0), textcoords="offset points", fontsize=SIZE[2],
                color=NEUTRAL, ha="right", va="bottom")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("real isomers of the feature's formula in PubChem (log scale)")
    ttl = "one GC-MS peak, this many candidate compounds"
    if accession:
        ttl += f"   ·   {accession}, {len(n)} features"
    ax.set_title(ttl)
    return fig
