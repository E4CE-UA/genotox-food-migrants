"""Declared-ambiguity bench: does propagating identity recover the answer?

Section 13 propagates a genotoxicity prediction over a feature's candidate
identities. On measured data that machinery cannot be checked, because the true
identity of a GC-MS feature is exactly what is unknown -- there is nothing to
score against. On the labelled set there is: every substance has one structure
and one conclusion. So the candidate list is built deliberately, the ranking is
given a quality that is set rather than measured, and the propagated score is
compared against the score the model would return with the identity settled.

What this establishes: whether the propagation behaves as claimed, and at what
ranking quality trusting the top-ranked candidate stops being defensible.

What it does not establish: that a real spectral match score is a ranking of
any particular quality. The rankings here are simulated. Decoys are structural
nearest neighbours, which is the hard case -- an isomer-rich spectrum produces
a list of look-alikes -- but they are not the list a spectrometer produces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

SCHEMES = ("uniform", "rank_decay", "top1")


def out_of_fold(predictions, model: str, n_rows: int):
    """Mean out-of-fold probability per row, over the seeds that held it out.

    ``predictions`` is the dict models.evaluate returns. Rows never held out
    come back as nan with a fold count of 0, and have to be dropped: an
    in-training probability would make the reference look better than it is.
    """
    total = np.zeros(n_rows, dtype=float)
    count = np.zeros(n_rows, dtype=int)
    for (name, _seed), (_y_te, p_te, idx_te) in predictions.items():
        if name != model:
            continue
        total[idx_te] += np.asarray(p_te, dtype=float)
        count[idx_te] += 1
    p = np.divide(total, count, out=np.full(n_rows, np.nan), where=count > 0)
    return p, count


def tanimoto(fingerprint_matrix) -> np.ndarray:
    """Full Tanimoto similarity matrix for a binary fingerprint matrix."""
    x = np.asarray(fingerprint_matrix, dtype=np.float32)
    intersection = x @ x.T
    popcount = x.sum(axis=1)
    union = popcount[:, None] + popcount[None, :] - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, intersection / union, 0.0)


def candidate_sets(similarity, pool, k: int) -> np.ndarray:
    """For each row in ``pool``, itself plus its k-1 nearest neighbours in pool.

    Column 0 is always the true identity; the caller is responsible for never
    letting a weighting scheme see that. Neighbours rather than random draws
    because a candidate list of look-alikes is the case that actually hurts:
    decoys with unrelated structures would be easy to average over.
    """
    pool = np.asarray(pool)
    if k < 2:
        raise ValueError("k must be at least 2 for a candidate set to exist")
    if k > len(pool):
        raise ValueError(f"k={k} exceeds the pool ({len(pool)} rows)")
    sub = similarity[np.ix_(pool, pool)].copy()
    np.fill_diagonal(sub, -1.0)
    order = np.argsort(-sub, axis=1)[:, : k - 1]
    return np.column_stack([pool, pool[order]])


def simulate_ranking(n: int, k: int, quality: float, rng) -> np.ndarray:
    """Rank position of the true candidate under a ranking of given quality.

    ``quality`` is P(true identity ranked first). The rest of the mass spreads
    uniformly over the other k-1 positions, so quality = 1/k is a ranking that
    carries no information and quality = 1 is a solved identity.
    """
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality must be a probability")
    ranked_first = rng.random(n) < quality
    elsewhere = rng.integers(1, k, size=n)
    return np.where(ranked_first, 0, elsewhere)


def rank_weights(k: int, scheme: str, tau: float = 1.0) -> np.ndarray:
    """Weights by rank position. Never by identity -- that would be the answer."""
    if scheme == "uniform":
        w = np.ones(k)
    elif scheme == "rank_decay":
        w = np.exp(-np.arange(k) / tau)
    elif scheme == "top1":
        w = np.zeros(k)
        w[0] = 1.0
    else:
        raise ValueError(f"unknown weighting scheme: {scheme}")
    return w / w.sum()


def propagate(p_candidates, rank_true, scheme: str, tau: float = 1.0) -> np.ndarray:
    """Identity-weighted score with the true candidate placed at ``rank_true``.

    ``p_candidates[:, 0]`` is the true candidate's probability and the rest are
    decoys. The ranking decides which slot the true one occupies; the weights
    then apply by slot, which is all a real pipeline can do.
    """
    p_candidates = np.asarray(p_candidates, dtype=float)
    n, k = p_candidates.shape
    rank_true = np.asarray(rank_true)
    w = rank_weights(k, scheme, tau)
    rows = np.arange(n)
    by_rank = np.empty_like(p_candidates)
    by_rank[rows, rank_true] = p_candidates[:, 0]
    free = np.ones((n, k), dtype=bool)
    free[rows, rank_true] = False
    by_rank[free] = p_candidates[:, 1:].ravel()
    return by_rank @ w


def bench(p_candidates, p_known, y_true, rank_true, schemes=SCHEMES,
          tau: float = 1.0) -> pd.DataFrame:
    """Each weighting scheme against the score a settled identity would give."""
    rows = []
    for scheme in schemes:
        score = propagate(p_candidates, rank_true, scheme, tau)
        rows.append({
            "scheme": scheme,
            "mae_vs_known": float(np.mean(np.abs(score - p_known))),
            "rmse_vs_known": float(np.sqrt(np.mean((score - p_known) ** 2))),
            "spearman_vs_known": float(spearmanr(score, p_known)[0]),
            "pr_auc_vs_label": float(average_precision_score(y_true, score)),
        })
    return pd.DataFrame(rows)


def quality_sweep(p_candidates, p_known, y_true, qualities, schemes=SCHEMES,
                  tau: float = 1.0, seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """The bench across ranking qualities, one row per quality, scheme and seed.

    Several seeds because a single simulated ranking is one draw: the
    differences between schemes at a given quality are small enough that one
    draw cannot tell them apart. Aggregate with median and spread, the way the
    multi-seed split results are aggregated.
    """
    p_candidates = np.asarray(p_candidates, dtype=float)
    n, k = p_candidates.shape
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for quality in qualities:
            rank_true = simulate_ranking(n, k, float(quality), rng)
            for scheme in schemes:
                score = propagate(p_candidates, rank_true, scheme, tau)
                rows.append({"quality": float(quality), "scheme": scheme, "seed": int(seed),
                             "mae_vs_known": float(np.mean(np.abs(score - p_known))),
                             "spearman_vs_known": float(spearmanr(score, p_known)[0]),
                             "pr_auc_vs_label": float(average_precision_score(y_true, score))})
    return pd.DataFrame(rows)


def sweep_median(sweep: pd.DataFrame, value: str = "mae_vs_known") -> pd.DataFrame:
    """Median across seeds, with the min-max spread, per quality and scheme."""
    g = sweep.groupby(["quality", "scheme"])[value]
    return (pd.DataFrame({"median": g.median(), "low": g.min(), "high": g.max()})
            .reset_index())


def permutation_test(p_candidates, p_known, quality: float, n_permutations: int = 500,
                     scheme: str = "rank_decay", tau: float = 1.0, seed: int = 0):
    """Does a ranking of this quality carry information into the propagated score?

    Observed: the true identity is ranked first with probability ``quality``.
    Null: its rank is uniform over the k slots, which is a ranking made of
    noise. The statistic is Spearman against the settled-identity score, and
    the p-value is the share of null draws that match or beat it. Returns the
    summary and the null draws, so the section can show the distribution rather
    than assert the number.
    """
    rng = np.random.default_rng(seed)
    p_candidates = np.asarray(p_candidates, dtype=float)
    n, k = p_candidates.shape
    observed_rank = simulate_ranking(n, k, quality, rng)
    observed = float(spearmanr(propagate(p_candidates, observed_rank, scheme, tau), p_known)[0])
    null = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        rank_null = simulate_ranking(n, k, 1.0 / k, rng)
        null[i] = spearmanr(propagate(p_candidates, rank_null, scheme, tau), p_known)[0]
    summary = {
        "statistic": observed,
        "null_median": float(np.median(null)),
        "null_p95": float(np.quantile(null, 0.95)),
        "p_value": float((np.sum(null >= observed) + 1) / (n_permutations + 1)),
        "n_permutations": int(n_permutations),
        "uninformative_quality": float(1.0 / k),
    }
    return summary, null


def decision_change(p_candidates, p_known, threshold: float, quality: float = 0.6,
                    schemes=SCHEMES, tau: float = 1.0,
                    seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """How often a scheme's call differs from the settled-identity call.

    The error metrics say how far a propagated score sits from the score a
    resolved identity would give. They do not say whether that distance
    matters, because a score only matters through the call made on it. So the
    same comparison is repeated on the calls: `score >= threshold` against
    `p_known >= threshold`, and the disagreement is split by direction.

    `missed` is the consequential one -- the settled identity would have called
    the substance and the scheme does not. `added` costs analyst time; `missed`
    is the error the TTC asymmetry is there to avoid.
    """
    p_candidates = np.asarray(p_candidates, dtype=float)
    p_known = np.asarray(p_known, dtype=float)
    n, k = p_candidates.shape
    reference = p_known >= threshold
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        rank_true = simulate_ranking(n, k, float(quality), rng)
        for scheme in schemes:
            call = propagate(p_candidates, rank_true, scheme, tau) >= threshold
            rows.append({
                "scheme": scheme, "seed": int(seed), "n": int(n),
                "reference_calls": int(reference.sum()),
                "changed": float(np.mean(call != reference)),
                "missed": float(np.mean(reference & ~call)),
                "added": float(np.mean(~reference & call)),
            })
    return pd.DataFrame(rows)


def log_sizes(pool_n: int, anchor: float = None,
              decades=(2, 3, 5, 10, 20, 50, 100, 200, 500, 1000)) -> tuple:
    """Candidate-set sizes on a log grid, clipped to the pool.

    A linear grid of 2..10 answers a question nobody has: real GC-MS features
    have hundreds to thousands of isomers per formula, so the interesting part
    of the curve is two orders of magnitude to the right of it and a linear
    grid spends all its points in the wrong place. `anchor` inserts the size
    actually measured on the public study, so the bench curve can be read at
    the value the data has rather than at a round number.

    Sizes above the pool are dropped rather than clamped: there are only
    `pool_n` substances to draw look-alikes from, and a bench that silently
    reported k = 1000 while computing k = pool_n would be answering a
    different question.
    """
    ks = {int(k) for k in decades if 2 <= int(k) <= pool_n}
    if anchor and 2 <= int(anchor) <= pool_n:
        ks.add(int(anchor))
    return tuple(sorted(ks))


def size_sweep(similarity, pool, p_out_of_fold, y_true, threshold: float,
               sizes=(2, 3, 5, 10), quality: float = 0.6, schemes=SCHEMES,
               tau: float = 1.0, seeds=(0, 1, 2, 3, 4)) -> pd.DataFrame:
    """The bench across candidate-set sizes, at a fixed ranking quality.

    The quality sweep asks how good the ranking has to be. This asks how much
    ambiguity the propagation can absorb: the same substances, the same
    ranking quality, more look-alikes competing for the identity.

    k = 1 is not computed because there is nothing to compute -- one candidate
    is a resolved identity, every scheme returns the known probability, and all
    the metrics take their perfect value by construction. The curve starts at
    the first size where the schemes can differ.
    """
    pool = np.asarray(pool)
    p_oof = np.asarray(p_out_of_fold, dtype=float)
    y_true = np.asarray(y_true)
    p_known = p_oof[pool]
    reference = p_known >= threshold
    rows = []
    for k in sizes:
        if k < 2 or k > len(pool):
            continue
        p_cand = p_oof[candidate_sets(similarity, pool, k=int(k))]
        for seed in seeds:
            rng = np.random.default_rng(seed)
            rank_true = simulate_ranking(len(pool), int(k), float(quality), rng)
            for scheme in schemes:
                score = propagate(p_cand, rank_true, scheme, tau)
                call = score >= threshold
                rows.append({
                    "k": int(k), "scheme": scheme, "seed": int(seed),
                    "n": int(len(pool)), "reference_calls": int(reference.sum()),
                    "mae_vs_known": float(np.mean(np.abs(score - p_known))),
                    "spearman_vs_known": float(spearmanr(score, p_known)[0]),
                    "pr_auc_vs_label": float(average_precision_score(y_true, score)),
                    "changed": float(np.mean(call != reference)),
                    "missed": float(np.mean(reference & ~call)),
                })
    return pd.DataFrame(rows)
