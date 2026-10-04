"""Bemis-Murcko scaffold splitting.

A random split over this dataset gives inflated metrics: there are entire
families of congeners (pesticides, mycotoxins, alkaloids) and it is enough for
one member to fall in training for the model to recognize its siblings in
test. Here scaffold groups go entirely to one side or the other.

The order of the groups is shuffled with the seed (randomized scaffold
split), so different seeds give different repartitions and the metrics can be
summarized with median and IQR instead of a single figure.
"""

from collections import defaultdict

import numpy as np
import pandas as pd


def scaffold_groups(scaffolds) -> dict:
    """scaffold -> positional indices. Acyclic ones each go to their own group."""
    groups = defaultdict(list)
    for i, s in enumerate(scaffolds):
        key = s if s else f"__acyclic_{i}"
        groups[key].append(i)
    return dict(groups)


def split_scaffold(scaffolds, seed: int, frac_test: float = 0.2):
    """Returns (idx_train, idx_test) without sharing any scaffold."""
    groups = list(scaffold_groups(scaffolds).values())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    n_target = int(round(frac_test * sum(len(g) for g in groups)))
    test, train = [], []
    for g in groups:
        (test if len(test) + len(g) <= n_target else train).extend(g)
    return np.array(sorted(train)), np.array(sorted(test))


def multiseed_splits(scaffolds, seeds, frac_test: float = 0.2):
    return [(s, *split_scaffold(scaffolds, s, frac_test)) for s in seeds]


def split_summary(y, scaffolds, seeds, frac_test: float = 0.2) -> pd.DataFrame:
    """Sizes and prevalence of each split, to see if any ends up without positives."""
    y = np.asarray(y, dtype=float)
    rows = []
    for seed, tr, te in multiseed_splits(scaffolds, seeds, frac_test):
        rows.append({
            "seed": seed,
            "n_train": len(tr), "n_test": len(te),
            "pos_train": int(y[tr].sum()), "pos_test": int(y[te].sum()),
            "prev_train_%": round(100 * y[tr].mean(), 2),
            "prev_test_%": round(100 * y[te].mean(), 2),
            "scaffolds_train": len(set(np.asarray(scaffolds, dtype=object)[tr])),
            "scaffolds_test": len(set(np.asarray(scaffolds, dtype=object)[te])),
        })
    return pd.DataFrame(rows)


def identity_groups(scaffolds, identities, fingerprints=None, acyclic_threshold=None):
    """Stable ring-scaffold groups and explicit acyclic series robustness groups.

    With no threshold, each acyclic parent is its own group. With a threshold,
    connected components of acyclic ECFP4 similarities >= threshold stay together.
    Components use structures only, without labels or fitted models. This is a
    sensitivity protocol, not a universal definition of a chemical series.
    """
    scaffolds = list(scaffolds)
    identities = list(identities)
    if len(scaffolds) != len(identities) or len(set(identities)) != len(identities):
        raise ValueError("One unique standardized identity is required per structure.")
    groups = np.array([f"ring:{s}" if s else f"acyclic:{k}"
                       for s, k in zip(scaffolds, identities)], dtype=object)
    if acyclic_threshold is None:
        return groups
    if fingerprints is None or not 0 < acyclic_threshold <= 1:
        raise ValueError("Fingerprints and a threshold in (0, 1] are required.")
    from rdkit import DataStructs
    acyclic = [i for i, s in enumerate(scaffolds) if not s]
    parent = {i: i for i in acyclic}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for position, i in enumerate(acyclic):
        earlier = acyclic[:position]
        similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[i], [fingerprints[j] for j in earlier])
        for j, similarity in zip(earlier, similarities):
            if similarity >= acyclic_threshold:
                parent[root(i)] = root(j)
    components = defaultdict(list)
    for i in acyclic:
        components[root(i)].append(i)
    for component in components.values():
        name = "acyclic-series:" + min(identities[i] for i in component)
        groups[component] = name
    return groups
