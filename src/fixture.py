"""An illustrative stand-in for a GC-MS candidate export.

WHY THIS EXISTS
---------------
The integration in sections 12-16 consumes two tables produced by a separate
chromatography pipeline. That database is not part of this project and is not
available here. Rather than ship cells that render "inactive" and leave the
reader unable to tell whether the code works, this module builds a small table
with the same column contract, so the integration runs end to end and its
output can be inspected.

WHAT IS REAL AND WHAT IS NOT
----------------------------
Real: the structures. Every SMILES below is a genuine compound reported as a
migrant from food-contact materials, so the probabilities the model assigns to
them are real predictions about real chemistry, made by the model trained in
this notebook.

Invented: everything describing the *measurement*. Retention indices, spectral
evidence scores, candidate rankings, MSI levels and the identity assignments
are fabricated to exercise the code paths. They are not measurements and no
statement about any of these compounds' presence, concentration or
identification follows from them.

CONSEQUENCE
-----------
Numbers produced from this fixture describe the behaviour of the code, not the
world. Any cell that uses it must say so on screen. `is_fixture` is carried in
the returned metadata so the notebook cannot display fixture output without
knowing it is fixture output.
"""

import numpy as np
import pandas as pd

# Compounds reported as food-contact migrants. name, SMILES, Cramer class.
# The Cramer classes are the conventional assignments for these structures and
# are used only to give the TTC layer something to act on.
MIGRANTS = [
    ("dibutyl phthalate", "CCCCOC(=O)c1ccccc1C(=O)OCCCC", "III"),
    ("bisphenol A", "CC(C)(c1ccc(O)cc1)c1ccc(O)cc1", "III"),
    ("butylated hydroxytoluene", "Cc1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1", "I"),
    ("caprolactam", "O=C1CCCCCN1", "III"),
    ("benzophenone", "O=C(c1ccccc1)c1ccccc1", "III"),
    ("dibutyl adipate", "CCCCOC(=O)CCCCC(=O)OCCCC", "I"),
    ("diethyl phthalate", "CCOC(=O)c1ccccc1C(=O)OCC", "III"),
    ("2,4-di-tert-butylphenol", "CC(C)(C)c1ccc(O)c(C(C)(C)C)c1", "III"),
    ("triphenyl phosphate", "O=P(Oc1ccccc1)(Oc1ccccc1)Oc1ccccc1", "III"),
    ("epsilon-caprolactone", "O=C1CCCCCO1", "I"),
    ("styrene", "C=Cc1ccccc1", "III"),
    ("4-nonylphenol", "CCCCCCCCCc1ccc(O)cc1", "III"),
    ("acetophenone", "CC(=O)c1ccccc1", "I"),
    ("diisobutyl phthalate", "CC(C)COC(=O)c1ccccc1C(=O)OCC(C)C", "III"),
    ("tributyl citrate", "CCCCOC(=O)CC(O)(C(=O)OCCCC)CC(=O)OCCCC", "I"),
]

TTC_BY_CRAMER = {"I": 1800.0, "II": 540.0, "III": 90.0}


def build(n_features: int = 40, n_candidates: int = 5, seed: int = 0):
    """Builds a candidates/features pair with the export column contract.

    Returns (candidates, features, info). `info["is_fixture"]` is always True.
    """
    rng = np.random.default_rng(seed)
    pool = list(range(len(MIGRANTS)))

    crows, frows = [], []
    for f in range(n_features):
        fid = f"DEMO-{f + 1:03d}"
        picks = rng.choice(pool, size=min(n_candidates, len(pool)), replace=False)
        # Evidence decays with rank, with enough noise that the ordering is not
        # always the same as the evidence ordering.
        ev = np.sort(rng.normal(60, 18, size=len(picks)))[::-1].clip(1, None)

        # MSI level: 1 means confirmed against an authentic standard.
        level = int(rng.choice([1, 2, 3], p=[0.20, 0.35, 0.45]))
        truth = None
        if level == 1:
            # Confirmed identity is deliberately NOT always the top candidate:
            # if it were, the ablation reference would be circular by design.
            pos = int(rng.choice(len(picks), p=[0.45] + [0.55 / (len(picks) - 1)] * (len(picks) - 1)))
            truth = MIGRANTS[picks[pos]][1]

        for r, (idx, e) in enumerate(zip(picks, ev), start=1):
            name, smi, cramer = MIGRANTS[idx]
            crows.append(dict(
                feature_id=fid, rank=r, inchikey=f"FIXTURE-{idx:02d}", name=name,
                evidence=float(e), n_samples_cand=int(rng.integers(1, 9)),
                smiles=smi, cas_official=None,
                # Two thirds of the candidate space has no experimental value,
                # which is the gap the model is there to fill.
                efsa_genotoxic=(float(rng.integers(0, 2)) if rng.random() < 0.34 else np.nan),
                genotox_alert=int(rng.random() < 0.30),
                ttc_class="Cramer", ttc_ug_day=TTC_BY_CRAMER[cramer], cramer_class=cramer,
            ))

        frows.append(dict(
            feature_id=fid, n_samples=int(rng.integers(2, 20)),
            rt_mean=float(rng.uniform(4, 32)), ri_mean=float(rng.uniform(900, 2400)),
            inchikey_id=(f"FIXTURE-{picks[pos]:02d}" if truth else None),
            name_id=(MIGRANTS[picks[pos]][0] if truth else None),
            msi_level=level, margin=float(ev[0] - ev[1]), coverage=float(rng.uniform(.3, 1)),
            spectral_score=float(ev[0]), delta_ri=float(rng.normal(0, 30)),
            delta_airi=float(rng.normal(0, 25)),
            mplus_confirms=int(rng.random() < .5), frag_confirms=int(rng.random() < .6),
            frag_contradicts=int(rng.random() < .2), airi_confirms=int(rng.random() < .5),
            iso_confirms=int(rng.random() < .3),
            is_surrogate=0, is_artifact=0, is_ghost=0,
        ))

    cand = pd.DataFrame(crows)
    feat = pd.DataFrame(frows)
    info = {"is_fixture": True, "n_features": len(feat), "n_candidates": len(cand),
            "distinct_structures": cand["smiles"].nunique()}
    return cand, feat, info


def banner(info: dict) -> str:
    """The warning that must accompany any output derived from the fixture."""
    if not info.get("is_fixture"):
        return ""
    return (
        "> **These numbers come from an illustrative fixture, not from measurements.**\n"
        f"> {info['n_features']} invented features over {info['distinct_structures']} real "
        "migrant structures. The chemistry is real and the model's probabilities for these "
        "compounds are genuine predictions; the retention indices, evidence scores, candidate "
        "rankings and MSI levels are fabricated to exercise the code. Nothing here is a finding "
        "about any sample, and no compound is being reported as detected."
    )
