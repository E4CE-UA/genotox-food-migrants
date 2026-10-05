# What if the compound identity is wrong?

When analysing substances that may migrate from food packaging, identifying a detected compound is not always straightforward. An assignment may be tentative, and different molecular structures can share the same molecular formula.

This notebook explores a practical question: **how would a model’s estimate change if the assigned chemical structure were different?**

It uses published candy-wrapper assignments, chemical structures represented with RDKit, and a LightGBM classifier trained on genotoxicity conclusions from EFSA OpenFoodTox. The interface lets readers compare structures, inspect their similarity to the training data, and explore hypothetical identity scenarios.

The notebook does not identify compounds from mass spectra or determine whether packaging is safe. Its purpose is to make the assumptions behind the calculations visible.

## What the notebook shows

For each eligible structure, the model estimates the probability of receiving a positive label under the project’s aggregation of EFSA genotoxicity conclusions. This is a prediction of the recorded label, conditional on the supplied structure—not a direct measurement of genotoxicity.

The notebook also shows:

- The nearest training molecule and its structural similarity.
- Matches to ten selected structural-alert rules.
- A comparison between two independently selected molecules.
- An average of model estimates under a chosen identity scenario.
- How that average changes when a similarity cutoff excludes structures.

Structural alerts identify particular molecular patterns. A match is not proof of genotoxicity, and the absence of a match does not establish safety.

## Data sources

**EFSA OpenFoodTox** provides the genotoxicity conclusions used to construct the training labels. The reference export is version 6, including `Genotoxicity_KJ_2023.xlsx`: [DOI 10.5281/zenodo.8120114](https://doi.org/10.5281/zenodo.8120114).

The published application case contains **33 compound assignments** curated from Tables 2–3 of:

> Galmán Graíño et al. (2018). *GC-MS Screening Analysis for the Identification of Potential Migrants in Plastic and Paper-Based Candy Wrappers*. Polymers, 10(7), 802.

[Read the source study](https://doi.org/10.3390/polym10070802).

These are reported assignments, not 33 raw chromatograms or independent toxicity measurements. The study’s chromatogram figure provides analytical context; the notebook’s model estimates are new calculations.

Alternative structures are sampled from listed PubChem structures with the same molecular formula. Their inclusion does not establish that they were present in the analysed packaging.

## Example: a tentative oleic-acid assignment

The default example starts with the study’s tentative oleic-acid assignment. Its structure-specific model estimate is approximately **0.91%**.

The notebook then considers a retained set of **20 structures**: the assigned molecule and 19 sampled alternatives with the same molecular formula.

| Scenario | Minimum similarity | Included weight | Scenario average |
|---|---:|---:|---:|
| Assigned structure only | 0.40 | 100% | 0.91% |
| Equal weights across 20 structures | 0.40 | 70% | 2.25% |
| Equal weights across 20 structures | 0.50 | 60% | 2.40% |
| Equal weights across 20 structures | 0.90 | 0% | Unavailable |

In the equal-weight scenario, 14 of the 20 structures meet the 0.40 similarity cutoff. Raising the cutoff to 0.50 leaves 12 included structures. The model remains fixed; the average changes because a different subset contributes to it.

Two denominators must be distinguished:

- **14/20** describes inclusion within the displayed scenario.
- **20/2,484** describes the retained set relative to the listed PubChem formula pool.

Neither quantity measures confidence in the chemical identification. The PubChem pool is also not an exhaustive list of every chemically possible identity.

The alternatives have no individual supporting spectra, retention indices or library-search scores in this dataset. They are used to explore sensitivity to structural assumptions. Equal weighting is hypothetical, and the resulting average is not the probability that the detected chromatographic signal is genotoxic.

When no weighted structure meets the cutoff, an average cannot be calculated. The interface therefore displays **Unavailable**, and the JSON export records `null`.

## Using the controls

Choose a published entry, then select **Molecule A** and **Molecule B** to compare retained structures. Each molecule has its own model estimate, nearest training neighbour and structural-alert results.

The identity-scenario control changes the weights used in the average. Selecting a molecule for inspection or comparison does not change those weights.

The similarity cutoff controls which structures contribute to the calculation. Similarity is measured using fingerprint-based Tanimoto similarity: higher values indicate more similar encoded structural features. The cutoff is an exploratory setting, not a validated confidence boundary.

For assignments confirmed with reference standards in the source study, the notebook retains the assigned structure’s weight in both scenario modes. Structures already present in the training data are marked because their fitted estimates are not held-out evaluations.

## Model preparation and evaluation

Training conclusions are joined to chemical structures and aggregated by standardized parent structure. A parent receives a positive label if at least one included conclusion is positive.

Eligible structures undergo a shared preparation procedure, including charge and tautomer standardization. The model uses radius-2, 2,048-bit Morgan fingerprints. Inputs outside the supported chemical scope are excluded; exclusion does not mean a negative genotoxicity result.

The demonstration uses **1,892 standardized training parents**, including **130 positive labels**.

Internal evaluation compares LightGBM, logistic regression and a prevalence baseline. It includes raw and sigmoid-calibrated models, with separate data used to fit the classifiers, fit calibration and evaluate predictions.

Two splitting approaches are reported:

- A scaffold-based protocol that separates ring scaffolds and groups acyclic parents individually.
- A robustness protocol that also keeps closely related acyclic structures together.

The second protocol addresses a limitation of the first: related compounds without ring scaffolds may otherwise appear on both sides of a split.

Performance is assessed using ranking metrics, probability-error metrics and calibration plots. Calibration results are mixed across metrics and protocols; calibration does not guarantee reliable probabilities for every queried structure.

Results are reported across five outer seeds. Their variation describes sensitivity to these internal splits and is not a confidence interval.

See [the validation report](docs/VALIDATION.md) for numerical results, split definitions and figures.

**Independent validation against compatible genotoxicity outcomes has not yet been performed.** Reference-standard confirmation of a compound’s identity in the candy-wrapper study does not validate its predicted genotoxicity label.

The proposed next steps are documented in [the external-validation protocol](docs/EXTERNAL_VALIDATION.md).

## Run the notebook

From the repository root:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/marimo edit notebooks/what_if_wrong_compound.py
```

The competition notebook is `notebooks/what_if_wrong_compound.py`. The broader exploratory analysis is available in `notebooks/genotox_marimo.py`.

## Review and reproducibility

The repository includes chemical-preparation code, label construction, model evaluation, input audits and regression tests. Outputs under `docs/` record model settings, data and code hashes, split identities, predictions and validation figures.

Source auditing checks the bundled EFSA export against the reference release. Historical enrichment logs were not retained in full; the documentation distinguishes verified file contents from unavailable historical records.

Useful starting points:

| Location | Contents |
|---|---|
| `notebooks/what_if_wrong_compound.py` | Interactive identity-sensitivity study |
| `notebooks/genotox_marimo.py` | Broader exploratory analysis |
| `src/` | Chemical preparation, labels, models and notebook support |
| `scripts/` | Validation, auditing and documentation generation |
| `docs/` | Methods, validation outputs and provenance |
| `tests/` | Regression tests for calculations and supported inputs |

## AI assistance

Claude Opus 5 and GPT-6.1 Sol assisted with code review, regression tests and (some) presentation text. RDKit renders the molecular structures, and the notebook’s calculations produce the displayed model results.
