# Independent endpoint validation: pending

The project is a transparent compound-identity uncertainty demonstration. No external predictive performance or independent probability calibration is claimed.

Before a predictive-performance paper:

1. Freeze the current training identities, model configuration and calibration before accessing external labels.
2. Obtain independently sourced compounds with chemically verified structures and documented genotoxicity evidence. Record stable source identifiers, references, endpoint definitions, evidence dates and the identity verification method.
3. Determine endpoint compatibility. An Ames mutation label, a chromosome-damage endpoint and an aggregated EFSA conclusion are distinct targets. A benchmark on one alone is transfer evaluation, not validation of the EFSA-label probability. Do not silently merge them.
4. Apply the same chemical representation policy. Audit raw identity and standardized-parent duplicates, conflicting conclusions, excluded inputs and any train/external overlap. Report external identities already present in training separately and exclude them from independent performance estimates.
5. Evaluate the frozen estimator without external tuning. Report AP, ROC area, Brier, log loss and reliability; positive counts and prevalence; results by nearest-training similarity; uncertainty appropriate to the external sampling design. Sparse external positives can make calibration conclusions unavailable.
6. Archive every input, endpoint mapping decision, checksum and excluded identity with a reason. Reserve any tuning dataset separately from the final evaluation.

The 22 FCM2018 reference-standard confirmations concern chemical identity. They provide no independent genotoxicity outcomes.

JRC/EURL ECVAM maintains public Ames-oriented genotoxicity datasets, for example [official dataset instructions](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EURL-ECVAM/datasets/genotox/instructions.pdf). These are a possible source for a separately declared endpoint-transfer study, not automatically a compatible EFSA-label test. No such external performance result has been generated here.

`external_validation_template.csv` is an empty schema, not a supplied dataset. Required fields: `smiles`, `y`, `source`, `reference_id`, `endpoint`, `identity_verification`, `evidence_date`. Document the biological endpoint rather than changing it to match the model's label name. The full exploratory notebook's optional legacy CSV merge is not this independent protocol and is disabled by default.
