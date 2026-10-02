# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "genotox-food-migrants[notebook] @ git+https://github.com/E4CE-UA/genotox-food-migrants",
#     "marimo",
#     "anywidget>=0.9",
# ]
# ///

import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full", app_title="What if the chemical identity is wrong?")


@app.cell(hide_code=True)
def _():
    import sys
    from pathlib import Path
    import marimo as mo

    # Use the existing repository when the file replaces notebooks/what_if_wrong_compound.py.
    _roots = [Path.cwd(), Path.cwd().parent]
    _notebook_dir = mo.notebook_dir()
    if _notebook_dir is not None:
        _roots.insert(0, Path(_notebook_dir).parent)
    # Repo checkout when src/ sits beside the notebook; on molab src/ is not
    # there and the pip-installed package is used instead.
    _root = next((p for p in _roots if (p / "src" / "public_features.py").exists()), None)
    if _root is not None:
        if str(_root) in sys.path:
            sys.path.remove(str(_root))
        sys.path.insert(0, str(_root))
    try:
        notebook_file = Path(__file__).resolve()
    except NameError:
        notebook_file = Path(mo.notebook_location() or Path.cwd())
    return mo, notebook_file


@app.cell(hide_code=True)
def _(notebook_file):
    # All demo-specific computation is included here; no new src modules are required.
    def _build_demo(notebook_source_file):
        """Recompute the competition figure from bundled data and the existing model.

        No synthetic signals, inferred spectral scores, or toxicity decisions are made
        here. Formula alternatives are an explicit sensitivity scenario. The three UI
        parameters change a view over a fixed fitted model; they never retrain it.
        """

        from functools import lru_cache
        import hashlib
        import importlib.metadata
        import json

        import numpy as np
        import pandas as pd
        from rdkit import Chem, rdBase, RDLogger
        from rdkit.Chem import rdMolDescriptors

        from src import chemistry, data, labels, models, paths, public_features

        STUDY = "FCM2018"  # Repository-local identifier, not a MetaboLights accession.
        HOOK_FEATURE = "FCM2018-0010"
        NIAS_NAMES = {
            "FCM2018-0010": "Irganox 1010 degradation product",
            "FCM2018-0004": "BHT degradation product",
        }


        def file_sha256(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()


        @lru_cache(maxsize=1)
        def prepare():
            """Return candidates, assignment metadata, and reproducibility information."""
            with rdBase.BlockLogs():
                return prepare_tables()


        def prepare_tables():
            if not (paths.PUBLIC_CANDIDATES.exists() and paths.PUBLIC_FEATURES.exists()):
                raise FileNotFoundError("The bundled public candidate and feature tables are required.")
            bundle = pd.read_parquet(paths.PUBLIC_CANDIDATES, columns=["study", "n_requested"])
            study_bundle = bundle.loc[bundle["study"].eq(STUDY)]
            bundled_features = pd.read_parquet(paths.PUBLIC_FEATURES, columns=["study"])
            if (study_bundle.empty or study_bundle["n_requested"].min() < 20
                    or not bundled_features["study"].eq(STUDY).any()):
                raise ValueError("The local bundle must contain FCM2018 and 20 retained structures per assignment.")
            candidates, features, info = public_features.build(STUDY, n_candidates=20)
            # The shared alert helper re-enables RDKit logging. Keep display output
            # quiet; parse and standardization failures remain recorded in tables.
            RDLogger.DisableLog("rdApp.*")
            if not info.get("from_bundle") or candidates.empty or features.empty:
                raise ValueError("FCM2018 must be present in the bundled tables; no demo fallback is used.")

            # Read the assigned names from the project table, including names
            # that name_id intentionally omits for unconfirmed assignments.
            maf = public_features.load_maf(STUDY)
            if maf["database_identifier"].duplicated().any():
                raise ValueError("Source identifiers must uniquely match the curated assignment rows.")
            metadata = maf.set_index("database_identifier")
            features = features.copy()
            for target, source in (
                ("assigned_name", "metabolite_identification"),
                ("assigned_smiles", "smiles"),
                ("assigned_uri", "uri"),
                ("material", "material"),
            ):
                features[target] = features["database_identifier"].map(metadata[source])
            if features["assigned_smiles"].isna().any():
                raise ValueError("An assignment is missing from the curated source table.")
            features["display_name"] = features["feature_id"].map(NIAS_NAMES).fillna(features["assigned_name"])
            features["is_nias"] = features["feature_id"].isin(NIAS_NAMES)
            features["evidence_group"] = np.where(
                features["msi_level"].eq(1), "Standard-confirmed", "Library-only"
            )

            # Check the actual retained structures, rather than assuming a PubChem
            # query returned the advertised formula. Rank remains enumeration order.
            formula = candidates["smiles"].map(
                lambda s: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s))
                if isinstance(s, str) and Chem.MolFromSmiles(s) is not None else None
            )
            expected = candidates["feature_id"].map(features.set_index("feature_id")["chemical_formula"])
            if formula.isna().any() or formula.ne(expected).any():
                raise ValueError("A retained structure is invalid or does not share the source formula.")
            assigned_counts = candidates["is_assigned"].fillna(False).groupby(candidates["feature_id"]).sum()
            if not assigned_counts.eq(1).all():
                raise ValueError("Each entry must retain exactly one assigned structure.")

            # Preserve the existing label rule, molecular representation, model, and
            # seed. Apply the SAME standardization to training and candidate structures.
            rdkit_version = importlib.metadata.version("rdkit")
            standardization_cache = paths.CACHE / f"demo_standardization_rdkit_{rdkit_version}.parquet"
            with rdBase.BlockLogs():
                efsa_records = data.load_efsa()
                aggregated = labels.aggregate_by_substance(
                    data.with_structure_only(efsa_records),
                    key="inchikey", rule="any_positive", ambiguous="exclude",
                )
                standardized = chemistry.standardize_table(aggregated, cache_path=standardization_cache)
                candidate_std = chemistry.standardize_table(
                    candidates[["smiles"]], cache_path=standardization_cache
                )
            training = (
                standardized[standardized["y"].notna() & standardized["ok"].fillna(False)]
                .drop_duplicates("inchikey_std").reset_index(drop=True)
            )
            label_conflicts = int(
                standardized.loc[standardized["ok"].fillna(False)].groupby("inchikey_std")["y"]
                .nunique().gt(1).sum()
            )
            y = training["y"].astype(int).to_numpy()
            train_fp = chemistry.fingerprints(training["smiles_std"], radius=2, n_bits=2048)
            classifier = models.build("lgbm", 0).set_params(n_jobs=2)
            classifier.fit(chemistry.matrix(train_fp, n_bits=2048), y)
            query_fp = chemistry.fingerprints(candidate_std["smiles_std"], radius=2, n_bits=2048)
            valid = [i for i, fp in enumerate(query_fp) if fp is not None]
            score = np.full(len(candidates), np.nan)
            similarity = np.full(len(candidates), np.nan)
            nearest_indices = np.full(len(candidates), -1, dtype=int)
            if valid:
                score[valid] = classifier.predict_proba(
                    chemistry.matrix([query_fp[i] for i in valid], n_bits=2048)
                )[:, 1]
                similarity[valid], nearest_indices[valid] = chemistry.nearest_neighbor(
                    [query_fp[i] for i in valid], train_fp
                )
            scored = candidates.copy()
            scored["score"] = score
            scored["tanimoto_neighbour"] = similarity
            scored["smiles_std"] = candidate_std["smiles_std"].to_numpy()
            scored["inchikey_std"] = candidate_std["inchikey_std"].to_numpy()
            scored["in_training"] = scored["inchikey_std"].isin(set(training["inchikey_std"]))
            scored["nn_smiles"] = [training.iloc[j]["smiles_std"] if j >= 0 else "" for j in nearest_indices]
            scored["nn_name"] = [str(training.iloc[j]["name"]) if j >= 0 else "Unavailable" for j in nearest_indices]
            scored["nn_inchikey"] = [training.iloc[j]["inchikey_std"] if j >= 0 else None for j in nearest_indices]
            assigned = scored[scored["is_assigned"].fillna(False)].set_index("feature_id")
            for col in ("score", "tanimoto_neighbour", "in_training", "nn_name"):
                features[col] = features["feature_id"].map(assigned[col])

            source_maf = next((paths.DATA / "metabolights").glob("FCM2018__*.tsv"))
            provenance = {
                "study_id": STUDY,
                "study_id_kind": "repository-local curated assignment table",
                "collection_description": "Original project demonstration combining real data from multiple sources.",
                "sources": {
                    "training_labels": "bundled EFSA OpenFoodTox table",
                    "molecular_structures": "retained PubChem structures and assigned structure URIs",
                    "assignments": "bundled FCM2018 input table",
                },
                "assignment_table_reference": {
                    "doi": "10.3390/polym10070802",
                    "tables": [2, 3],
                    "scope": "Reference for the bundled FCM2018 assignment table, not a publication of this project or its model results.",
                    "documented_in": "src/public_features.py; repository's original FCM2018 notebook",
                    "url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6403844/",
                },
                "n_efsa_records": len(efsa_records),
                "n_efsa_resolved_identities": int(data.with_structure_only(efsa_records)["inchikey"].nunique()),
                "n_efsa_structured_records": int(efsa_records["has_structure"].sum()),
                "n_assignments": len(features),
                "n_standard_confirmed": int(features["msi_level"].eq(1).sum()),
                "n_training_substances": len(training),
                "n_training_positive": int(y.sum()),
                "n_standardized_label_conflicts": label_conflicts,
                "label_rule": "any_positive; ambiguous excluded; deduplicated standardized InChIKey",
                "model": "LightGBM; src.models.build('lgbm', 0)",
                "model_parameters": classifier.get_params(),
                "fingerprint": {"name": "ECFP4", "radius": 2, "bits": 2048},
                "standardization": "Cleanup > FragmentParent > Uncharger > canonical tautomer",
                "candidate_scope": "retained same-formula structures, not spectrum-supported candidates",
                "versions": {name: importlib.metadata.version(name) for name in (
                    "rdkit", "lightgbm", "numpy", "pandas", "scikit-learn", "marimo"
                )},
                "sha256": {
                    "efsa_estructuras.csv": file_sha256(paths.EFSA_CSV),
                    "public_candidates.parquet": file_sha256(paths.PUBLIC_CANDIDATES),
                    "public_features.parquet": file_sha256(paths.PUBLIC_FEATURES),
                    source_maf.name: file_sha256(source_maf),
                    "what_if_wrong_compound.py": file_sha256(notebook_source_file),
                    "models.py": file_sha256(paths.ROOT / "src" / "models.py"),
                    "chemistry.py": file_sha256(paths.ROOT / "src" / "chemistry.py"),
                    "labels.py": file_sha256(paths.ROOT / "src" / "labels.py"),
                },
            }
            return scored, features, provenance


        def evaluate(scored, features, feature_id, cutoff, assumption):
            """Evaluate identity WEIGHT with an explicit residual, never a safety call."""
            if not 0 <= cutoff <= 1:
                raise ValueError("Tanimoto cutoff must lie in [0, 1].")
            if assumption not in {"published", "same_formula"}:
                raise ValueError("Unknown identity assumption.")
            feature = features.loc[features["feature_id"].eq(feature_id)].iloc[0]
            subset = scored.loc[scored["feature_id"].eq(feature_id)].copy().reset_index(drop=True)
            confirmed = bool(feature["msi_level"] == 1)
            usable = subset["score"].notna() & subset["tanimoto_neighbour"].ge(cutoff)
            published_weight = subset["is_assigned"].fillna(False).astype(float)
            if confirmed:
                alternative_weight = published_weight.copy()
            else:
                alternative_weight = pd.Series(np.full(len(subset), 1.0 / len(subset)), index=subset.index)
            weight = published_weight if assumption == "published" else alternative_weight
            if not np.isclose(weight.sum(), 1.0):
                raise ValueError("Identity weights must sum to one.")
            covered = float(weight[usable].sum())
            weighted_sum = float((weight[usable] * subset.loc[usable, "score"]).sum())
            assigned = subset.loc[subset["is_assigned"].fillna(False)].iloc[0]
            subset["in_domain"] = usable
            subset["weight"] = weight
            return {
                "feature": feature,
                "candidates": subset,
                "assigned": assigned,
                "cutoff": float(cutoff),
                "assumption": assumption,
                "confirmed": confirmed,
                "published_covered_weight": float(published_weight[usable].sum()),
                "alternative_covered_weight": float(alternative_weight[usable].sum()),
                "covered_weight": covered,
                "uncovered_weight": max(0.0, 1.0 - covered),
                "n_retained": len(subset),
                "n_evaluable": int(usable.sum()),
                # Zero sum when nothing is evaluable is never exposed as a low score.
                "covered_score_sum": weighted_sum,
                "conditional_score": weighted_sum / covered if covered > 0 else None,
                "assigned_evaluable": bool(pd.notna(assigned["score"]) and assigned["tanimoto_neighbour"] >= cutoff),
                "n_assigned_evaluable": int((features["score"].notna() & features["tanimoto_neighbour"].ge(cutoff)).sum()),
            }


        def inspect_structure(view, selection_key=None):
            """Return the actual candidate requested by a molecule click.

            Inspection never changes chemical evidence or scenario weights.
            A selection from another feature, or assigned-only mode, falls back
            to the current feature's assigned structure.
            """
            candidates = view["candidates"]
            keys = candidates["rank"].map(lambda rank: f"{view['feature']['feature_id']}:{int(rank)}")
            requested = candidates.loc[keys.eq(selection_key)] if selection_key is not None else candidates.iloc[0:0]
            if view["assumption"] == "published" or requested.empty:
                selected = candidates.loc[candidates["is_assigned"].fillna(False)].iloc[0]
            else:
                selected = requested.iloc[0]
            return {
                "selected": selected,
                "selection_key": f"{view['feature']['feature_id']}:{int(selected['rank'])}",
                "selected_evaluable": bool(pd.notna(selected["score"]) and selected["tanimoto_neighbour"] >= view["cutoff"]),
            }


        def snapshot(view, provenance, inspected=None):
            """JSON export of the actual setting, structures, scores, and provenance."""
            inspected = inspect_structure(view) if inspected is None else inspected
            selected = inspected["selected"]
            cols = ["rank", "cid", "is_assigned", "inchikey", "smiles", "smiles_std",
                    "tanimoto_neighbour", "nn_name", "nn_inchikey", "score", "in_training",
                    "in_domain", "weight"]
            # pandas converts absent CIDs/invalid scores to JSON null, not NaN.
            rows = json.loads(view["candidates"][cols].to_json(orient="records", double_precision=15))
            payload = {
                "provenance": provenance,
                "setting": {"feature_id": view["feature"]["feature_id"], "cutoff": view["cutoff"],
                            "identity_assumption": "assigned_identity" if view["assumption"] == "published" else "other_same_formula_structures"},
                "result": {key: view[key] for key in (
                    "covered_weight", "uncovered_weight", "n_retained", "n_evaluable",
                    "conditional_score", "assigned_evaluable", "n_assigned_evaluable"
                )},
                "inspected_structure": {
                    "selection_key": inspected["selection_key"],
                    "rank": int(selected["rank"]),
                    "cid": int(selected["cid"]) if pd.notna(selected["cid"]) else None,
                    "smiles": selected["smiles"],
                    "smiles_std": selected["smiles_std"],
                    "nn_smiles": selected["nn_smiles"],
                    "nn_name": selected["nn_name"],
                    "nn_inchikey": selected["nn_inchikey"],
                    "similarity": float(selected["tanimoto_neighbour"]) if pd.notna(selected["tanimoto_neighbour"]) else None,
                    "classifier_score": float(selected["score"]) if pd.notna(selected["score"]) else None,
                    "meets_selected_domain_criterion": inspected["selected_evaluable"],
                },
                "interpretation": {
                    "score": "uncalibrated classifier output, not a probability or safety verdict",
                    "cutoff": "exploratory similarity criterion; not an empirically validated reliability threshold",
                    "weight": "scenario weight, not measured identity confidence",
                    "uncovered": "excluded identity weight has no assigned hazard score",
                    "standard_confirmation": "preserved in both identity scenarios",
                },
                "structural_alerts": selected_alerts(selected),
                "candidates": rows,
            }
            return json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")

        def selected_alerts(row):
            from src import candidate_view, chemistry
            valid = pd.notna(row['score'])
            smiles = row['smiles_std'] if valid else row['smiles']
            parsed = isinstance(smiles,str) and Chem.MolFromSmiles(smiles) is not None
            hits = candidate_view.alert_hits(smiles) if parsed else []
            return {
                'representation': 'standardized model input' if valid else 'retained raw structure',
                'smiles': smiles, 'parsed': bool(parsed),
                'rule_count': len(chemistry.DNA_REACTIVE_ALERTS),
                'matched_names': [h[0] for h in hits] if parsed else None,
                'matched_atoms': sorted({a for h in hits for a in h[1]}),
            }

        from types import SimpleNamespace
        return SimpleNamespace(
            STUDY=STUDY, HOOK_FEATURE=HOOK_FEATURE, prepare=prepare,
            evaluate=evaluate, inspect_structure=inspect_structure, snapshot=snapshot, selected_alerts=selected_alerts,
        )

    demo = _build_demo(notebook_file)
    return (demo,)


@app.cell(hide_code=True)
def _():
    # Guidance and real molecular drawings are included in this file.
    def _build_demo_view():
        """Guided presentation for the real-data competition figure."""

        from functools import lru_cache
        from html import escape
        from io import StringIO
        import xml.etree.ElementTree as ET
        import numpy as np
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D

        PAPER = "#faf8f4"
        INK = "#253b43"
        PETROL = "#126b78"
        VIOLET = "#6550a8"
        AMBER = "#aa6916"
        MUTED = "#61737b"
        LINE = "#dce2df"

        CSS = """
        <style>
        :root { --background:#faf8f4; --color-background:#faf8f4; }
        body, .marimo, #root { background:#faf8f4; }
        .gx { color:#253b43; max-width:1120px; margin-inline:auto; font-family:ui-sans-serif,system-ui,-apple-system,sans-serif; line-height:1.5; }
        .gx * { box-sizing:border-box; }
        .gx h1,.gx h2,.gx p { margin:0; }
        .gx h1 { font:400 clamp(36px,4.4vw,57px)/1.06 Georgia,"Iowan Old Style",serif; letter-spacing:-1.5px; margin:13px 0 15px; max-width:880px; }
        .gx h1 em { color:#6550a8; font-style:normal; }
        .gx h2 { font:400 29px/1.15 Georgia,"Iowan Old Style",serif; letter-spacing:-.45px; }
        .gx-kicker,.gx-label,.gx-step-name { font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace; text-transform:uppercase; letter-spacing:.07em; color:#61737b; }
        .gx-kicker { color:#126b78; border-top:3px solid #126b78; padding-top:14px; }
        .gx-mono { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
        .gx-small { font-size:12px; color:#61737b; line-height:1.5; }
        .gx-intro { background:#eeeafb; border-left:5px solid #6550a8; border-radius:0 9px 9px 0; padding:17px 21px; }
        .gx-intro p { font-size:15px; max-width:940px; }
        .gx-intro p+p { margin-top:8px; }
        .gx-intro strong { color:#513d90; }
        .gx-task { display:flex; flex-wrap:wrap; align-items:baseline; gap:9px; margin-top:13px; font-size:14px; }
        .gx-task b { color:#126b78; }
        .gx-data-scale { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:9px; margin:18px 0 14px; }
        .gx-data-scale > div { background:#fffdf9; border:1px solid #dce2df; border-top:3px solid #126b78; border-radius:6px; padding:11px 13px; }
        .gx-data-scale > div:nth-child(3) { border-top-color:#426ca8; background:#edf3f9; }
        .gx-data-scale > div:last-child { border-top-color:#6550a8; background:#eeeafb; }
        .gx-data-scale strong { display:block; color:#126b78; font:400 29px/1.2 ui-monospace,monospace; margin:3px 0 4px; }
        .gx-data-scale > div:last-child strong { color:#6550a8; }
        .gx-case-intro { color:#40555c; background:#f0ede5; border-radius:6px; padding:11px 14px; font-size:13px; }
        .gx-case-intro b { color:#6550a8; }
        .gx-route { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; margin:18px 0 5px; }
        .gx-route-item { padding:11px 13px; border:1px solid #dce2df; border-radius:6px; font-size:12px; }
        .gx-route-item b { display:block; color:#126b78; font-size:13px; margin-bottom:3px; }
        .gx-route-item:nth-child(2) { background:#fff3db; border-color:#e8cfa3; }
        .gx-route-item:nth-child(2) b { color:#915811; }
        .gx-route-item:nth-child(3) { background:#eeeafb; border-color:#d9cfee; }
        .gx-route-item:nth-child(3) b { color:#6550a8; }
        .gx-section { border-top:1px solid #dce2df; padding:21px 0 9px; margin-top:17px; }
        .gx-step { display:flex; align-items:flex-start; gap:12px; }
        .gx-step-number { display:flex; align-items:center; justify-content:center; flex:0 0 32px; height:32px; border-radius:50%; background:#126b78; color:#fff; font:600 14px/1 ui-monospace,monospace; }
        .gx-step.amber .gx-step-number { background:#aa6916; }
        .gx-step.violet .gx-step-number { background:#6550a8; }
        .gx-step p { font-size:13px; color:#61737b; margin-top:7px; max-width:900px; }
        .gx-hero { display:grid; grid-template-columns:1.15fr 1.15fr .85fr; gap:12px; margin-top:9px; }
        .gx-hero > section { min-width:0; padding:16px; border:1px solid #c5ddd9; border-top:4px solid #126b78; border-radius:8px; background:#e9f4f1; }
        .gx-hero > section:nth-child(2) { background:#edf3f9; border-color:#ccdae9; border-top-color:#426ca8; }
        .gx-hero > section:last-child { background:#e9f4f1; }
        .gx-hero > section.warn { background:#fff3db; border-color:#e8cfa3; border-top-color:#aa6916; }
        .gx-hero .gx-label { color:#365762; }
        .gx-structure { height:183px; display:flex; justify-content:center; align-items:center; }
        .gx-structure svg { width:100%; max-height:183px; }
        .gx-name { font-size:14px; font-weight:600; line-height:1.4; min-height:40px; }
        .gx-evidence { display:inline-block; padding:3px 7px; font:10px/1.4 ui-monospace,monospace; background:#ffffff8c; border:1px solid #b7d7d0; color:#126b78; margin:8px 5px 7px 0; border-radius:3px; }
        .gx-evidence.warn { color:#915811; border-color:#dfbf86; background:#fff7e8; }
        .gx-tc { display:flex; align-items:baseline; gap:10px; margin:8px 0 3px; }
        .gx-tc b { color:#126b78; font:400 35px/1.1 ui-monospace,monospace; letter-spacing:-1.4px; }
        .gx-status { display:inline-block; padding:6px 9px; border-radius:4px; font:600 19px/1.2 ui-monospace,monospace; margin:13px 0 10px; color:#fff; background:#126b78; }
        .gx-status.warn { background:#aa6916; }
        .gx-score { font:400 25px/1.2 ui-monospace,monospace; margin:7px 0!important; }
        .gx-output { margin-top:22px; padding-top:13px; border-top:1px solid #00000017; }
        .gx-note { border-left:4px solid #126b78; background:#e9f4f1; padding:12px 15px; margin:12px 0 2px; border-radius:0 5px 5px 0; font-size:14px; }
        .gx-note.warn { background:#fff3db; border-color:#aa6916; }
        .gx-note.violet { background:#eeeafb; border-color:#6550a8; }
        .gx-note strong { display:block; margin-bottom:3px; }
        .gx-inline { display:flex; align-items:center; justify-content:space-between; gap:20px; }
        .gx-metric { font:400 27px/1.1 ui-monospace,monospace; color:#126b78; }
        .gx-coverage { display:flex; gap:3px; margin:10px 0 8px; }
        .gx-coverage span { flex:1; height:8px; border-radius:2px; background:#126b78; }
        .gx-coverage span.out { background:#dcb477; }
        .gx-guide { font-size:13px; background:#f1f0e9; padding:11px 14px; margin:9px 0 2px; border-radius:6px; color:#40555c; }
        .gx-guide strong { color:#915811; }
        .gx-guide.violet { background:#eeeafb; }
        .gx-guide.violet strong { color:#6550a8; }
        .gx-weight { display:grid; grid-template-columns:1fr 1fr; gap:24px; padding:18px 0; }
        .gx-weight-value { font:400 52px/1.07 Georgia,serif; color:#126b78; letter-spacing:-1.8px; margin:10px 0; }
        .gx-weight-value.violet { color:#6550a8; }
        .gx-weight-value.warn { color:#aa6916; }
        .gx-weight-value .gx-before { color:#61737b; font-size:32px; }
        .gx-weight-value .gx-arrow { color:#6550a8; font:24px/1 sans-serif; margin:0 8px; }
        .gx-bar { margin:7px 0 15px; }
        .gx-bar-label { display:flex; justify-content:space-between; gap:8px; font-size:12px; margin-bottom:6px; }
        .gx-track { height:12px; border:1px solid #ccd7d3; background:#edf0e9; border-radius:3px; overflow:hidden; }
        .gx-fill { height:100%; background:#126b78; transition:width .18s; }
        .gx-bar.alt .gx-fill { background:#6550a8; }
        .gx-bar.unrevealed .gx-track { background:repeating-linear-gradient(135deg,#f0ecf8,#f0ecf8 5px,#e5dff2 5px,#e5dff2 10px); }
        .gx-bar.inactive { opacity:.65; }
        .gx-bar.active .gx-bar-label { font-weight:650; }
        .gx-weight-box { background:#f2f0fa; border:1px solid #ddd5ef; border-radius:8px; padding:14px 17px; }
        .gx-residual { font-size:12px; color:#915811; margin-top:10px!important; }
        .gx-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:9px; margin-top:15px; }
        .gx-card { background:#e9f4f1; border:1px solid #bcd9d1; padding:9px 10px; border-radius:6px; min-width:0; }
        .gx-card.out { background:#fff7e8; border-color:#dfc498; border-style:dashed; }
        .gx-card.assigned { border:2px solid #6550a8; padding:8px 9px; }
        .gx-card-head { display:flex; justify-content:space-between; gap:4px; font:10px/1.3 ui-monospace,monospace; color:#61737b; }
        .gx-card-mol { height:109px; display:flex; align-items:center; justify-content:center; }
        .gx-card-mol svg { width:100%; max-height:109px; }
        .gx-card-score { display:flex; justify-content:space-between; gap:5px; font:11px/1.4 ui-monospace,monospace; }
        .gx-card-state { color:#126b78; font-weight:700; }
        .gx-card.out .gx-card-state { color:#915811; }
        .gx-card a { color:#546770; text-decoration:underline; font-size:10px; }
        .gx-plot svg { width:100%; height:auto; display:block; }
        .gx-plot { padding:10px 0 0; }
        .gx-feature { margin:14px auto; padding:14px 17px; background:#eeeafb; border-left:4px solid #6550a8; border-radius:0 7px 7px 0; }
        .gx-feature-top { display:flex; justify-content:space-between; align-items:baseline; gap:14px; flex-wrap:wrap; }
        .gx-feature h2 { font-size:23px; }
        .gx-feature-meta { display:flex; gap:24px; flex-wrap:wrap; margin-top:10px; font-size:12px; }
        .gx-feature-meta b { display:block; color:#253b43; font:16px/1.4 ui-monospace,monospace; }
        .gx-workspace { display:grid; grid-template-columns:minmax(0,1.06fr) minmax(0,1fr); gap:20px; align-items:start; margin:16px auto; }
        .gx-choices { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:8px; margin-top:12px; }
        .gx-choices.single { grid-template-columns:minmax(150px,240px); }
        .gx-choice { min-width:0; }
        .gx-choice button { width:100%; height:auto!important; padding:0!important; display:block!important; white-space:normal; border:0!important; border-radius:6px; background:transparent!important; }
        .gx-choice button:hover { box-shadow:0 0 0 2px #6550a8; }
        .gx-choice button:focus-visible { outline:3px solid #6550a8; outline-offset:2px; }
        .gx-choice .prose { max-width:none; margin:0; }
        .gx-choice .prose p { margin:0; }
        .gx-choice-card { background:#e9f4f1; border:1px solid #bddbd4; padding:9px 8px; border-radius:6px; text-align:left; color:#253b43; }
        .gx-choice-card.out { background:#fff7e8; border-color:#dfc498; border-style:dashed; }
        .gx-choice-card.selected { border:2px solid #6550a8; padding:8px 7px; box-shadow:0 0 0 1px #6550a8 inset; }
        .gx-choice-card svg { display:block; width:100%; height:98px; }
        .gx-choice-id { font:10px/1.4 ui-monospace,monospace; overflow-wrap:anywhere; }
        .gx-choice-values { display:flex; justify-content:space-between; gap:5px; margin-top:4px; font:11px/1.4 ui-monospace,monospace; }
        .gx-choice-action { margin-top:5px; font-size:10px; color:#6550a8; }
        .gx-inspector { position:sticky; top:18px; padding:17px; background:#fffdf9; border:1px solid #dce2df; border-top:4px solid #6550a8; border-radius:8px; }
        .gx-inspector h2 { font-size:23px; margin:5px 0; }
        .gx-inspector-pair { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:13px 0; }
        .gx-inspector-pair > div { min-width:0; }
        .gx-inspector-pair .gx-name { font-size:12px; overflow-wrap:anywhere; }
        .gx-inspector-pair .gx-structure { height:145px; }
        .gx-inspector-pair svg { max-height:145px; }
        .gx-results { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
        .gx-results > div { min-width:0; padding:12px; border:1px solid #dce2df; border-radius:6px; background:#edf3f9; }
        .gx-results > div:first-child { background:#e9f4f1; }
        .gx-result-value { font:400 25px/1.2 ui-monospace,monospace; color:#126b78; margin:5px 0!important; overflow-wrap:anywhere; }
        .gx-inspector .gx-status { font-size:13px; margin:8px 0 6px; }
        .gx-input-note { font-size:12px; color:#61737b; margin-top:8px!important; }
        .gx-plot-window { position:relative; width:100%; isolation:isolate; z-index:1; }
        .gx-plot-window > svg { width:100%; height:auto; display:block; pointer-events:none; }
        .gx-plot-hit { position:absolute; transform:translate(-50%,-50%); width:26px; height:26px; z-index:50; overflow:hidden; opacity:0; pointer-events:auto; }
        .gx-plot-hit:focus-within { opacity:.5; outline:2px solid #6550a8; border-radius:50%; }
        .gx-plot-hit button { width:26px!important; min-width:0!important; height:26px!important; padding:0!important; border:0!important; border-radius:50%; opacity:0; pointer-events:auto; }
        .gx-plot-hit button:focus-visible { opacity:.4; background:#6550a8; outline:2px solid #6550a8; }
        .gx-point { cursor:help; }
        .gx-point:hover use,.gx-point:focus use { stroke-width:2.2; }
        .gx-control-title { font-size:14px; font-weight:650; color:#126b78; margin-top:6px!important; }
        .gx-rule-readout { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:8px 0; font:14px/1.5 ui-monospace,monospace; }
        .gx-rule-readout b { color:#126b78; }
        .gx-rule-readout.warn b { color:#aa6916; }
        .gx-finding { border-left:4px solid #aa6916; background:#fff3db; padding:13px 16px; font-size:15px; line-height:1.5; margin:10px 0 12px!important; }
        .gx-close { padding:25px 24px; border-radius:9px; margin-top:24px; background:#193e48; color:#fff; }
        .gx-close h2 { max-width:900px; }
        .gx-close p { max-width:880px; margin-top:12px; font-size:14px; color:#e2efed; }
        .gx-close .gx-label { color:#a4d6cb; margin-bottom:10px; }
        .gx-routes { display:grid; grid-template-columns:1fr 1fr; gap:14px; margin:18px auto; }
        .gx-route-col { padding:16px 18px; border-radius:8px; min-width:0; }
        .gx-route-col.trad { background:#f1f0e9; border:1px solid #ddd9c9; border-top:4px solid #aa6916; }
        .gx-route-col.ours { background:#eeeafb; border:1px solid #d9cfee; border-top:4px solid #6550a8; }
        .gx-route-col h3 { font:400 20px/1.2 Georgia,serif; margin:6px 0 3px; }
        .gx-route-col .gx-label { color:#915811; }
        .gx-route-col.ours .gx-label { color:#6550a8; }
        .gx-route-step { padding:9px 0; border-top:1px solid #0000001a; font-size:13px; }
        .gx-route-step:first-of-type { border-top:0; }
        .gx-route-step b { display:block; font:10px/1.5 ui-monospace,monospace; text-transform:uppercase; letter-spacing:.06em; color:#61737b; margin-bottom:2px; }
        .gx-verdict { margin-top:11px; padding:11px 13px; border-radius:6px; background:#fff; font-size:14px; }
        .gx-verdict strong { display:block; margin-bottom:3px; }
        .gx-route-col.trad .gx-verdict strong { color:#915811; }
        .gx-route-col.ours .gx-verdict strong { color:#6550a8; }
        @media(max-width:800px) { .gx-routes { grid-template-columns:1fr; } }
        .gx-conclusion { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin-top:20px; }
        .gx-conclusion div { border-top:2px solid #77c3b7; padding-top:10px; font-size:12px; color:#e2efed; }
        .gx-conclusion div:nth-child(2) { border-color:#b5a2e0; }
        .gx-conclusion div:nth-child(3) { border-color:#e4b875; }
        .gx-conclusion b { display:block; color:#fff; margin-bottom:4px; font-size:13px; }
        .gx-details p { margin:9px 0; font-size:13px; }
        .gx-details a { color:#126b78; text-decoration:underline; }
        .gx-details h3 { font-size:14px; margin-top:18px; }
        @media(max-width:800px) {
         .gx-workspace { grid-template-columns:1fr; }
         .gx-inspector { position:static; grid-row:1; }
         .gx-hero { grid-template-columns:1fr 1fr; }
         .gx-hero > section:last-child { grid-column:1/-1; }
         .gx-grid { grid-template-columns:repeat(4,minmax(0,1fr)); }
         .gx-data-scale { grid-template-columns:repeat(2,minmax(0,1fr)); }
        }
        @media(max-width:540px) {
         .gx-choices { grid-template-columns:repeat(2,minmax(0,1fr)); }
         .gx h1 { font-size:38px; }
         .gx-route,.gx-hero,.gx-weight,.gx-conclusion { grid-template-columns:1fr; }
         .gx-hero > section:last-child { grid-column:auto; }
         .gx-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
         .gx-inline { flex-wrap:wrap; gap:9px; }
         .gx-weight-value { font-size:46px; }
         .gx-intro,.gx-close { padding:16px; }
        }

        .gx-selection-banner {display:grid;grid-template-columns:105px minmax(0,1fr) auto;gap:16px;align-items:center;padding:12px 17px;border:1px solid #d9cfee;border-left:4px solid #6550a8;border-radius:7px;background:#fffdf9;margin:15px auto;}
        .gx-selection-banner strong {display:block;font:400 23px/1.2 Georgia,serif;margin:5px 0;overflow-wrap:anywhere;}
        .gx-selection-thumb {width:105px;height:78px;flex:0 0 105px;}
        .gx-selection-thumb svg {display:block;width:100%;height:100%;}
        .gx-selection-numbers {display:grid;gap:5px;font:12px/1.5 ui-monospace,monospace;color:#6550a8;}
        .gx-reference-molecule {height:130px;background:#fffdf98c;border-radius:5px;margin:12px 0;}
        .gx-reference-molecule svg {display:block;width:100%;height:100%;}
        .gx-comparison-values {display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:10px 0;font-size:11px;color:#61737b;}
        .gx-comparison-values b {display:block;font:20px/1.3 ui-monospace,monospace;color:#253b43;overflow-wrap:anywhere;}
        .gx-identity-transition {display:flex;flex-wrap:wrap;gap:18px;align-items:center;margin:14px 0 10px;font-size:12px;}
        .gx-identity-transition b {display:block;font:29px/1.2 ui-monospace,monospace;color:#6550a8;}
        .gx-identity-map {display:flex;flex-wrap:wrap;gap:5px;margin:12px 0 8px;}
        .gx-identity-tile {padding:4px 7px;min-width:37px;text-align:center;border:1px solid #bddbd4;background:#e9f4f1;color:#126b78;border-radius:3px;font:11px/1.4 ui-monospace,monospace;}
        .gx-identity-tile.out {background:#fff3db;color:#915811;border-color:#e8cfa3;border-style:dashed;}
        .gx-identity-tile.no-output {background:#eeefef;color:#61737b;border-color:#bfc8cb;border-style:dotted;}
        .gx-identity-tile.zero-weight {opacity:.45;}
        .gx-identity-tile.selected {outline:2px solid #6550a8;outline-offset:1px;opacity:1;}
        .gx-set-result {font-size:13px;margin:12px 0 9px!important;}
        .gx-grain-note {padding-top:8px;border-top:1px solid #d9cfee;}
        .gx-motif-heading {display:flex;gap:15px;align-items:center;margin:7px 0;}
        .gx-sensitivity-panels {display:grid;grid-template-columns:1fr 1fr;gap:18px;}
        .gx-sensitivity-panels > section {min-width:0;border:1px solid #dce2df;border-radius:8px;padding:14px;background:#fffdf9;}
        .gx-sensitivity-panels > section:first-child {border-top:3px solid #6550a8;}
        .gx-sensitivity-panels > section:last-child {border-top:3px solid #126b78;}
        .gx-sensitivity-panels h3 {font:400 20px/1.25 Georgia,serif;margin:5px 0 7px;overflow-wrap:anywhere;}
        .gx-sensitivity-panels svg {display:block;width:100%;height:auto;margin-top:9px;}
        @media(max-width:800px){.gx-sensitivity-panels {grid-template-columns:1fr;}.gx-selection-banner {grid-template-columns:85px minmax(0,1fr);gap:10px;}.gx-selection-thumb {width:85px;flex-basis:85px;}.gx-selection-numbers {grid-column:2;font-size:11px;}.gx-selection-banner strong {font-size:20px;}}
        </style>
        """


        def escaped(value):
            return escape(str(value), quote=True)


        def fmt_score(value):
            if value is None or not np.isfinite(float(value)):
                return "Unavailable"
            return "0" if float(value) == 0 else f"{float(value):.3g}"


        @lru_cache(maxsize=1800)
        def structure_svg(smiles, width=320, height=190, highlights=()):
            molecule = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
            if molecule is None:
                return '<span class="gx-small">Structure unavailable</span>'
            drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
            options = drawer.drawOptions()
            options.clearBackground = False
            options.useDefaultAtomPalette()
            options.bondLineWidth = 1.6
            options.padding = .09
            rdMolDraw2D.PrepareAndDrawMolecule(drawer, molecule, highlightAtoms=list(highlights), highlightAtomColors={atom:(0.94,0.76,0.44) for atom in highlights})
            drawer.FinishDrawing()
            svg = drawer.GetDrawingText()
            return svg[svg.find("<svg"):]


        def header(features, provenance):
            return f'''<div class="gx">
              <div class="gx-kicker">Food-contact migrants · Genotoxicity screening · EFSA OpenFoodTox</div>
              <h1>What if the chemical <em>identity</em> is wrong?</h1>
              <div class="gx-intro"><p>A GC–MS assignment proposes a molecular identity. The genotoxicity model evaluates the structure supplied with that identity.</p>
                <p><strong>Select an entry, then select a structure.</strong> Follow its nearest EFSA training structure, genotoxicity score and matched structural motifs. Move the similarity criterion to see which results your screening rule includes.</p></div>
              <div class="gx-data-scale">
                <div><div class="gx-label">EFSA records</div><strong>{provenance['n_efsa_records']:,}</strong><span class="gx-small">Substance–output records</span></div>
                <div><div class="gx-label">Resolved identities</div><strong>{provenance['n_efsa_resolved_identities']:,}</strong><span class="gx-small">Distinct raw InChIKeys</span></div>
                <div><div class="gx-label">Model reference</div><strong>{provenance['n_training_substances']:,}</strong><span class="gx-small">Standardized EFSA training structures</span></div>
                <div><div class="gx-label">This identity experiment</div><strong>{len(features)}</strong><span class="gx-small">Reported GC–MS assignments</span></div>
              </div></div>'''


        def feature_context(view):
            f = view["feature"]
            evidence = "Confirmed with a standard · recorded MSI 1" if view["confirmed"] else "Library assignment · recorded MSI 2"
            return f'''<div class="gx gx-feature">
              <div class="gx-feature-top"><div><div class="gx-label">GC–MS entry · reported assignment</div><h2>{escaped(f['feature_id'])}</h2></div>
                <div><b>{escaped(f['display_name'])}</b><div class="gx-small">Assigned identity · {evidence}</div></div></div>
              <div class="gx-feature-meta">
                <span>Reported retention time<b>{f['rt_mean']:.2f} min</b></span>
                <span>Reported fragment ion<b>m/z {f['mass_to_charge']:.0f}</b></span>
                <span>Formula of assigned structure<b>{escaped(f['chemical_formula'])}</b></span>
                <span>Assigned identifier<b>{escaped(f['database_identifier'])}</b></span>
              </div>
              <p class="gx-input-note">One entry in the 33-assignment table; its FCM2018 identifier is a project label. Retention time and m/z describe the reported observation. The selected molecular structure is what the model evaluates.</p>
            </div>'''


        def structure_name(row, feature):
            if bool(row["is_assigned"]):
                return str(feature["display_name"])
            name = row.get("name")
            if isinstance(name, str) and name.strip():
                return name
            cid = row.get("cid")
            return f"PubChem CID {int(cid)}" if cid is not None and cid == cid else f"Retained structure #{int(row['rank'])}"


        def structure_uri(row, feature):
            cid = row.get("cid")
            return f"https://pubchem.ncbi.nlm.nih.gov/compound/{int(cid)}" if cid is not None and cid == cid else str(feature["assigned_uri"])


        def choice_label(row, view, selected_key, control_html=""):
            key = f"{view['feature']['feature_id']}:{int(row['rank'])}"
            inside = bool(row["in_domain"])
            cid = row.get("cid")
            tag = "Assigned identity" if bool(row["is_assigned"]) else f"CID {int(cid)}"
            classes = "gx-choice-card" + (" out" if not inside else "") + (" selected" if key == selected_key else "")
            has_output = np.isfinite(float(row['score'])) and np.isfinite(float(row['tanimoto_neighbour']))
            tc_text = f"{row['tanimoto_neighbour']:.3f}" if has_output else "—"
            state = ('IN' if inside else 'OUT') if has_output else 'NO OUTPUT'
            smiles = row['smiles_std'] if has_output else row['smiles']
            return f'''<div class="{classes}"><div class="gx-choice-id">#{int(row['rank']):02d} · {escaped(tag)}</div>
              {structure_svg(smiles,200,135)}
              <div class="gx-choice-values"><span>Tc {tc_text}</span><b>{state}</b></div>
              <div class="gx-choice-id">Score {fmt_score(row['score'])}</div>
              <div class="gx-choice-action">{control_html or ('Selected · results shown' if key == selected_key else 'Inspect this structure')}</div>
            </div>'''


        def selection_banner(view, inspected):
            row = inspected['selected']
            f = view['feature']
            has_output = np.isfinite(float(row['score']))
            smiles = row['smiles_std'] if has_output else row['smiles']
            tc = f"{row['tanimoto_neighbour']:.3f}" if has_output else 'Unavailable'
            state = 'IN' if inspected['selected_evaluable'] else ('OUT' if has_output else 'NO OUTPUT')
            return f'''<div class="gx gx-selection-banner" data-selection-key="{inspected['selection_key']}" aria-live="polite">
              <div class="gx-selection-thumb">{structure_svg(smiles,150,95)}</div>
              <div><div class="gx-label">Currently inspected · {escaped(f['feature_id'])} · structure #{int(row['rank']):02d}</div>
                <strong>{escaped(structure_name(row,f))}</strong><p class="gx-small">{'Assigned identity' if bool(row['is_assigned']) else 'Same-formula alternative'} · all individual-result panels below follow this molecule.</p></div>
              <div class="gx-selection-numbers"><b>Tc {tc} · {state}</b><span>Score {fmt_score(row['score'])}</span></div></div>'''


        def choices_intro(view):
            f = view['feature']
            if view['assumption'] == 'published':
                title = 'Assigned molecular identity'
                text = 'Open Explore same-formula structures to inspect the other retained molecules.'
            else:
                title = f"{view['n_retained']} molecular structures · {escaped(f['chemical_formula'])}"
                text = 'Click a molecule. Its drawing, nearest EFSA neighbour, score and structural motifs update in the selected-result panels.'
            return f'''<div><div class="gx-label">Molecules for {escaped(f['feature_id'])}</div>
              <h2 style="font-size:24px;margin:5px 0">{title}</h2><p class="gx-small">{text}</p></div>'''


        def inspector(view, inspected):
            row = inspected["selected"]
            f = view["feature"]
            name = structure_name(row, f)
            accepted = inspected["selected_evaluable"]
            tc = float(row["tanimoto_neighbour"])
            has_output = np.isfinite(float(row['score'])) and np.isfinite(tc)
            source = "Assigned identity" if bool(row["is_assigned"]) else "Other retained same-formula structure"
            state = "IN SELECTED DOMAIN" if accepted else "OUTSIDE SELECTED DOMAIN"
            op = "≥" if accepted else "<"
            membership = '<p class="gx-small">This standardized identity is present in the EFSA training set.</p>' if bool(row["in_training"]) else ""
            if not has_output:
                state = "MODEL INPUT COULD NOT BE READ"
                interpretation = "The standardized SMILES could not be parsed for fingerprinting. The retained raw structure is shown; this pipeline produced no score or similarity for it."
            elif accepted:
                interpretation = "The selected structure meets your similarity criterion. Its score contributes only when this identity scenario gives the structure weight."
            else:
                interpretation = "The score is calculated for this structure. Screening interpretation is withheld under your current similarity criterion."
            tc_text = f"{tc:.3f}" if has_output else "Unavailable"
            comparison = f"Tc {tc:.6f} {op} {view['cutoff']:.2f}" if has_output else "Fingerprint unavailable for standardized model input"
            model_label = "Model input" if has_output else "Retained raw structure"
            model_smiles = row['smiles_std'] if has_output else row['smiles']
            neighbour = (structure_svg(row['nn_smiles'],300,200) if has_output
                         else '<span class="gx-small">No fingerprint to compare</span>')
            return f'''<aside id="current-molecular-result" class="gx-inspector" data-selection-key="{inspected['selection_key']}" aria-live="polite">
              <div class="gx-label">Selected molecular result · {escaped(f['feature_id'])} · #{int(row['rank']):02d}</div>
              <h2>{escaped(name)}</h2><p class="gx-small">{source}</p>
              <div class="gx-inspector-pair">
                <div><div class="gx-label">{model_label}</div><div class="gx-structure">{structure_svg(model_smiles,300,200,alert_details(row)["atoms"])}</div>
                  <a class="gx-small" href="{escaped(structure_uri(row,f))}" target="_blank" rel="noopener noreferrer">Open molecular record in PubChem</a></div>
                <div><div class="gx-label">Closest EFSA training structure</div><div class="gx-structure">{neighbour}</div>
                  <div class="gx-name">{escaped(row['nn_name'])}</div></div>
              </div>
              <div class="gx-results">
                <div><div class="gx-label">Genotoxicity model score</div><p class="gx-result-value">{fmt_score(row['score'])}</p>
                  <p class="gx-small">0–1 classifier output: higher means a stronger positive-genotoxicity signal. Uncalibrated.</p></div>
                <div><div class="gx-label">Nearest-structure similarity</div><p class="gx-result-value">{tc_text}</p>
                  <p class="gx-small">Tanimoto · ECFP4<br>Your selected minimum: {view['cutoff']:.2f}</p></div>
              </div>
              <div class="gx-status{'' if accepted else ' warn'}">{state}</div>
              <p class="gx-small gx-mono">{comparison}</p>
              <p style="font-size:13px;margin-top:8px">{interpretation}</p>{membership}
              <p class="gx-input-note">Selecting a molecule changes the structure, its nearest neighbour and its score. Moving the Tc criterion changes applicability while those calculations stay fixed.</p>
            </aside>'''


        def two_routes(view, inspected):
            f, assigned, chosen = view['feature'], view['assigned'], inspected['selected']
            def molecule(row):
                return structure_svg(row['smiles_std'] if np.isfinite(float(row['score'])) else row['smiles'], 340, 150)
            def similarity(row):
                return f"{row['tanimoto_neighbour']:.3f}" if np.isfinite(float(row['tanimoto_neighbour'])) else 'Unavailable'
            baseline_state = 'IN' if view['assigned_evaluable'] else 'OUT'
            selection_state = 'IN' if inspected['selected_evaluable'] else 'OUT'
            if not np.isfinite(float(chosen['score'])):
                selection_state = 'NO MODEL OUTPUT'
            relation = 'The assigned identity is currently selected.' if bool(chosen['is_assigned']) else 'Same formula; a different molecular structure.'
            return f'''<div class="gx gx-routes" data-selection-key="{inspected['selection_key']}" data-feature-id="{escaped(f['feature_id'])}">
              <section class="gx-route-col trad" data-reference-key="{escaped(f['feature_id'])}:{int(assigned['rank'])}">
                <div class="gx-label">Recorded assignment · reference · {escaped(f['feature_id'])}</div>
                <h3>{escaped(f['display_name'])}</h3><div class="gx-reference-molecule">{molecule(assigned)}</div>
                <p class="gx-small">{'Standard-confirmed · MSI 1' if view['confirmed'] else 'Library assignment · MSI 2'}</p>
                <div class="gx-comparison-values"><span>Similarity Tc<b>{similarity(assigned)}</b></span><span>Model score<b>{fmt_score(assigned['score'])}</b></span></div>
                <div class="gx-verdict"><strong>{baseline_state} at Tc ≥ {view['cutoff']:.2f}</strong>Reference identity from this GC–MS entry. Choose another entry to change it.</div></section>
              <section class="gx-route-col ours" data-selection-key="{inspected['selection_key']}">
                <div class="gx-label">Selected molecule · {escaped(f['feature_id'])} · #{int(chosen['rank']):02d}</div>
                <h3>{escaped(structure_name(chosen, f))}</h3><div class="gx-reference-molecule">{molecule(chosen)}</div>
                <p class="gx-small">{relation}</p>
                <div class="gx-comparison-values"><span>Similarity Tc<b>{similarity(chosen)}</b></span><span>Model score<b>{fmt_score(chosen['score'])}</b></span></div>
                <div class="gx-verdict"><strong>{selection_state} at Tc ≥ {view['cutoff']:.2f}</strong>Individual molecular result. Click another structure to change it.</div></section>
            </div>'''


        def weight_summary(view, inspected):
            f, row = view['feature'], inspected['selected']
            covered = view['covered_weight']
            if view['confirmed']:
                scope = 'Confirmed identity'
                scenario = '100% of identity weight stays on the standard-confirmed assignment. Other molecules remain available for inspection.'
            elif view['assumption'] == 'same_formula':
                scope = f"All {view['n_retained']} structures · equal weights"
                scenario = f"Each retained structure receives {100/view['n_retained']:.0f}% in this hypothetical identity scenario."
            else:
                scope = 'Assigned identity only'
                scenario = '100% of identity weight is on the assigned structure.'
            tiles = []
            for _, candidate in view['candidates'].iterrows():
                rank = int(candidate['rank'])
                key = f"{f['feature_id']}:{rank}"
                selected = key == inspected['selection_key']
                classes = 'gx-identity-tile' + (' no-output' if not np.isfinite(float(candidate['score'])) else (' in' if candidate['in_domain'] else ' out'))
                if float(candidate['weight']) == 0:
                    classes += ' zero-weight'
                if selected:
                    classes += ' selected'
                tc = f"{candidate['tanimoto_neighbour']:.3f}" if np.isfinite(float(candidate['tanimoto_neighbour'])) else 'unavailable'
                title = f"Structure #{rank:02d} · Tc {tc} · scenario weight {100*float(candidate['weight']):.0f}%"
                tiles.append(f'<span class="{classes}" data-key="{key}" title="{escaped(title)}">#{rank:02d}</span>')
            transition = ''
            if not view['confirmed'] and view['assumption'] == 'same_formula':
                transition = f'''<div class="gx-identity-transition"><span>Assigned identity <b>{100*view['published_covered_weight']:.0f}%</b></span><span aria-hidden="true">→</span><span>Equal-weight set <b>{100*covered:.0f}%</b></span></div>'''
            if view['assumption'] == 'same_formula' and not view['confirmed']:
                headline = f"{view['n_evaluable']} of {view['n_retained']} structures meet Tc ≥ {view['cutoff']:.2f}"
            else:
                headline = f"{100*covered:.0f}% of the active identity weight meets Tc ≥ {view['cutoff']:.2f}"
            score = fmt_score(view['conditional_score'])
            return f'''<div class="gx gx-note violet gx-weight-live" data-feature-id="{escaped(f['feature_id'])}" data-selection-key="{inspected['selection_key']}">
              <div class="gx-label">Identity set · {escaped(f['feature_id'])} · {escaped(f['display_name'])}</div>
              <h2 style="font-size:23px;margin:6px 0">{headline}</h2><p class="gx-small">{scope}. {scenario}</p>
              {transition}<div class="gx-track" style="margin-top:10px"><div class="gx-fill" style="width:{100*covered:.6f}%;background:#6550a8"></div></div>
              <div class="gx-identity-map">{''.join(tiles)}</div>
              <p class="gx-small">Violet outline: inspected molecule #{int(row['rank']):02d}. Teal: meets Tc. Amber: below Tc. Grey: no model output. Faded tiles carry no identity weight.</p>
              <p class="gx-set-result"><b>{100*covered:.0f}% included identity weight</b> · {100*view['uncovered_weight']:.0f}% not included · conditional model score {score}.</p>
              <p class="gx-small gx-grain-note">This summary uses the complete identity set. It changes with the GC–MS entry, Tc or identity assumption; inspecting one member changes the outlined tile and its individual result above.</p>
            </div>'''


        def whole_intro(features, provenance, view, inspected):
            row = inspected['selected']
            tc = f"{row['tanimoto_neighbour']:.3f}" if np.isfinite(float(row['tanimoto_neighbour'])) else 'Unavailable'
            overlay = 'The violet diamond is the inspected alternative.' if not bool(row['is_assigned']) else 'The ring marks the inspected assignment.'
            return f'''<div class="gx gx-section"><div class="gx-label">{len(features)} reported GC–MS assignments · assigned structures</div>
              <h2>Choose an identity. Explore its molecular result.</h2>
              <p class="gx-small" style="margin-top:8px"><b>{view['n_assigned_evaluable']} of {len(features)}</b> assigned structures meet Tc ≥ {view['cutoff']:.2f}.
                Each point represents one assigned structure from the input table, compared with {provenance['n_training_substances']:,} EFSA training structures.
                Click a point to select that GC–MS entry.</p>
              <p class="gx-small" style="margin-top:8px;padding:8px 11px;border-left:3px solid #6550a8;background:#eeeafb"><b>Inspected: {escaped(view['feature']['feature_id'])} · structure #{int(row['rank']):02d} · Tc {tc}.</b> {overlay} The 33 source points represent the original assignments.</p>
            </div>'''


        def closing(view, inspected):
            row = inspected['selected']
            name = structure_name(row,view['feature'])
            has_output = np.isfinite(float(row['score']))
            if not has_output:
                outcome = 'This standardized model input produced no score or similarity.'
            elif inspected['selected_evaluable']:
                outcome = f"This molecule meets your criterion Tc ≥ {view['cutoff']:.2f}."
            else:
                outcome = f"This molecule is outside your criterion Tc ≥ {view['cutoff']:.2f}; its score is withheld from screening interpretation."
            tc = f"{row['tanimoto_neighbour']:.3f}" if has_output else 'Unavailable'
            return f'''<div class="gx gx-close" data-selection-key="{inspected['selection_key']}">
              <div class="gx-label">Selected molecular result · {escaped(view['feature']['feature_id'])} · #{int(row['rank']):02d}</div>
              <h2>{escaped(name)}</h2><p>{outcome}</p><p>Model score {fmt_score(row['score'])} · similarity Tc {tc}.</p>
              <p>Identity determines the molecular input. Similarity determines whether your screening rule includes its score.</p></div>'''


        def plot_positions(features):
            positions = []
            for x, group in enumerate(("Standard-confirmed", "Library-only")):
                rows = features.loc[features["evidence_group"].eq(group)].sort_values("feature_id")
                slots = np.linspace(-.3,.3,len(rows)) if len(rows)>1 else [0.0]
                for slot, (_, row) in zip(slots, rows.iterrows()):
                    px = .075 + ((x + slot + .45) / 1.9) * (.98 - .075)
                    py = 1 - (.19 + float(row["tanimoto_neighbour"]) / 1.055 * (.94 - .19))
                    positions.append((str(row["feature_id"]), float(px), float(py), str(row["display_name"]), float(row["tanimoto_neighbour"])))
            return positions


        def figure_svg(fig):
            buf = StringIO()
            fig.savefig(buf, format="svg", facecolor=PAPER)
            plt.close(fig)
            svg = buf.getvalue()
            return svg[svg.find("<svg"):]

        def study_plot(features, view, inspected=None):
            with plt.rc_context({"font.family":"DejaVu Sans", "font.size":10,
                                 "text.color":INK, "axes.labelcolor":MUTED,
                                 "xtick.color":MUTED, "ytick.color":MUTED,
                                 "svg.fonttype":"none"}):
                fig, ax = plt.subplots(figsize=(10.8,3.3))
                fig.subplots_adjust(left=.075,right=.98,bottom=.19,top=.94)
                ax.set_facecolor(PAPER)
                ax.axhspan(0, view["cutoff"], color="#efece4", alpha=.65, zorder=0)
                point_information = []
                for x, group in enumerate(("Standard-confirmed", "Library-only")):
                    rows = features.loc[features["evidence_group"].eq(group)].sort_values("feature_id")
                    slots = np.linspace(-.3,.3,len(rows)) if len(rows)>1 else [0.0]
                    for slot, (_, row) in zip(slots, rows.iterrows()):
                        pos = x+slot
                        marker = "*" if row["is_nias"] else "o"
                        color = AMBER if row["is_nias"] else (PETROL if group == "Standard-confirmed" else VIOLET)
                        face = color if group == "Standard-confirmed" or row["is_nias"] else PAPER
                        size = 140 if row["is_nias"] else 47
                        dot = ax.scatter(pos,row["tanimoto_neighbour"],s=size,marker=marker,
                                         facecolors=face,edgecolors=color,linewidths=1.1,zorder=3)
                        point_id = "compound-" + str(row["feature_id"])
                        dot.set_gid(point_id)
                        passes = bool(row["score"] == row["score"] and row["tanimoto_neighbour"] >= view["cutoff"])
                        evidence = "Confirmed in input data" if row["msi_level"] == 1 else "Tentative identity"
                        tooltip = (f"{row['feature_id']} · {row['display_name']}\n"
                                   f"Similarity Tc: {row['tanimoto_neighbour']:.6f}\n"
                                   f"Selected minimum: {view['cutoff']:.2f} · {'PASSES' if passes else 'BELOW MINIMUM'}\n"
                                   f"{evidence}{' · NIAS: non-intentionally added substance' if row['is_nias'] else ''}\n"
                                   f"Closest training structure: {row['nn_name']}")
                        point_information.append((point_id, tooltip))
                        if row["feature_id"] == view["feature"]["feature_id"]:
                            ax.scatter(pos,row["tanimoto_neighbour"],s=size+150,
                                       facecolors="none",edgecolors=INK,linewidths=1,zorder=4)
                if inspected is not None:
                    selected_row = inspected['selected']
                    if not bool(selected_row['is_assigned']) and np.isfinite(float(selected_row['tanimoto_neighbour'])):
                        point = next(p for p in plot_positions(features) if p[0] == str(view['feature']['feature_id']))
                        x_value = (point[1]-.075)/(.98-.075)*1.9-.45
                        ax.plot([x_value,x_value], [view['assigned']['tanimoto_neighbour'],selected_row['tanimoto_neighbour']],
                                color=VIOLET,linewidth=1.1,linestyle=(0,(2,2)),zorder=4)
                        marker = ax.scatter(x_value,selected_row['tanimoto_neighbour'],s=115,marker='D',
                                            facecolor=VIOLET,edgecolor='white',linewidth=.8,zorder=5)
                        marker.set_gid('inspected-structure')
                ax.axhline(view["cutoff"],color=INK,linestyle=(0,(4,3)),linewidth=1,zorder=2)
                ax.text(1.38,view["cutoff"]+.025,f"Minimum similarity: {view['cutoff']:.2f}",ha="right",va="bottom",fontsize=9,color=INK)
                ax.set_ylim(0,1.055); ax.set_xlim(-.45,1.45)
                ax.set_yticks([0,.2,.4,.6,.8,1.0]); ax.set_ylabel("Nearest-EFSA similarity · Tc",fontsize=10)
                n1=int(features["msi_level"].eq(1).sum()); n2=len(features)-n1
                ax.set_xticks([0,1],[f"Confirmed in input data · {n1}",f"Tentative identity · {n2}"])
                ax.spines[["top","right"]].set_visible(False)
                for side in ("left","bottom"): ax.spines[side].set_color(LINE)
                ax.tick_params(axis="both",length=0,pad=8)
                ax.yaxis.grid(True,color=LINE,linewidth=.5); ax.set_axisbelow(True)
                ET.register_namespace("", "http://www.w3.org/2000/svg")
                ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
                root = ET.fromstring(figure_svg(fig))
                by_id = {element.get("id"): element for element in root.iter() if element.get("id")}
                for point_id, tooltip in point_information:
                    point = by_id[point_id]
                    point.set("class", "gx-point")
                    point.set("tabindex", "0")
                    point.set("role", "img")
                    point.set("aria-label", tooltip)
                    title = ET.Element("{http://www.w3.org/2000/svg}title")
                    title.text = tooltip
                    point.insert(0, title)
                return ET.tostring(root, encoding="unicode")

        def study_finding(features, view, inspected):
            passing = features['score'].notna() & features['tanimoto_neighbour'].ge(view['cutoff'])
            confirmed = int((passing & features['msi_level'].eq(1)).sum())
            tentative = int((passing & features['msi_level'].ne(1)).sum())
            row = inspected['selected']
            tc = f"{row['tanimoto_neighbour']:.3f}" if np.isfinite(float(row['tanimoto_neighbour'])) else 'unavailable'
            extra = 'The diamond shows your inspected alternative; the 33 original dots remain assigned structures.' if not bool(row['is_assigned']) and np.isfinite(float(row['tanimoto_neighbour'])) else 'The ring marks your selected assignment.'
            return f'''<div class="gx gx-study-readout" aria-live="polite" data-selection-key="{inspected['selection_key']}">
              <p class="gx-finding"><b>{int(passing.sum())}/{len(features)}</b> assigned structures meet Tc ≥ {view['cutoff']:.2f}: {confirmed} confirmed and {tentative} tentative identities.
                Current entry: <span class="gx-mono">{escaped(view['feature']['feature_id'])}</span> · inspected structure #{int(row['rank']):02d} · Tc {tc}.</p>
              <p class="gx-small">Filled circle: confirmed. Open circle: library assignment. Star: flagged NIAS. {extra} Horizontal spacing separates entries.</p></div>'''

        def sensitivity_plot(view, inspected):
            candidates = view['candidates']
            similarity = candidates['tanimoto_neighbour'].to_numpy()
            valid = candidates['score'].notna().to_numpy() & np.isfinite(similarity)
            assigned_weights = candidates['is_assigned'].fillna(False).astype(float).to_numpy()
            alternative_weights = assigned_weights if view['confirmed'] else np.ones(len(candidates))/len(candidates)
            boundaries = similarity[valid & (similarity >= .1) & (similarity <= .9)]
            thresholds = np.unique(np.r_[np.linspace(.1,.9,161), boundaries, np.nextafter(boundaries, np.inf)])
            thresholds = thresholds[thresholds <= .9]
            assigned_curve = np.array([float(assigned_weights[valid & (similarity >= t)].sum()) for t in thresholds])
            alternative_curve = np.array([float(alternative_weights[valid & (similarity >= t)].sum()) for t in thresholds])
            row = inspected['selected']
            tc = float(row['tanimoto_neighbour'])
            has_output = np.isfinite(tc) and np.isfinite(float(row['score']))
            name = structure_name(row,view['feature'])
            rank = int(row['rank'])
            cutoff = view['cutoff']
            with plt.rc_context({'font.family':'DejaVu Sans','font.size':9,'svg.fonttype':'none'}):
                def base_axis():
                    fig, ax = plt.subplots(figsize=(5.25,2.6))
                    fig.subplots_adjust(left=.16,right=.96,bottom=.24,top=.87)
                    ax.set_facecolor(PAPER)
                    ax.set_xlim(.1,.9); ax.set_ylim(-.06,1.08)
                    ax.set_xticks([.1,.3,.5,.7,.9])
                    ax.axvline(cutoff,color=AMBER,linewidth=1.2)
                    ax.set_xlabel('Minimum similarity · Tc',color=MUTED)
                    ax.spines[['top','right']].set_visible(False)
                    ax.spines[['left','bottom']].set_color(LINE)
                    ax.tick_params(colors=MUTED)
                    ax.yaxis.grid(True,color=LINE,linewidth=.5)
                    return fig,ax
                fig, ax = base_axis()
                if has_output:
                    selected_curve = (tc >= thresholds).astype(float)
                    ax.plot(thresholds,selected_curve,color=PETROL,linewidth=2,drawstyle='steps-post')
                    ax.scatter([cutoff],[float(inspected['selected_evaluable'])],color=AMBER,s=45,zorder=5)
                    if .1 <= tc <= .9:
                        marker = ax.axvline(tc,color=VIOLET,linewidth=1,linestyle=(0,(2,3)))
                        marker.set_gid('inspected-sensitivity')
                    note = f'Structure #{rank:02d} · Tc {tc:.3f}'
                else:
                    note = 'No model output for this structure'
                    ax.text(.5,.48,'No inclusion curve',ha='center',transform=ax.transAxes,color=MUTED)
                ax.set_yticks([0,1],['OUT','IN'])
                ax.set_ylabel('Selected molecule',color=MUTED)
                ax.text(.98,1.1,note,ha='right',transform=ax.transAxes,color=VIOLET,fontsize=9)
                left = figure_svg(fig)
                fig, ax = base_axis()
                fig.subplots_adjust(top=.74)
                ax.plot(thresholds,assigned_curve,color=PETROL,linewidth=1.7,drawstyle='steps-post',linestyle='--',label='Assigned identity')
                if not view['confirmed']:
                    ax.plot(thresholds,alternative_curve,color=VIOLET,linewidth=1.8,drawstyle='steps-post',label=f"Equal weights · {view['n_retained']} structures")
                ax.scatter([cutoff],[view['covered_weight']],color=AMBER,s=45,zorder=5)
                ax.set_yticks([0,.5,1],['0%','50%','100%'])
                ax.set_ylabel('Included identity weight',color=MUTED)
                ax.legend(loc='lower left',bbox_to_anchor=(0,1.02),frameon=False,fontsize=8)
                right = figure_svg(fig)
            state = 'IN' if inspected['selected_evaluable'] else ('OUT' if has_output else 'NO OUTPUT')
            return f'''<div class="gx-sensitivity-panels">
              <section class="gx-selected-sensitivity" data-selection-key="{inspected['selection_key']}"><div class="gx-label">One molecule · #{rank:02d}</div>
                <h3>{escaped(name)}</h3><p class="gx-small">At Tc ≥ {cutoff:.2f}: <b>{state}</b>. Click a molecule to change this curve.</p>{left}</section>
              <section class="gx-ensemble-sensitivity" data-feature-id="{escaped(view['feature']['feature_id'])}"><div class="gx-label">Whole identity set · {escaped(view['feature']['feature_id'])}</div>
                <h3>{escaped(view['feature']['display_name'])}</h3><p class="gx-small">At Tc ≥ {cutoff:.2f}: <b>{100*view['covered_weight']:.0f}% identity weight included</b>. Choose another GC–MS entry to change this set.</p>{right}</section>
            </div>'''

        @lru_cache(maxsize=1800)
        def rule_hits(smiles):
            from src import candidate_view
            if not isinstance(smiles, str) or Chem.MolFromSmiles(smiles) is None:
                return None
            return candidate_view.alert_hits(smiles)

        def alert_details(row):
            valid = np.isfinite(float(row['score']))
            smiles = row['smiles_std'] if valid else row['smiles']
            hits = rule_hits(smiles)
            return dict(
                names=[h[0] for h in hits] if hits is not None else None,
                atoms=tuple(sorted({a for h in (hits or []) for a in h[1]})),
                representation='standardized model input' if valid else 'retained raw structure',
            )

        def alert_panel(view, inspected):
            from src import chemistry
            row = inspected['selected']
            details = alert_details(row)
            names = details['names']
            motif_background = '#fff3db' if names is None or names else '#e9f4f1'
            motif_border = '#e8cfa3' if names is None or names else '#bddbd4'
            motif_color = '#915811' if names is None or names else '#126b78'
            if names is None:
                match = 'Structure could not be read for motif matching'
                explanation = 'No structural-alert result was produced for this representation.'
            elif names:
                match = ' · '.join(names)
                explanation = f"{len(names)} matched motif class{'es' if len(names)!=1 else ''}. Highlighted atoms in the inspector are the matched atoms."
            else:
                match = f"No match in the {len(chemistry.DNA_REACTIVE_ALERTS)}-rule set"
                explanation = 'This describes the listed motif rules; it is not a genotoxicity classification.'
            domain = 'meets' if inspected['selected_evaluable'] else 'does not meet'
            if not np.isfinite(float(row['score'])):
                model_text = 'The standardized input could not be read; no classifier result is available.'
            else:
                model_text = f"Score {fmt_score(row['score'])}; this structure {domain} your similarity criterion Tc ≥ {view['cutoff']:.2f}."
            return f'''<div class="gx gx-alert-live gx-section" data-selection-key="{inspected['selection_key']}" aria-live="polite">
              <div class="gx-label">Structural motifs · {escaped(view['feature']['feature_id'])} · structure #{int(row['rank']):02d}</div>
              <div class="gx-motif-heading"><div class="gx-selection-thumb">{structure_svg(row['smiles_std'] if np.isfinite(float(row['score'])) else row['smiles'],150,95,details['atoms'])}</div><h2 style="font-size:25px;margin:6px 0">{escaped(structure_name(row,view['feature']))}</h2></div>
              <div class="gx-results" style="margin-top:12px"><div style="background:{motif_background};border-color:{motif_border}"><div class="gx-label">Project SMARTS rules</div>
                <p class="gx-result-value" style="font-size:18px;color:{motif_color}">{escaped(match)}</p><p class="gx-small">{explanation}</p></div>
                <div><div class="gx-label">Independent classifier output</div><p style="font-size:14px;margin:8px 0">{model_text}</p>
                  <p class="gx-small">A motif match and a classifier score measure different things. Neither determines genotoxicity on its own.</p></div></div>
              <p class="gx-small" style="margin-top:8px">Rules applied to the {details['representation']}. Changing the selected molecule recomputes its matched motifs; changing Tc updates the model-inclusion rule.</p>
            </div>'''

        from types import SimpleNamespace
        return SimpleNamespace(
            CSS=CSS, header=header, feature_context=feature_context,
            choices_intro=choices_intro, choice_label=choice_label,
            structure_name=structure_name, inspector=inspector, selection_banner=selection_banner,
            weight_summary=weight_summary, two_routes=two_routes, whole_intro=whole_intro, closing=closing,
            study_plot=study_plot, study_finding=study_finding,
            plot_positions=plot_positions, sensitivity_plot=sensitivity_plot, alert_panel=alert_panel, alert_details=alert_details,
        )

    demo_view = _build_demo_view()
    return (demo_view,)


@app.cell(hide_code=True)
def _():
    def _build_widgets():
        """Local widget definitions embedded in the delivered Marimo notebook."""
        import anywidget
        import traitlets

        STUDY_JS = r'''
        export default { render({model, el}) {
          el.classList.add('gx-study-widget');
          const host=document.createElement('div'); host.className='gx-chart-svg';
          const hint=document.createElement('div'); hint.className='gx-chart-hint'; hint.setAttribute('aria-live','polite');
          el.append(host,hint);
          function paint() {
            const active=el.getRootNode().activeElement || document.activeElement;
            const focusId=host.contains(active) ? active.id : '';
            host.innerHTML=model.get('svg');
            host.querySelectorAll('.gx-point').forEach(node=>{
              const id=node.id.replace(/^compound-/, '');
              node.setAttribute('role','button');node.setAttribute('tabindex','0');
              node.setAttribute('aria-label','Select '+id+' · '+node.getAttribute('aria-label'));
              const title=node.querySelector('title')?.textContent || id;
              function select(){model.set('click_key',id);model.set('click_seq',(model.get('click_seq')||0)+1);model.save_changes();}
              node.addEventListener('click',select);
              node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}});
              node.addEventListener('pointerenter',()=>{hint.textContent=title.replaceAll('\n',' · ');});
              node.addEventListener('focus',()=>{hint.textContent=title.replaceAll('\n',' · ');});
            });
            if(focusId) host.querySelector('[id="'+focusId+'"]')?.focus({preventScroll:true});
            hint.textContent='Select a plotted assignment. Its molecular results appear directly below.';
          }
          model.on('change:svg',paint);paint();
          return ()=>model.off('change:svg',paint);
        }};
        '''

        GRID_JS = r'''
        export default { render({model,el}) {
          el.classList.add('gx-grid-widget');
          const summary=document.createElement('div');summary.className='gx-grid-selection';
          const label=document.createElement('span');label.setAttribute('aria-live','polite');
          const jump=document.createElement('button');jump.type='button';jump.textContent='Show selected result';
          function findResult(root){const direct=root.querySelector('#current-molecular-result');if(direct)return direct;for(const node of root.querySelectorAll('*')){if(node.shadowRoot){const found=findResult(node.shadowRoot);if(found)return found;}}return null;}
          jump.addEventListener('click',()=>findResult(document)?.scrollIntoView({block:'center',behavior:'smooth'}));
          summary.append(label,jump);
          const grid=document.createElement('div');grid.className='gx-molecule-grid';el.append(summary,grid);
          function paint(){
            grid.innerHTML='';const cards=model.get('cards')||[];
            const selected=cards.find(card=>card.key===model.get('selected_key'));
            label.textContent=selected ? 'Selected: '+selected.key.split(':')[0]+' · #'+selected.key.split(':')[1].padStart(2,'0')+' · '+selected.name : 'Select a molecule';
            grid.classList.toggle('single',cards.length===1);
            for(const card of cards){
              const button=document.createElement('button');button.type='button';button.className='gx-mol-button';
              button.dataset.key=card.key;button.setAttribute('aria-label','Inspect '+card.name);
              button.setAttribute('aria-pressed',String(card.key===model.get('selected_key')));
              button.innerHTML=card.html;
              button.addEventListener('click',()=>{model.set('click_key',card.key);model.set('click_seq',(model.get('click_seq')||0)+1);model.save_changes();});grid.append(button);
            }
          }
          model.on('change:cards',paint);model.on('change:selected_key',paint);paint();
          return ()=>{model.off('change:cards',paint);model.off('change:selected_key',paint);};
        }};
        '''

        TRACE_JS = r'''
        export default {render({model,el}){
          el.classList.add('gx-trace-widget');
          const tabs=document.createElement('div');tabs.className='gx-trace-tabs';tabs.setAttribute('role','group');tabs.setAttribute('aria-label','Source figures');
          const controls=document.createElement('div');controls.className='gx-trace-controls';
          const viewport=document.createElement('div');viewport.className='gx-trace-viewport';
          const img=document.createElement('img');img.draggable=false;viewport.append(img);
          const meter=document.createElement('span');meter.className='gx-trace-meter';
          let z=model.get('zoom')||1, px=model.get('pan_x')||0, py=model.get('pan_y')||0, timer, dragging=false, last;
          function transform(){img.style.transform=`translate(${px}px,${py}px) scale(${z})`;meter.textContent=z.toFixed(1)+'×';}
          function commit(){model.set('zoom',z);model.set('pan_x',px);model.set('pan_y',py);model.save_changes();}
          function zoom(factor){z=Math.max(1,Math.min(6,z*factor));if(z===1){px=0;py=0;}transform();clearTimeout(timer);timer=setTimeout(commit,160);}
          function action(label,fn){const b=document.createElement('button');b.type='button';b.textContent=label;b.addEventListener('click',fn);controls.append(b);}
          action('Zoom +',()=>zoom(1.35));action('Zoom −',()=>zoom(1/1.35));action('Reset view',()=>{z=1;px=0;py=0;transform();commit();});controls.append(meter);
          function paint(){
            tabs.innerHTML='';const panels=model.get('panels')||[];const layer=model.get('layer')||0;tabs.hidden=panels.length<=1;
            panels.forEach((p,i)=>{const b=document.createElement('button');b.type='button';b.textContent=p.label;b.setAttribute('aria-pressed',String(i===layer));
              b.addEventListener('click',()=>{model.set('layer',i);model.save_changes();});tabs.append(b);});
            const p=panels[layer]||panels[0];if(p){img.src='data:'+(p.mime||'image/png')+';base64,'+p.image;img.alt=p.alt||p.label;if(p.width&&p.height)viewport.style.aspectRatio=p.width+'/'+p.height;}
          }
          viewport.addEventListener('wheel',e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();zoom(e.deltaY<0?1.12:1/1.12);}},{passive:false});
          viewport.addEventListener('pointerdown',e=>{if(z<=1)return;dragging=true;last=[e.clientX,e.clientY];viewport.setPointerCapture(e.pointerId);});
          viewport.addEventListener('pointermove',e=>{if(!dragging)return;px+=e.clientX-last[0];py+=e.clientY-last[1];last=[e.clientX,e.clientY];
            const limitX=viewport.clientWidth*(z-1)/2,limitY=viewport.clientHeight*(z-1)/2;px=Math.max(-limitX,Math.min(limitX,px));py=Math.max(-limitY,Math.min(limitY,py));transform();});
          viewport.addEventListener('pointerup',()=>{if(dragging){dragging=false;commit();}});
          el.append(tabs,controls,viewport);model.on('change:layer',paint);paint();transform();
          return ()=>{clearTimeout(timer);model.off('change:layer',paint);};
        }};
        '''

        STUDY_CSS = '''
        .gx-study-widget {max-width:1120px;margin:auto;color:#253b43;}
        .gx-chart-svg svg {width:100%;height:auto;display:block;pointer-events:none;}
        .gx-chart-svg .gx-point {cursor:pointer;pointer-events:bounding-box;}
        .gx-chart-svg .gx-point use {pointer-events:all;}
        .gx-chart-svg .gx-point:focus use,.gx-chart-svg .gx-point:hover use {stroke-width:2.5px;}
        .gx-chart-hint {font:12px/1.5 system-ui;color:#61737b;min-height:36px;padding:6px 0;}
        '''
        GRID_CSS = '''
        .gx-grid-selection {display:flex;align-items:center;justify-content:space-between;gap:12px;flex-wrap:wrap;background:#eeeafb;border:1px solid #d9cfee;border-radius:5px;padding:9px;margin:0 0 11px;color:#6550a8;font:11px/1.5 ui-monospace,monospace;}
        .gx-grid-selection button {cursor:pointer;border:1px solid #6550a8;background:#fffdf9;color:#6550a8;border-radius:4px;padding:5px 8px;font:11px/1.4 system-ui;}
        .gx-molecule-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;}
        .gx-molecule-grid.single {grid-template-columns:minmax(150px,240px);}
        .gx-mol-button {background:transparent;border:0;padding:0;cursor:pointer;text-align:left;min-width:0;color:#253b43;font-family:system-ui;}
        .gx-mol-button:hover {box-shadow:0 0 0 1px #253b4355;border-radius:6px;}
        .gx-mol-button:focus-visible {outline:2px dotted #253b43;outline-offset:2px;border-radius:6px;}
        .gx-choice-card {height:100%;box-sizing:border-box;background:#e9f4f1;border:1px solid #bddbd4;padding:9px 8px;border-radius:6px;}
        .gx-choice-card.out {background:#fff7e8;border-color:#dfc498;border-style:dashed;}
        .gx-choice-card.selected {border:2px solid #6550a8;padding:8px 7px;box-shadow:0 0 0 1px #6550a8 inset;}
        .gx-choice-card > .gx-choice-id:first-child {min-height:28px;}
        .gx-choice-id {font:10px/1.4 ui-monospace,monospace;overflow-wrap:anywhere;}
        .gx-choice-card svg {display:block;width:100%;height:98px;}
        .gx-choice-values {display:flex;justify-content:space-between;gap:5px;font:11px/1.4 ui-monospace,monospace;margin-top:4px;}
        .gx-choice-action {font:11px/1.5 system-ui;color:#6550a8;margin-top:8px;padding-top:5px;border-top:1px solid #00000014;}
        @media(max-width:540px){.gx-molecule-grid {grid-template-columns:repeat(2,minmax(0,1fr));}}
        '''
        TRACE_CSS = '''
        .gx-trace-widget {max-width:1120px;margin:auto;font:13px/1.5 system-ui;color:#253b43;}
        .gx-trace-tabs[hidden] {display:none!important;}
        .gx-trace-tabs,.gx-trace-controls {display:flex;flex-wrap:wrap;gap:7px;margin:9px 0;align-items:center;}
        .gx-trace-widget button {border:1px solid #dce2df;border-radius:5px;background:#fffdf9;color:#253b43;padding:7px 11px;cursor:pointer;font:12px/1.4 system-ui;}
        .gx-trace-tabs button[aria-pressed=true] {background:#126b78;color:white;border-color:#126b78;}
        .gx-trace-widget button:focus-visible {outline:3px solid #6550a8;}
        .gx-trace-viewport {position:relative;aspect-ratio:980/552;overflow:hidden;border:1px solid #dce2df;border-radius:7px;background:white;touch-action:pan-y;}
        .gx-trace-viewport img {width:100%;height:100%;object-fit:contain;transform-origin:center;user-select:none;}
        .gx-trace-meter {font:12px/1.4 ui-monospace,monospace;color:#126b78;}
        '''


        class StudyChart(anywidget.AnyWidget):
            _esm = STUDY_JS
            _css = STUDY_CSS
            svg = traitlets.Unicode('').tag(sync=True)
            click_key = traitlets.Unicode('').tag(sync=True)
            click_seq = traitlets.Int(0).tag(sync=True)


        class MoleculeGrid(anywidget.AnyWidget):
            _esm = GRID_JS
            _css = GRID_CSS
            cards = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
            selected_key = traitlets.Unicode('').tag(sync=True)
            click_key = traitlets.Unicode('').tag(sync=True)
            click_seq = traitlets.Int(0).tag(sync=True)


        class TraceViewer(anywidget.AnyWidget):
            _esm = TRACE_JS
            _css = TRACE_CSS
            panels = traitlets.List(traitlets.Dict(), default_value=[]).tag(sync=True)
            layer = traitlets.Int(0).tag(sync=True)
            zoom = traitlets.Float(1).tag(sync=True)
            pan_x = traitlets.Float(0).tag(sync=True)
            pan_y = traitlets.Float(0).tag(sync=True)
        from types import SimpleNamespace
        return SimpleNamespace(StudyChart=StudyChart, MoleculeGrid=MoleculeGrid, TraceViewer=TraceViewer)
    widgets = _build_widgets()
    return (widgets,)




@app.cell(hide_code=True)
def _(demo_view, mo):
    mo.Html(demo_view.CSS + '<style>\n.gx-live-sensitivity{margin:16px auto}.gx-live-sensitivity svg{display:block;width:100%;height:auto}\n.gx-workspace{grid-template-columns:minmax(0,1.08fr) minmax(0,1fr)}\n.gx-route-col h3{overflow-wrap:anywhere}\n@media(max-width:800px){.gx-workspace{grid-template-columns:1fr}}\n</style>')
    return


@app.cell(hide_code=True)
def _(demo, mo):
    with mo.status.spinner(title="Preparing the fixed EFSA model…", subtitle="Using bundled data. The first standardization takes longer; moving the controls does not retrain the model."):
        scored, features, provenance = demo.prepare()
    return features, provenance, scored


@app.cell(hide_code=True)
def _(demo_view, features, mo, provenance):
    mo.Html(demo_view.header(features, provenance))
    return


@app.cell(hide_code=True)
def _():
    def _build_detection():
        from types import SimpleNamespace
        PANELS = [{'label': 'Figure 1 · M1.1, M3, P4 and P3.1', 'image': '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oOARw5XHopMACZAACZAACZAACZAACZAACZAACbgToODhzoVbSYAESIAESIAESIAESIAESIAESIAEYpgABY8YrjwWnQRIgARIgARIgARIgARIgARIgARIwJ0ABQ93LtxKAiRAAiRAAiRAAiRAAiRAAiRAAiQQwwQoeMRw5bHoJEACJEACJEACJEACJEACJEACJEAC7gQoeLhz4VYSIAESIAESIAESIAESIAESIAESIIEYJkDBI4Yrj0UnARIgARIgARIgARIgARIgARIgARJwJ0DBw50Lt5IACZAACZAACZAACZAACZAACZAACcQwAQoeMVx5LDoJkAAJkAAJkAAJkAAJkAAJkAAJkIA7AQoe7ly4lQRIgARIgARIgARIgARIgARIgARIIIYJUPCI4cpj0UmABEiABEiABEiABEiABEiABEiABNwJUPBw58KtJEACJEACJEACJEACJEACJEACJEACMUyAgkcMVx6LTgIkQAIkQAIkQAIkQAIkQAIkQAIk4E6Agoc7F24lARIgARIgARIgARIgARIgARIgARKIYQIUPGK48lh0EiABEiABEiABEiABEiABEiABEiABdwIUPNy5cCsJkAAJkAAJkAAJkAAJkAAJkAAJkEAME6DgEcOVx6KTAAmQAAmQAAmQAAmQAAmQAAmQAAm4E6Dg4c6FW0mABEiABEiABEiABEiABEiABEiABGKYAAWPGK48Fp0ESIAESIAESIAESIAESIAESIAESMCdAAUPdy7cSgIkQAIkQAIkQAIkQAIkQAIkQAIkEMMEKHjEcOWx6CRAAiRAAiRAAiRAAiRAAiRAAiRAAu4EKHi4c+FWEiABEiABEiABEiABEiABEiABEiCBGCZAwSOGK49FJwESIAESIAESIAESIAESIAESIAEScCdAwcOdC7eSAAmQAAmQAAmQAAmQAAmQAAmQAAnEMAEKHjFceSw6CZAACZAACZAACZAACZAACZAACZCAOwEKHu5cuJUESIAESIAESIAESIAESIAESIAESCCGCVDwiOHKY9FJgARIgARIgARIgARIgARIgARIgATcCVDwcOfCrSRAAiRAAiRAAiRAAiRAAiRAAiRAAjFMgIJHDFcei04CJEACJEACJEACaUFg8uTJMn36dClQoID069dPqlataheje/fuEh8fL+3atbO3LVq0SAYPHmz/xgq25cmTR29zS+MTmT9IgARIgARIIAwCFDzCgMYkJEACJEACJEACJOBVAj/99JPUr19fvvzyS9mxY4c88cQTsnnzZo1j3rx50rp1axk/frx069bNRnTmzBn5/fff9e8pU6bIt99+qwUPbAiUxk7MFRIgARIgARIIkwAFjzDBMRkJkAAJkAAJkAAJeJHAsWPHtHhRpkwZ2b17t1SrVk2OHDkiBw8elPvuu0/+9a9/ScWKFX0ED8Np//792vrj66+/lsKFC4eUxqTlkgRIgARIgASSSoCCR1KJMT4JkAAJkAAJkAAJkIAWOerVqyddunSRHj16yL333it9+/aVZcuWSYkSJVwFD+zPlCmTvPTSS2JZVkhpiJoESIAESIAEwiVAwSNcckxHAiRAAiRAAiRAAh4lcOjQIalbt6506NBBixxz5syR/v37a/EDgsdVV10lI0eOlLJly9qELly4ICVLltRDYeLi4iSUNHZirpAACZAACZBAGAQoeIQBjUlIgARIgARIgARIwKsETp8+LZUqVdJCR+fOnTWG7777TtavX6/XFyxYIHnz5pXhw4dLoUKFtCVHtmzZZNOmTdrqY8OGDYmmKV68uFfR8rxJgARIgAQiTICCR4SBMjsSIAESIAESIAESSM8Exo0bpx2VFixY0D7NAwcOSObMmfXvgQMH6iEtjzzyiLz22muyZ88emThxosycOVOWL18uU6dOtdOZFWcas41LEiABEiABEkguAc8IHsePH5c6derImjVrJEeOHHLy5EnBFGhbtmyRtm3bCl60CHPnztUmmLly5RJ4ES9dunRyGTM9CZAACZAACZAACZAACZAACZAACZBAKhPwhOCxceNGPcYUnsQhdOTMmVN69+4tMK987LHHpEGDBvLGG2/Iddddpz2Lr1y5UpYuXarFjyVLlqRylfBwJEACJEACJEACJEACJEACJEACJEACySXgCcED87sXK1ZMWrVqJTt37tSCx5133qlNK0uVKqXHmMJj+M0336y3zZgxQy5fviz58uXT2zD9GsaiOgPmjx82bJg8/vjjzs1cJwESIAESIAESIAES+IcApp+tWbOmnD17lkxIgARIgARIINUJeELwMFThBGvXrl1a8MD6tm3btJCBcaU//PCDlCtXTm8bO3asTgJHWxinCkuQPHnymGz08qGHHpJXX31VmjZt6rOdP0iABEiABEiA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UPAIEy6HtIQJjslIgARIgARIgAQ8QwBCB6akLVCggOzatcsz580TJQESIAESiA4CFDzCrAcKHmGCYzISIAESIAESIAHPEIDgAV9plStXlmeeecYz580TJQESIAESiA4CFDzCrAcKHmGCYzISIAESIAESIAHPEKDg4Zmq5omSAAmQQFQSoOARYrW0adNGbrzxRj17C5K88cYbMm7cOGncuHGIOTAaCZAACZAACZAACXiLAAUPb9U3z5YESIAEoo0ABY8QawTOScuXL28LHjNnztSiR7NmzULMgdFIgARIgARIgARIwFsEKHh4q755tiRAAiQQbQQoeIRZIxzSEiY4JiMBEiABEiABEvAMAQoenqlqnigJkAAJRCUBCh5hVgsFjzDBMRkJkAAJkAAJkIBnCFDw8ExV80RJgARIICoJUPAIs1ooeIQJjslIgARIgARIgAQ8Q4CCh2eqmidKAiRAAlFJgIJHiNVSokQJiYuLk6xZs+oU33zzjUyYMEHatWsXYg6MRgIkQAIkQAIkQALeIkDBw1v1zbMlARIggWgjQMEjxBoZP368XHvttbbg8dhjj8mYMWOkXr16IebAaCRAAiRAAiRAAiTgLQIUPLxV3zxbEiABEog2AhQ8wqwRDmkJExyTkQAJkAAJkAAJeIYABQ/PVDVPlARIgASikoBnBY/Tp08LrDbmzp0rd911l4waNUoyZMigf48cOVJy5colU6ZMkdKlS7tWHAUPVyzcSAIkQAIkQAIkQAI2AQoeNgqukAAJkAAJpAEBzwoeAwYMkPPnz8uwYcOkffv20rVrV6lYsaJAyFi5cqUsXbpUix9LlixxrRYKHq5YuJEESIAESIAESIAEbAIUPGwUXCEBEiABEkgDAp4VPG666SYZPny47N27V5o0aSJwSrpo0SKZOXOmzJgxQy5fviz58uWT3r17y7lz56RgwYI+1QOLkHHjxknDhg19tvMHCZAACZAACZAACZDA3wQoePBKIAESIAESSEsCnhU88uTJI+XLlxe8iCdNmiQrVqyQtWvXyrZt22Ts2LG6TgoVKiR169aVixcvSoECBXzqaf78+fL6669L8+bNfbbzBwmQAAmQAAmQAAmQwN8EKHjwSiABEiABEkhLAp4VPEqWLKmtOeLj42XIkCHafwcEkMWLF2vfHagUWHUcPnzYtX44pMUVCzeSAAmQAAmQ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NNeS1fdSBBQBRaDpIKCER5Fl1a5dO9OhQ4cs4TFx4kRz3XXXmf3337/IGPQxRUARUARqBwElPGqnLDQnikBzRUAJj+ZasvpeioAioAg0HQSU8CixrHRKS4nAaTBFQBGoCQR6977ETJs204wadUpN5EczoQgoAs0PASU8ml+Z6hspAoqAItDUEFDCo8QSU8KjROA0mCKgCNQEAosscrEZPXqqmTHjnJrIj2ZCEVAEmh8CSng0vzLVN1IEFAFFoKkhMMcSHsOGDTNnnHGG+fTTT82WW25pzj77bNOyZUtz//33m0svvdSt13HDDTeYJZZYIlimSngEYdGLioAi0EQQWG21q8znn/9ifv75ZNOlS7smkmvNpiKgCDQlBJoi4VFXd5rZY4/lzZ137tqUoNa8KgKKgCKgCCQgMMcSHqy9seKKK5oDDzzQ7Ljjju645pprGoiMIUOGmOeff96RH//5z3+C0CnhEYRFLyoCikATQWD11a8WsmOKOeaYteXvH00k15pNRUARaEoIlIPwSDJQnXDCCea1115zcCy++OJCUNyZhWbMmDHmxBNPNJ988onZddddzXHHHefuDR482Nx2221m3nnnNSeddJJZa621smE4GTduqunR40Kz114rmVtv3Snnnv5QBBQBRUARaJoIzLGEx8cff2yWWmopw2Kku+22m1l77bWdN8ddd91l7rjjDjN79mwz99xzm+WWW85MmDDBdO3aNaeE6UQHDRpk9txzz5zr+kMRUAQUgaaAAITHqFFTzIYb9pI2b5emkGXNoyKgCDQxBMpBeIQMVNtvv71ZcsklzcMPPyweal1M69atzQILLJBF56yzzpLpejPM6aefbjBm3XPPPaZjx45ms802M2+++abz7j3qqKPMf//732wYTl577Xuzzz4PmfXXX8zcdFO/nHv6QxFQBBQBRaBpIjDHEh6+uG688UZz0UUXmQ8++MA88MADBiLk3//+t7vds2dPtxMLpEic8BgwYIC5/PLLzSabbOKj0qMioAgoAk0GAQiPnj07mZ13Xt707796k8m3ZlQRUASaDgLlIDxCBqr+/fubxRZbzJx22mmmffv2zvjUqVOnLDDodZAZkBoYtZ5++mkz//zzm7Fjxzpj15dffukMXb/88ks2DCeDB78tniIfyjPzKOGRg4z+UAQUAUWg6SIwRxMeV111lbn22msN01YWXnhh8+CDD5rnnnvOsHYH0qNHD4NbZEh0SksIFb2mCCgCTQUBJTyaSklpPhWBpotAOQgP//ZRA9UPP/zgSA6mtbz88stm+PDh5tlnn/WPmhdffNFNV0a3gxBBz+vWrZu7D8mx6aabmv32288cdthh2TCc9Ox5vmnVqoXZfPMllfDIQUZ/KAKKgCLQdBGYYwkPSA3+YP3nmWceV4JfffWV6du3r/P2+PDDD83hhx9uhg4dGixdJTyCsOhFRUARaCIIKOHRRApKs6kINGEEykV4xA1UUUj++OMPtybH999/b7p37+5urbHGGuaSSy4xG2ywgTn00EPNIoss4tb0+Pnnn51n7l577WUgS+Jy1FFPmC++GGusNWIA2y9+W38rAoqAIqAINEEE5ljCg/U56CQ7dOjgig23RxawuuCCC8zdd99tpk6d6tbyWGedddx9PD2Y2lJXV+d+b7zxxmbgwIHG32+CZa9ZVgQUgTkYASU85uDC11dXBKqEQDkIj5CBinU42GnvhRdeMF9//bVbm+O7774z06ZNM23atHG/zzvvPLd+x9FHH+2mvxxwwAFm9dVXd0THvvvuG0QAwmPIkO/NiBGTzC+/nBZ8Ri8qAoqAIqAINC0E5ljCI62Y6DDbtm1rWrRokX2MDpRFsfy13377za3v8c9//jP7jJ4oAoqAItBUEFDCo6mUlOZTEWi6CJSD8EgyUO29995u8dGRI0eaf/3rX243lp1O2s4su0Yfs1S35czNV91ipo2bbubqOrczZN16661uTQ+mK3v56aefZApLK/9T7j8hi592kbVBXhQPj73EQ2TJ7D09UQQUAUVAEWiaCCjhUWK56ZSWEoHTYIqAIlATCCy22CWyM1V32X5xFV20tCZKRDOhCDQ/BMpBeKShMmXKFOd960mLJ795yHwz7kszy84y438bYyZMG2fatmpn5mo/r5m7w7xmng495NjD9OjQ06zQY5W8qCE8Flusuzn33CHixbuF7NiiCzrngaQXFAFFQBFoYggo4VFigSnhUSJwGkwRUAQaHYHZs614sZ1h1lhjATNgwBpKeDR6iWgGFIHmiUClCY9CqFlZjGPijF/N2N9Gy98YM27aGDNmymgz/Ocx5qwtzs4L7gmPu+76UBa1307ayIXzntELioAioAgoAk0MAekMVEpAYK211rJvvPFGCSE1iCKgCCgCuQhMmDDBrrbaalbWDsrekHWC7Jprrun+Lrvssuz10aNH27///e85fzKP3Z5//vk513bddddsmPjJqac+a+eb73y799732VtvfTd+W38rAopAE0Hg999/t8cee6xdeeWVraxRYWWKhsv5kCFD7IYbbujaBNm9JO9tTj31VLvKKqvYQw45xE6ePNndf+mll+wWW2xhZW0ye/vtt+eFKeXCggteaNdf/3r71FOflxK8ImE+/XSkLEl6qh09OvPe0USOPPJxe8UVr9nVV7/KDh36Y/SWTcJadoyxstuLlYVSraw3khPG/5AdAe0555zjf9p77rkn277Tzv/666/Zew09CeXzmWeeyekf6EPod7zIVG4ra6LYFVdc0R588MHuXbl33333CRarSxmub2WtFP944vG6666z6MfbbrutlXVW3HPvvPOO3XLLLV0cTz75ZE7YpHwVqr/RSEJpvv76664ur7vuuvb555+PPu7Ojz/++Cz+e+yxh7tWnzQJQLpgIzv+WFk/xsUhGx7YTTbZxH1bTzzxhLsW/a+UMNHw1LWdd97ZldOll14avWXfe+89u/XWW+dc48fEiRPt7rvvbpdddlkr69pk74cwyN6sx0kIN1lA2MrCwA4H2pqZM2fmxJhWJwgXKrNoBKE007AhbFKY/fff3+Xz3//+dzQJPW9mCMhK1CqlINAQwiNpwBLNR7xRLCZMNLyeKwKKQNNAACVlmWWWsbI+kBX3bJdpBi0oJ8OGDXN/48aNy77MrFmzstdRqBZYYAHLfdlqMXt9o402srIAczZM/OSf/3zIrrTSv+c4wiOuSN15551OOd9xxx3t//73vxyY0pSnJMUyJwL9oQhUAQGICQZADFgZUB955JFucLHEEkvYjz76yH7yySdCbs6XkxMUfwaChIEkufrqqy3tylJLLWU/+OADIQJGW9nVxB1zApbwoxYJj8MOe8QRHr16XZL3RmmERwhrIhCPX/vYY4+5NhjCIN5eH3PMMVbWhbPHHXdcNj1wv/7667NtNviXS0L5hEz3/QkD0M022ywnOQgYSAkGx9QhyI9Ro0bZXr16uXAQOQzk04RBLnWIvki2CnYkHM+vvfbaFtLt22+/tbJNsJUNA7LRhPLF4Dit/mYDy0lSmn369LHUc+4vuOCCOeQO4Yn/448/du82YsSIgt9MNE3OZVdHu+SSSzq8ZAch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tpeNCn9O4pQF36cMKHKuz4k/xIhjZO2hRn31xm4jGvXlI4nLgN/14bIjXeRq9U+r3VaF0gtdqyQS1U4v9C61kAchIY2Q1TnZI1/Vbi9rAYscEPRHvRBQwqNecOnDioAioAgoAoqAIqAIKAKKgCKgCCgCioAi0BQQUMKjKZSS5lERUAQUAUVAEVAEFAFFQBFQBBQBRUARUATqhYASHvWCSx9WBBQBRUARUAQUAUVAEVAEFAFFQBFQBBSBpoCAEh5NoZQ0j4qAIqAIKAKKgCKgCCgCioAioAgoAoqAIlAvBJTwqBdc+rAioAgoAoqAItA0EHj66afNoYce6hYPji6Guf7665sBAwYY2emrwS8iOzmYk046ych2qolxybbU5uyzzzb33HNP3jOy44E56KCDzIMPPph3Ty8oAoqAIqAIKAKKgCLQUASU8GgoghpeEVAEFAFFQBGoQQT8bjCyjbrZfPPNDcQDMmLECCNb6bkdtthJSbYWdH+ses/uFK1bt3Z//pWSVqeXrSSNbFVobr75Zkeq8Dw7bLAj0axZsww7PLCL1/Tp083w4cPNkksumSVfxo0bZ+aee26XxMknn2w22GADs+mmm/ok9agIKAKKgCKgCCgCikBZEFDCoywwaiSKgCKgCCgCikBtIgDhgVfH999/7zK4ww47mKOOOspt/cl5ly5dzE8//WT69u1r3nzzTcMWxk899ZQjKo455hgD4dG7d29z22235Wylu+eee5q9997bERULLbSQWXjhhc2oUaPMWmutZT777DNHnpxxxhlm5ZVXNkcccYR5+eWXzYILLmjmm28+ty0vxxdeeMF88sknLj8vvvhibQKouVIEFAFFQBFQBBSBJouAEh5Ntug044qAIqAIKAKKQGEEQoQHBET37t3NFlts4YiQu+++21x22WWOqDj//POdhwdEBITIlltuaSA+NtpoI/fbp7j88subl156yfTo0cPF9corr5jOnTu76S0jR4403377rSGuiy66KEt4kOZjjz1m1llnHbPIIouYd99913mbLLDAAubnn3/2UetREVAEFAFFQBFQBBSBsiCghEdZYNRIFAFFQBFQBBSB2kQgjfA4/PDDzauvvmqefPJJc99995k77rjDXHPNNWbMmDHmxhtvNHhutGnTxr0YHiAQH17wDJkwYYJp2bKlIzyYpsKUmGWWWcZNYcHL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0MEcccQR7jfK7DfffGMeffRR8/e//z0Yb1O+yLt169bNzDPPPCW/xtChQ83555/vlCmw7Nq1q7nlllsKxvfZZ5+Zk046ybRr184pceTjxhtvLBhOH1AE5iQERowYYXbaaSfz6aefmsmTJ5u77rrL7L777g2G4Pbbb3ff7bfffmtatmzp2sAGRzoHRPDAAw+Ys88+23z55Zeu75g2bZprw+aAV2+Wr8hA5YMPPjCQDieeeKK56KKLqvqefN+bbLKJGTBggDnyyCNz0h4+fLi5+uqrDX3l119/bb7//nunl9BXLrzwwk4n2XLLLU3fvn1zwvHj888/d/0ppNwPP/xgxo8fb9q0aWN69OhhVlxxRbPBBhuYvfbay/Ts2TMvrL9w1FFHGfJA+wAJcdhhh5lNN93U3048nnLKKS590kO3Ovzww81GG21k+FYuv/xywzf01VdfuTjnm28+l5f+/fu7uB955BGXzgsvvGCWWWaZxDTqc0O/2fqgpc8WiwDf65VXXukev+GGG8wBBxxQbNAm8VyldI8m8fKayeohICy+SpkRkA7bnnbaaVYUBSsl6f6E3HC/e/XqZUURsEKCuOu//vqrveeee7LP7brrrmXOTeNFJ4qdveaaa+xaa63l3u++++5rUGZEibEff/yxFcUri9cTTzxRME4ZtGWfHzhwoBWlrGAYfcBaIebsq6++2mygaG7vU+6CEVLW/vjjj3a77bZz34sQHmVJYurUqVas2lbIXysDk6LjnNPLC9xksGaF2HXlQfvXnKW5l/fIkSOtEO+uLIXwqGpRineVXXbZZa2QLlasqNm0uX788cdbMQbY9ddf38qA3Qr+1txTAAA1CElEQVThacULxAqRYZ9++mm73377WfH0sKuuumo2HCdCitrddtvNvQ99/L333uvCjB071r7yyitWBmh2ueWWc/ffeuutnLDxH2BD2nV1de751VZbLf5I3m+xdmf1qFVWWcW+8847dsqUKZZ3WmqppeySSy7p8o8ewr0LLrjAoodtvfXW2bj23Xdfu+CCC1ohW7LXGnIyp32z3333nR02bFhDINOwBRA499xz7corr+y+y5tuuqlZ6WT+1Sule/j49agIgAAWb5UKIcDg2hMewtDmpCJWBdu+fXt3DeVi3nnndYOBcg0ychKr8g+xFFmxJLkBjn9/js8++2xZckL8Pl7xhkmN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cKcdRm4xbosRSPsQEY1m8n5ux6BxUd4wlG0ancF2OYoGyPNGTIvC6JoI9BoB9cW9VmPl+R3svrE8B/IhAn0JSODRl4muiIAIiIAIiIAIiIAIiIAIiIAIiIAI9DgBCTx6vAKVfREQAREQAREQAREQAREQAREQAREQgb4EJPDoy0RXREAEREAEREAEREAEREAEREAEREAEepyABB49XoHKvgiIgAiIgAiIgAiIgAiIgAiIgAiIQF8CEnj0ZaIrIiACIiACIiACIiACIiACIiACIiACPU5AAo8er0BlXwREQAREQAREQAREQAREQAREQAREoC8BCTz6MtEVERABERABERABERABERABERABERCBHicggUePV6Cy/+8i8Mknn4Q33ngjrLnmmv+ugqu0IiACIiACIiACIiACIiACIlCTgAQeNYFV8f7888+HY445Jkw88cThjz/+CLPPPns4++yzS4NeccUV4aabbgoTTTRRDLfxxhuHXXbZpSncXXfdFUaPHh1uuOGGputlJ59//nk455xzwqKLLhq23377Mu997u+3337ho48+CuOOO2749ddfw5FHHhmWW265Pv7SC7/88kvYcccdYxjCcU6+4fLbb78Fygsr7q288sphnXXWCdNOO20aRenvUaNGhSeeeCJMMMEEYfzxx4/+//rrr8iP88UXXzzmc/311y+Mi3xcdNFF4fXXXw/nn39+oZ9uuEg5L7744vD++++H5ZdfPpa5v3y1w6QT9THQNtZfeXRdBERABERABERABERABERABNoikMl1nIBN7DObyGdrr712ZpUS/55++umW6diEM5tlllmi3xlmmCG7//77s2+++SaG+eqrrzITmGSLLLJIvG+T+5ZxpTcfeuihbMSIEZkJGWLYE088Mb1d+bdpFmRXXXVVjIMybbTRRqVhzzvvvIb/vfbaK3vzzTezv//+O/v4448zE7zEeyaUaPiZbbbZsrfffrs03tTDl19+mZkQKJtqqqliPJT12muvzS688MJsk002ySaZZJJ4faeddoppp2H5bYKpeH+FFVbI3+qqc9rA7bffXimvdZkMtD461ca6CrgyIwIiIAIiIAIiIAIiIAIi0PMEQs+XoIsLcMopp8QJKgKCLbfcsmVOL7300oZfW67Q5BdBiH3Vz6677rps+umnz6oKPC655JJsiimmyEw7IzvssMNi/O0KPDxDLjihTC+88IJf7nM0zZZszjnnbJSJSbG7Cy64IJtmmmmioAJBz6OPPtoQgIwcOdK91TquuOKKMS3TkGkK99hjj2XjjDNOvDdmzJime2+99VYjf3nmTR675OSLL76I+a0qnKnKZCD1MRhtrEtwKxsiIAIiIAIiIAIiIAIiIAI9TkACj0GswNNPPz2be+654ySVSferr75amJotwcjmn3/+bOGFF45+bWlHkz9bxtA4RwuiqsDj22+/zb777rsY9rLLLotxD1TgMfnkkzfKtN122zXylf9hy1WyqaeeOptxxhljugg13Nkyl+zGG2/003i8/vrroz+0Ytpxq622WgyfF3gQ17zzzhvvXX755U1Rr7vuutmhhx4a7/G7211dgUdVJgOpj8FoY91eD8qfCIiACIiACIiACIiACIhAbxCQwGMQ6wmBx1ZbbZWtt956cVK98847F6Zmdi2icMBsSUR/eYFHGqiOwCMN10mBB1/10fAYb7zxMrQk8o5lKyy/Of7447MFFlgg+k0FHs8880xmdkCagqEtQpzHHnts0/WqJ60m965pki4rQltmlVVWyV577bWY7gYbbFA1qUJ/CJZuu+227O67784QTOTdn3/+mZmtkYbQ6+GHH45Lb5588skMgVd/7tNPP81uvvnm7L777ssQfMGoqoZHVSadqo9OtbH+WOi6CIiACIiACIiACIiACIiACNQhIIFHHVo1/brA48EHH4wTVexVvPfee31iWWqppbKjjjoqu+aaa6K/bhd4vPTSS9kaa6wR87rbbrv1KQ8Tf5bSYIOkSODRJ4BdQBiEXY/vv/++6Hbptf4m9w888EDMJ0uCEDrgSGOOOebIKIcZK4332xV4PPXUU9FWy5RTThkFQAgksCeSCni22GKL7D//+U9MB5shCMHw53/k7YcffmgqI4IY/JkR17iMCb8TTjhhDDNQgUcRk6bE7aSd+pDAI09R5yIgAiIgAiIgAiIgAiIgAsNJQAKPQaTvAg+ScHsKGO9M3b333ptNOumkGYYme0ngQb59Eo7Ry9RR1kMOOSReaiXwwDArjBDwEBfLYFppO6Rp5H+nAg8ETAgcMPSK8GHJJZeM2hEeZv/9988OOuigeOoCjw033NBv1zoSbu+9944GUTFWu8cee8Sy2K4wjXgQimy77bbxOkub9tlnn6jtgYDADdWedNJJ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RRnJOyzzrh0mVeqDdNptY4Sl3cNWTgREQAREQAREQAREQAREQASGgoAEHkNBuYvTePDBB4PZjwi33npr2HTTTbsmp8OdLzMaGgURpq0TBRVdA0YZEQEREAEREAEREAEREAEREAERqERAAo9KmMZeT2anIphtiGDLbLqqkMOZL7NVEUj/3nvvDbZ0pKu4KDMiIAIiIAIiIAIiIAIiIAIiIALVCEjgUY3TWOuLpS0sVek2N9z5Gu70u60+lB8REAEREAEREAEREAEREAER6DUCEnj0Wo0pvyIgAiIgAiIgAiIgAiIgAiIgAiIgAqUEJPAoRSQPIiACIiACIiACIiACIiACIiACIiACvUZAAo9eqzHlVwREQAREQAREQAREQAREQAREQAREoJSABB6liORBBERABERABERABERABERABERABESg1whI4NFrNab8ioAIiIAIiIAIiIAIiIAIiIAIiIAIlBKQwKMUkTyIgAiIgAiIgAiIgAiIgAiIgAiIgAj0GgEJPHqtxpRfERABERABERABERABERABERABERCBUgISeJQikgcREAEREAEREAEREAEREAEREAEREIFeIyCBR6/VmPIrAiIgAiIgAiIgAiIgAiIgAiIgAiJQSkACj1JE8iACIiACIiACIiACIiACIiACIiACItBrBCTw6LUaU35FQAREQAREQAREQAREQAREQAREQARKCUjgUYpIHkRABERABERABERABERABERABERABHqNgAQevVZjyq8IiIAIiIAIiIAIiIAIiIAIiIAIiEApAQk8ShHJgwiIgAiIgAiIgAiIgAiIgAiIgAiIQK8RkMCj12pM+RUBERABERABERABERABERABERABESglIIFHKSJ5EAEREAEREAEREAEREAEREAEREAER6DUCEnj0Wo0pvyIgAiIgAiIgAiIgAiIgAiIgAiIgAqUE/h849JjckEVpEgAAAABJRU5ErkJggg==', 'mime': 'image/png', 'width': 1084, 'height': 628, 'alt': 'Figure 1 from Galmán Graíño et al. (2018): four GC–MS chromatograms labelled M1.1, M3, P4 and P3.1.'}]
        REFERENCE = {'authors': ['Soraya Galmán Graíño', 'Raquel Sendón', 'Julia López Hernández', 'Ana Rodríguez-Bernaldo de Quirós'], 'year': 2018, 'title': 'GC-MS Screening Analysis for the Identification of Potential Migrants in Plastic and Paper-Based Candy Wrappers', 'journal': 'Polymers', 'volume': 10, 'issue': 7, 'article_number': 802, 'doi': '10.3390/polym10070802', 'url': 'https://doi.org/10.3390/polym10070802', 'publisher_url': 'https://www.mdpi.com/2073-4360/10/7/802', 'figure_number': 1, 'samples': ['M1.1', 'M3', 'P4', 'P3.1'], 'license': 'CC BY 4.0', 'license_url': 'https://creativecommons.org/licenses/by/4.0/', 'display_asset': 'User-supplied screenshot of Figure 1'}
        ASSET_SHA256 = '211453b0920dca4b4a4aacd46a5a9ca450782b0f39cf45baf4981fb375296e98'
        CITATION_HTML = '''<div class="gx-paper-citation" style="margin-top:12px;border-left:3px solid #126b78;padding:9px 13px;background:#e9f4f1">
          <div class="gx-label">Figure 1 · Source and attribution</div>
          <p class="gx-small" style="margin:6px 0;color:#253b43">Galmán Graíño, S.; Sendón, R.; López Hernández, J.; Rodríguez-Bernaldo de Quirós, A. (2018).
            <i>GC-MS Screening Analysis for the Identification of Potential Migrants in Plastic and Paper-Based Candy Wrappers.</i>
            <b>Polymers, 10</b>(7), 802.
            <a href="https://doi.org/10.3390/polym10070802" target="_blank" rel="noopener noreferrer">doi:10.3390/polym10070802</a>.</p>
          <p class="gx-small" style="margin:4px 0">Figure 1 reproduced from the article. © 2018 the authors.
            <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer">CC BY 4.0</a>.</p>
        </div>'''
        def section(features, provenance, state, viewer_html):
            return f'''<div class="gx gx-section gx-det"><div class="gx-label">GC–MS evidence · Source study</div>
              <h2 style="margin:7px 0">Four samples. Many possible identities.</h2>
              <p class="gx-small" style="margin:8px 0">Figure 1 shows the GC–MS chromatograms of samples M1.1, M3, P4 and P3.1. Zoom, then drag to inspect the traces.</p>
              {viewer_html}
              <div class="gx-note gx-trace-readout" aria-live="polite"><strong>Figure 1 · M1.1, M3, P4 and P3.1</strong>Display zoom: {float(state.get('zoom',1)):.1f}×.</div>
              {CITATION_HTML}
              <p class="gx-small" style="margin-top:12px"><b>From chromatograms to molecular identities.</b> The {len(features)} curated FCM2018 assignments below use Tables 2–3 of the same article. Choose an assignment, then inspect its structure and same-formula alternatives with your EFSA-trained genotoxicity model.</p>
            </div>'''
        return SimpleNamespace(panels=PANELS, section=section, reference=REFERENCE,
                               asset_sha256=ASSET_SHA256)
    detection = _build_detection()
    return (detection,)




@app.cell(hide_code=True)
def _(detection, mo, widgets):
    trace_viewer = mo.ui.anywidget(widgets.TraceViewer(panels=detection.panels))
    return (trace_viewer,)


@app.cell(hide_code=True)
def _(detection, features, mo, provenance, trace_viewer):
    mo.Html(detection.section(features, provenance, trace_viewer.value, trace_viewer.text))
    return


@app.cell(hide_code=True)
def _(demo, mo):
    get_feature, set_feature = mo.state(demo.HOOK_FEATURE, allow_self_loops=True)
    get_structure, set_structure = mo.state(None, allow_self_loops=True)
    return get_feature, get_structure, set_feature, set_structure


@app.cell(hide_code=True)
def _(demo, features, get_feature, mo, set_feature, set_structure):
    _ordered = features.assign(
        _lead=features["feature_id"].ne(demo.HOOK_FEATURE),
        _confirmed=features["msi_level"].eq(1),
    ).sort_values(["_lead", "_confirmed", "feature_id"])
    _options = {
        f"{_row['feature_id']} · {_row['display_name']} · RT {_row['rt_mean']:.2f} min · {'confirmed' if _row['msi_level'] == 1 else 'library assignment'}": _row["feature_id"]
        for _, _row in _ordered.iterrows()
    }
    _selected_label = next(_label for _label, _fid in _options.items() if _fid == get_feature())
    def _select_feature(feature_id):
        set_structure(None)
        set_feature(feature_id)
    feature_picker = mo.ui.dropdown(
        options=_options, value=_selected_label, searchable=True,
        label="GC–MS entry · select the case", full_width=True,
        on_change=_select_feature,
    )
    return (feature_picker,)


@app.cell(hide_code=True)
def _(mo):
    cutoff = mo.ui.slider(
        start=.10, stop=.90, step=.01, value=.40,
        show_value=True, full_width=True, debounce=False,
        label="Minimum molecular similarity · Tc",
    )
    return (cutoff,)


@app.cell(hide_code=True)
def _(mo, set_structure):
    identity_assumption = mo.ui.radio(
        options={"Assigned structure": "published", "Explore same-formula structures": "same_formula"},
        value="Assigned structure", inline=False, label="Identity assumption",
        on_change=lambda _value: set_structure(None),
    )
    return (identity_assumption,)


@app.cell(hide_code=True)
def _(cutoff, demo, features, get_feature, identity_assumption, scored):
    view = demo.evaluate(
        scored, features, get_feature(), float(cutoff.value), identity_assumption.value
    )
    return (view,)


@app.cell(hide_code=True)
def _(demo, get_structure, view):
    inspected = demo.inspect_structure(view, get_structure())
    return (inspected,)


@app.cell(hide_code=True)
def _(cutoff, feature_picker, identity_assumption, mo):
    mo.vstack([
        feature_picker,
        mo.hstack([
            mo.vstack([cutoff, mo.Html('<p class="gx-small">Tc compares molecular fingerprints, from 0 to 1. The starting criterion, 0.40, is chosen for this example. Move it to change applicability.</p>')], gap=.4),
            mo.vstack([identity_assumption, mo.Html('<p class="gx-small">Assigned: use the table identity. Explore: inspect the retained same-formula structures and compare the identity scenarios.</p>')], gap=.4),
        ], widths=[1, 1], gap=1.4, align="start", wrap=True),
    ], gap=.7).style({
        "max-width": "1120px", "margin": "14px auto", "padding": "16px",
        "border": "1px solid #dce2df", "border-radius": "8px", "background": "#fffdf9",
    })
    return


@app.cell(hide_code=True)
def _(demo_view, features, inspected, mo, provenance, study_chart, view):
    study_chart.widget.svg = demo_view.study_plot(features,view,inspected)
    mo.Html(demo_view.whole_intro(features,provenance,view,inspected) + study_chart.text + demo_view.study_finding(features,view,inspected))
    return


@app.cell(hide_code=True)
def _(features, mo, set_feature, set_structure, widgets):
    _feature_ids = set(features['feature_id'])
    _study_widget = widgets.StudyChart()
    def _select_plotted_assignment(_change):
        _clicked = _study_widget.click_key
        if _clicked in _feature_ids:
            set_structure(None)
            set_feature(_clicked)
    _study_widget.observe(_select_plotted_assignment, names='click_seq')
    study_chart = mo.ui.anywidget(_study_widget)
    return (study_chart,)


@app.cell(hide_code=True)
def _(demo_view, inspected, mo, view):
    mo.Html(demo_view.feature_context(view) + demo_view.selection_banner(view,inspected))
    return


@app.cell(hide_code=True)
def _(demo_view, inspected, mo, molecule_grid, view):
    _rows = view['candidates']
    if view['assumption']=='published':
        _rows = _rows.loc[_rows['is_assigned'].fillna(False)]
    _cards = [dict(key=f"{view['feature']['feature_id']}:{int(_row['rank'])}",
                   name=demo_view.structure_name(_row,view['feature']),
                   html=demo_view.choice_label(_row,view,inspected['selection_key']))
              for _,_row in _rows.iterrows()]
    with molecule_grid.widget.hold_sync():
        molecule_grid.widget.cards = _cards
        molecule_grid.widget.selected_key = inspected['selection_key']
    mo.Html('<div class="gx gx-workspace"><section>' + demo_view.choices_intro(view)
            + '<div style="margin-top:12px">' + molecule_grid.text + '</div>'
            + '<p class="gx-small" style="margin-top:9px">The violet card is the inspected molecule. IN / OUT refers to that molecule’s Tc compared with your current minimum.</p></section>'
            + demo_view.inspector(view,inspected) + '</div>')
    return


@app.cell(hide_code=True)
def _(features, mo, set_structure, widgets):
    _feature_ids = set(features['feature_id'])
    _molecular_widget = widgets.MoleculeGrid()
    def _select_molecule(_change):
        _clicked = _molecular_widget.click_key
        if ':' in _clicked and _clicked.split(':',1)[0] in _feature_ids:
            set_structure(_clicked)
    _molecular_widget.observe(_select_molecule, names='click_seq')
    molecule_grid = mo.ui.anywidget(_molecular_widget)
    return (molecule_grid,)


@app.cell(hide_code=True)
def _(demo_view, inspected, mo, view):
    mo.Html(demo_view.two_routes(view, inspected) + demo_view.weight_summary(view, inspected))
    return


@app.cell(hide_code=True)
def _(demo_view, inspected, mo, view):
    mo.Html(demo_view.alert_panel(view,inspected))
    return


@app.cell(hide_code=True)
def _(demo_view, inspected, mo, view):
    mo.Html('<div class="gx gx-live-sensitivity gx-section" data-selection-key="' + inspected['selection_key'] + '">'
            + '<div class="gx-label">Sensitivity to your Tc criterion</div><h2>One molecule. A whole identity set.</h2>'
            + '<p class="gx-small" style="margin:8px 0 14px">Amber marks the current Tc setting in both charts. The left chart follows the inspected molecule; the right chart summarizes the identity set for this GC–MS entry.</p>'
            + demo_view.sensitivity_plot(view,inspected) + '</div>')
    return


@app.cell(hide_code=True)
def _(demo_view, inspected, mo, view):
    mo.Html(demo_view.closing(view, inspected))
    return


@app.cell(hide_code=True)
def _(demo_view, mo, provenance, view):
    _f = view["feature"]
    _details = f'''<div class="gx gx-details">
      <h3>Project data and this illustrative case</h3>
      <p>The project's EFSA table contains {provenance['n_efsa_records']:,} substance–output records,
        representing {provenance['n_efsa_resolved_identities']:,} resolved raw InChIKey identities.
        The selected labeling and standardization pipeline yields {provenance['n_training_substances']:,}
        modeling substances. This demonstration uses {provenance['n_assignments']} assigned identities from the project input table.</p>
      <h3>Example and input data</h3>
      <p>This is an original project demonstration combining real data from several sources:
        EFSA training conclusions, the bundled FCM2018 assignments and PubChem structures.
        FCM2018 is the local identifier of the selected input table.</p>
      <p>The repository documents this particular FCM2018 table as curated from Tables 2–3 of
        <a href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6403844/" target="_blank" rel="noopener noreferrer">Galmán Graíño et al. (2018), doi:10.3390/polym10070802</a>.
        This reference concerns the assignment data; the model calculations and identity experiment are produced by this notebook.</p>
      <p>The assigned identities are loaded from the bundled project table in data/metabolights/FCM2018__*.tsv.
        Assigned molecular records have PubChem links; retained other structures come from data/public_candidates.parquet.
        The source SHA-256 hashes and available molecule identifiers are included in the downloadable record.</p>
      <p>The input data mark {provenance['n_standard_confirmed']} identities as standard-confirmed (MSI 1);
        the others are tentative (MSI 2). These are recorded evidence categories, not new measurements performed here.
        The 33 rows are curated GC–MS assignments, not 33 raw chromatographic traces or independently sampled peaks.
        This page displays reported analytical metadata alongside structure-specific predictions.
        All retained structures remain available for molecular inspection.
        For confirmed entries, feature-level screening weight stays on the assigned identity.
        For library assignments, the exploration scenario gives each retained structure equal weight.</p>
      <p>Selected entry: {_f['feature_id']}; assigned name: {_f['assigned_name']};
        structure identifier: {_f['database_identifier']}; input retention time {_f['rt_mean']:.2f} min;
        input representative fragment ion m/z {_f['mass_to_charge']:.0f}.
        The formula is calculated from the assigned structure, rather than independent exact-mass evidence.
        No raw chromatogram or measured migration into food is generated by this notebook.</p>

      <h3>Source figure and attribution</h3>
      <p>The displayed image reproduces Figure 1 of Galmán Graíño, S.; Sendón, R.; López Hernández, J.;
        Rodríguez-Bernaldo de Quirós, A. (2018). <i>GC-MS Screening Analysis for the Identification of
        Potential Migrants in Plastic and Paper-Based Candy Wrappers.</i> <b>Polymers, 10</b>(7), 802.
        <a href="https://doi.org/10.3390/polym10070802" target="_blank" rel="noopener noreferrer">doi:10.3390/polym10070802</a>.
        Figure 1 contains chromatograms for M1.1, M3, P4 and P3.1. © 2018 the authors;
        <a href="https://creativecommons.org/licenses/by/4.0/" target="_blank" rel="noopener noreferrer">CC BY 4.0</a>.</p>
      <p>The figure is embedded from the supplied screenshot, with zoom and pan as display controls.
        The project's FCM2018 table is curated from Tables 2–3 of the same source study.
        Project feature identifiers are local row labels. Model scores use the selected molecular
        structures, rather than intensities read from the figure. The export includes the figure reference,
        image SHA-256, zoom and pan separately from structure-specific model results.</p>

      <h3>The fixed model</h3>
      <p>Your existing LightGBM configuration, seed 0, is fitted on {provenance['n_training_substances']:,}
        standardized EFSA substances ({provenance['n_training_positive']} positive labels).
        Label aggregation: any positive; ambiguous labels excluded. No external endpoint is merged in this figure.
        Both training and retained structures use the same cleanup, parent, uncharging and tautomer pipeline,
        then radius-2, 2,048-bit Morgan fingerprints.</p>
      <p>The existing aggregation pipeline deduplicates standardized identities by keeping the first row;
        {provenance['n_standardized_label_conflicts']} standardized keys have conflicting labels before that step.
        Training membership is checked by standardized InChIKey; Tc = 1 alone does not prove identical compounds.</p>
      <p>The score is an uncalibrated classifier output. In-training identities are marked explicitly.
        This is not external validation, a genotoxicity assay, or a risk assessment.
        The full model analysis remains in <a href="https://github.com/E4CE-UA/genotox-food-migrants/blob/main/notebooks/genotox_marimo.py" target="_blank" rel="noopener noreferrer">genotox_marimo.py</a>.</p>

      <h3>What the controls mean</h3>
      <p>Training structures are the {provenance['n_training_substances']:,} standardized EFSA substances used to fit the classifier.
        Molecular similarity is Tanimoto similarity between radius-2, 2,048-bit fingerprints: 0 means no shared on-bits,
        1 means identical fingerprints, which is not a guarantee of identical molecules.</p>
      <p>The minimum similarity (also called a cutoff) is a rule selected by the visitor.
        0.40 is the initial value chosen for this demonstration; it is not derived as a validated accuracy,
        confidence or regulatory boundary. The rule uses full precision when comparing similarity values.
        Clicking a molecule selects that retained structure for the inspector and structural-alert panel; it changes its displayed structure,
        nearest EFSA neighbour, similarity, genotoxicity score and domain status.
        Clicking a point in the overview selects its corresponding GC–MS feature.
        Moving the criterion changes only domain status and evaluable coverage, not the fitted model or its raw scores.
        Other structures are the retained PubChem molecules with the same formula plus the assigned identity.
        They are not validated by EI spectra, retention index, volatility or migration plausibility.
        Enumeration order is not an evidence ranking; equal weights are not measured confidence.</p>
      <p>Evaluable identity weight = Σ wₖ over structures that have a valid model output and pass the cutoff.
        A conditional score averages only that part, dividing Σ wₖsₖ by the evaluable weight.
        When that weight is zero, the score is unavailable. The residual is never counted as safe.</p>

      <h3>Minimum-similarity sensitivity for the selected entry</h3>
      <p>The left curve is the inspected molecule’s domain membership over Tc settings (IN/OUT). The right curves aggregate identity weights for the current GC–MS entry. Clicking a molecule changes the left curve; it does not reweight the complete set. Amber markers show the active Tc and identity scenario. These deterministic curves are not confidence intervals.</p>
    </div>'''
    mo.accordion({
        "Methods, provenance & sensitivity details": mo.vstack([
            mo.Html(_details),
        ]),
    })
    return


@app.cell(hide_code=True)
def _(demo, detection, inspected, mo, provenance, trace_viewer, view):
    import json as _json
    _payload = _json.loads(demo.snapshot(view,provenance,inspected))
    _state = trace_viewer.value
    _payload['analytical_context_display'] = dict(
        source='Galmán Graíño et al. (2018), Figure 1',
        reference=detection.reference, image_sha256=detection.asset_sha256,
        same_source_study_as_assignments=True,
        zoom=float(_state.get('zoom',1)),
        pan_x=float(_state.get('pan_x',0)), pan_y=float(_state.get('pan_y',0)),
        contributes_to_model_results=False,
    )
    mo.download(_json.dumps(_payload,ensure_ascii=False,indent=2,allow_nan=False).encode('utf-8'),
        filename=f"{view['feature']['feature_id']}_structure{int(inspected['selected']['rank']):02d}_Tc{view['cutoff']:.2f}.json",
        mimetype='application/json',label='Download the current results and source identifiers')
    return


if __name__ == "__main__":
    app.run()
