"""Shared, versioned competition demo component. No compatibility implementation."""

def build_demo(notebook_source_file):
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
    from genotox_food_migrants import chemistry, data, labels, models, paths, public_features, splitting
    STUDY = 'FCM2018'
    HOOK_FEATURE = 'FCM2018-0014'
    NIAS_NAMES = {'FCM2018-0010': 'Irganox 1010 degradation product', 'FCM2018-0004': 'BHT degradation product'}

    def file_sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @lru_cache(maxsize=1)
    def prepare():
        """Return candidates, assignment metadata, and reproducibility information."""
        with rdBase.BlockLogs():
            return prepare_tables()

    def prepare_tables():
        validation_file = paths.VALIDATION / 'calibration_method.json'
        if validation_file.exists():
            expected_versions = json.loads(validation_file.read_text()).get('versions', {})
            mismatches = {name: {'expected': expected_versions[name], 'installed': importlib.metadata.version(name)}
                          for name in ('rdkit', 'lightgbm', 'numpy', 'pandas', 'scikit-learn')
                          if name in expected_versions and importlib.metadata.version(name) != expected_versions[name]}
            if mismatches:
                raise RuntimeError('This release requires the pinned scientific environment. '
                    'Install requirements.txt or run this complete file with uv run. Versions: ' + json.dumps(mismatches))
        if not (paths.PUBLIC_CANDIDATES.exists() and paths.PUBLIC_FEATURES.exists()):
            raise FileNotFoundError('The bundled public candidate and feature tables are required.')
        bundle = pd.read_parquet(paths.PUBLIC_CANDIDATES, columns=['study', 'n_requested'])
        study_bundle = bundle.loc[bundle['study'].eq(STUDY)]
        bundled_features = pd.read_parquet(paths.PUBLIC_FEATURES, columns=['study'])
        if study_bundle.empty or study_bundle['n_requested'].min() < 20 or (not bundled_features['study'].eq(STUDY).any()):
            raise ValueError('The local bundle must contain FCM2018 and 20 retained structures per assignment.')
        candidates, features, info = public_features.build(STUDY, n_candidates=20)
        RDLogger.DisableLog('rdApp.*')
        if not info.get('from_bundle') or candidates.empty or features.empty:
            raise ValueError('FCM2018 must be present in the bundled tables; no demo fallback is used.')
        maf = public_features.load_maf(STUDY)
        if maf['database_identifier'].duplicated().any():
            raise ValueError('Source identifiers must uniquely match the curated assignment rows.')
        metadata = maf.set_index('database_identifier')
        features = features.copy()
        for target, source in (('assigned_name', 'metabolite_identification'), ('assigned_smiles', 'smiles'), ('assigned_uri', 'uri'), ('material', 'material')):
            features[target] = features['database_identifier'].map(metadata[source])
        if features['assigned_smiles'].isna().any():
            raise ValueError('An assignment is missing from the curated source table.')
        features['display_name'] = features['feature_id'].map(NIAS_NAMES).fillna(features['assigned_name'])
        features['is_nias'] = features['feature_id'].isin(NIAS_NAMES)
        features['evidence_group'] = np.where(features['msi_level'].eq(1), 'Standard-confirmed', 'Library-only')
        formula = candidates['smiles'].map(lambda s: rdMolDescriptors.CalcMolFormula(Chem.MolFromSmiles(s)) if isinstance(s, str) and Chem.MolFromSmiles(s) is not None else None)
        expected = candidates['feature_id'].map(features.set_index('feature_id')['chemical_formula'])
        if formula.isna().any() or formula.ne(expected).any():
            raise ValueError('A retained structure is invalid or does not share the source formula.')
        assigned_counts = candidates['is_assigned'].fillna(False).groupby(candidates['feature_id']).sum()
        if not assigned_counts.eq(1).all():
            raise ValueError('Each entry must retain exactly one assigned structure.')
        rdkit_version = importlib.metadata.version('rdkit')
        standardization_cache = paths.CACHE / f'demo_standardization_rdkit_{rdkit_version}.parquet'
        with rdBase.BlockLogs():
            efsa_records = data.load_efsa()
            aggregated = labels.aggregate_by_substance(data.with_structure_only(efsa_records), key='inchikey', rule='any_positive', ambiguous='exclude')
            standardized = chemistry.standardize_table(aggregated, cache_path=standardization_cache)
            candidate_std = chemistry.standardize_table(candidates[['smiles']], cache_path=standardization_cache)
        reconciled = labels.aggregate_standardized(standardized, rule='any_positive')
        training = reconciled.loc[reconciled['y'].notna()].reset_index(drop=True)
        label_conflicts = int(standardized.loc[standardized['ok'].fillna(False)].groupby('inchikey_std')['y'].nunique().gt(1).sum())
        y = training['y'].astype(int).to_numpy()
        train_fp = chemistry.fingerprints(training['smiles_std'], radius=2, n_bits=2048)
        train_matrix = chemistry.matrix(train_fp, n_bits=2048)
        raw_classifier = models.build('lgbm', 0).set_params(n_jobs=2).fit(train_matrix, y)
        scaffolds = [chemistry.scaffold(s) for s in training['smiles_std']]
        groups = splitting.identity_groups(scaffolds, training['inchikey_std'])
        classifier, calibration_report = models.fit_calibrated(train_matrix, y, scaffolds, seed=0, groups=groups)
        query_fp = chemistry.fingerprints(candidate_std['smiles_std'], radius=2, n_bits=2048)
        valid = [i for i, fp in enumerate(query_fp) if fp is not None]
        score = np.full(len(candidates), np.nan)
        raw_score = np.full(len(candidates), np.nan)
        similarity = np.full(len(candidates), np.nan)
        nearest_indices = np.full(len(candidates), -1, dtype=int)
        if valid:
            raw_score[valid] = raw_classifier.predict_proba(chemistry.matrix([query_fp[i] for i in valid], n_bits=2048))[:, 1]
            score[valid] = classifier.predict_proba(chemistry.matrix([query_fp[i] for i in valid], n_bits=2048))[:, 1]
            similarity[valid], nearest_indices[valid] = chemistry.nearest_neighbor([query_fp[i] for i in valid], train_fp)
        scored = candidates.copy()
        scored['score'] = score
        scored['raw_score'] = raw_score
        scored['tanimoto_neighbour'] = similarity
        scored['smiles_std'] = candidate_std['smiles_std'].to_numpy()
        scored['inchikey_std'] = candidate_std['inchikey_std'].to_numpy()
        scored['structure_note'] = candidate_std['note'].fillna('').to_numpy()
        scored['in_training'] = scored['inchikey_std'].isin(set(training['inchikey_std']))
        scored['nn_smiles'] = [training.iloc[j]['smiles_std'] if j >= 0 else '' for j in nearest_indices]
        scored['nn_name'] = [str(training.iloc[j]['name']) if j >= 0 else 'Unavailable' for j in nearest_indices]
        scored['nn_inchikey'] = [training.iloc[j]['inchikey_std'] if j >= 0 else None for j in nearest_indices]
        assigned = scored[scored['is_assigned'].fillna(False)].set_index('feature_id')
        for col in ('score', 'tanimoto_neighbour', 'in_training', 'nn_name'):
            features[col] = features['feature_id'].map(assigned[col])
        source_maf = next((paths.DATA / 'metabolights').glob('FCM2018__*.tsv'))
        provenance = {'study_id': STUDY, 'study_id_kind': 'repository-local curated assignment table', 'collection_description': 'Original project demonstration combining real data from multiple sources.', 'sources': {'training_labels': 'bundled EFSA OpenFoodTox table', 'molecular_structures': 'retained PubChem structures and assigned structure URIs', 'assignments': 'bundled FCM2018 input table'}, 'assignment_table_reference': {'doi': '10.3390/polym10070802', 'tables': [2, 3], 'scope': 'Reference for the bundled FCM2018 assignment table, not a publication of this project or its model results.', 'documented_in': "src/public_features.py; repository's original FCM2018 notebook", 'url': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6403844/'}, 'n_efsa_records': len(efsa_records), 'n_efsa_resolved_identities': int(data.with_structure_only(efsa_records)['inchikey'].nunique()), 'n_efsa_structured_records': int(efsa_records['has_structure'].sum()), 'n_assignments': len(features), 'n_standard_confirmed': int(features['msi_level'].eq(1).sum()), 'n_training_substances': len(training), 'n_training_positive': int(y.sum()), 'n_standardized_label_conflicts': label_conflicts, 'n_structure_exclusions': int((~standardized['ok'].fillna(False)).sum()), 'structure_exclusion_reasons': standardized.loc[~standardized['ok'].fillna(False), 'note'].value_counts().to_dict(), 'n_parent_identities_merged': int((reconciled['n_raw_identities'] > 1).sum()), 'label_rule': 'any_positive; ambiguous excluded; conclusion counts reconciled by standardized InChIKey', 'model': 'LightGBM with sigmoid calibration; 3 scaffold calibration folds, seed 0', 'model_parameters': raw_classifier.get_params(), 'calibration': calibration_report, 'probability_endpoint': 'estimated probability of an any-positive EFSA conclusion', 'fingerprint': {'name': 'ECFP4', 'radius': 2, 'bits': 2048}, 'standardization': 'single organic parent scope > Cleanup > organic fragment > Uncharger > canonical tautomer', 'standardization_policy': chemistry.STANDARDIZATION_POLICY, 'candidate_scope': 'retained same-formula structures, not spectrum-supported candidates', 'versions': {name: importlib.metadata.version(name) for name in ('rdkit', 'lightgbm', 'numpy', 'pandas', 'scikit-learn', 'marimo')}, 'sha256': {'efsa_estructuras.csv': file_sha256(paths.EFSA_CSV), 'public_candidates.parquet': file_sha256(paths.PUBLIC_CANDIDATES), 'public_features.parquet': file_sha256(paths.PUBLIC_FEATURES), source_maf.name: file_sha256(source_maf), 'what_if_wrong_compound.py': file_sha256(notebook_source_file), 'models.py': file_sha256(paths.PACKAGE / 'models.py'), 'chemistry.py': file_sha256(paths.PACKAGE / 'chemistry.py'), 'labels.py': file_sha256(paths.PACKAGE / 'labels.py')}}
        validation_file = paths.VALIDATION / 'calibration_method.json'
        provenance['validation'] = json.loads(validation_file.read_text()) if validation_file.exists() else {'status': 'not available'}
        release_audit_file = paths.VALIDATION / 'source_release_audit.json'
        provenance['efsa_release'] = json.loads(release_audit_file.read_text()) if release_audit_file.exists() else {'status': 'unverified'}
        provenance['claim'] = 'transparent compound-identity uncertainty screening demonstration'
        provenance['calibration']['group_policy'] = 'Stable ring scaffolds and individual acyclic standardized identities'
        provenance['sampled_pool_audit'] = {
            'n_sampled_assignments': int(features['sampled'].fillna(False).sum()),
            'n_assignments': len(features),
            'median_retained_fraction': float((features['n_candidates'] / features['n_isomers_total']).median()),
            'pool_definition': 'listed PubChem formula pool, not all chemically possible identities',
            'forced_assigned_inclusion': True,
            'alternative_ei_scores_available': False,
            'alternative_retention_indices_available': False,
        }
        provenance['model_fit_token'] = hashlib.sha256(json.dumps({
            'inputs': {k:v for k,v in provenance['sha256'].items() if k != 'what_if_wrong_compound.py'}, 'versions': provenance['versions'],
            'groups': groups.tolist(), 'calibration': calibration_report,
        }, sort_keys=True).encode()).hexdigest()
        return (scored, features, provenance)

    def evaluate(scored, features, feature_id, cutoff, assumption):
        """Change scenario assumptions over a fixed set of structure scores; never fit."""
        if not 0 <= cutoff <= 1:
            raise ValueError('Tanimoto cutoff must lie in [0, 1].')
        if assumption not in {'published', 'same_formula'}:
            raise ValueError('Unknown identity assumption.')
        feature = features.loc[features['feature_id'].eq(feature_id)].iloc[0]
        subset = scored.loc[scored['feature_id'].eq(feature_id)].copy().reset_index(drop=True)
        if subset.empty or subset['is_assigned'].fillna(False).sum() != 1:
            raise ValueError('Exactly one assigned structure is required in a nonempty scenario.')
        confirmed = bool(feature['msi_level'] == 1)
        usable = subset['score'].notna() & subset['tanimoto_neighbour'].ge(cutoff)
        published_weight = subset['is_assigned'].fillna(False).astype(float)
        alternative_weight = published_weight.copy() if confirmed else pd.Series(
            np.full(len(subset), 1.0 / len(subset)), index=subset.index)
        weight = published_weight if assumption == 'published' else alternative_weight
        if not np.isclose(weight.sum(), 1.0):
            raise ValueError('Identity weights must sum to one.')
        def average(weights):
            denominator = float(weights[usable].sum())
            return float((weights[usable] * subset.loc[usable, 'score']).sum()) / denominator if denominator > 0 else None
        covered = float(weight[usable].sum())
        weighted_sum = float((weight[usable] * subset.loc[usable, 'score']).sum())
        assigned = subset.loc[subset['is_assigned'].fillna(False)].iloc[0]
        subset['in_domain'], subset['weight'] = usable, weight
        pool_value = feature.get('n_isomers_total')
        pool_size = int(pool_value) if pd.notna(pool_value) and float(pool_value) > 0 else None
        return dict(feature=feature, candidates=subset, assigned=assigned, cutoff=float(cutoff),
            assumption=assumption, confirmed=confirmed,
            published_covered_weight=float(published_weight[usable].sum()),
            alternative_covered_weight=float(alternative_weight[usable].sum()),
            published_scenario_average=average(published_weight),
            alternative_scenario_average=average(alternative_weight),
            covered_weight=covered, uncovered_weight=max(0.0, 1.0 - covered),
            n_retained=len(subset), n_evaluable=int(usable.sum()),
            scenario_structure_coverage=float(usable.mean()),
            listed_formula_pool_size=pool_size,
            retained_pool_fraction=len(subset) / pool_size if pool_size else None,
            sampled=bool(feature.get('sampled', False)),
            covered_score_sum=weighted_sum, conditional_score=average(weight),
            n_weighted_evaluable=int((usable & weight.gt(0)).sum()),
            n_evaluable_training_identities=int((usable & subset['in_training']).sum()),
            assigned_evaluable=bool(pd.notna(assigned['score']) and assigned['tanimoto_neighbour'] >= cutoff),
            n_assigned_evaluable=int((features['score'].notna() & features['tanimoto_neighbour'].ge(cutoff)).sum()))

    def inspect_structure(view, selection_key=None):
        """Return the actual candidate requested by a molecule click.

            Inspection never changes chemical evidence or scenario weights.
            A selection from another feature, or assigned-only mode, falls back
            to the current feature's assigned structure.
            """
        candidates = view['candidates']
        keys = candidates['rank'].map(lambda rank: f"{view['feature']['feature_id']}:{int(rank)}")
        requested = candidates.loc[keys.eq(selection_key)] if selection_key is not None else candidates.iloc[0:0]
        if view['assumption'] == 'published' or requested.empty:
            selected = candidates.loc[candidates['is_assigned'].fillna(False)].iloc[0]
        else:
            selected = requested.iloc[0]
        return {'selected': selected, 'selection_key': f"{view['feature']['feature_id']}:{int(selected['rank'])}", 'selected_evaluable': bool(pd.notna(selected['score']) and selected['tanimoto_neighbour'] >= view['cutoff'])}

    def json_safe(value):
        """Preserve missing/nonfinite values as null, including nested provenance."""
        if isinstance(value, dict):
            return {key: json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [json_safe(item) for item in value]
        if isinstance(value, np.generic):
            return json_safe(value.item())
        if value is None or value is pd.NA or value is pd.NaT:
            return None
        if isinstance(value, float) and not np.isfinite(value):
            return None
        return value

    def compare_structure(view, selection_key):
        """Select a retained candidate independently of the weight scenario."""
        candidates = view['candidates']
        keys = candidates['rank'].map(lambda rank: f"{view['feature']['feature_id']}:{int(rank)}")
        rows = candidates.loc[keys.eq(selection_key)]
        if rows.empty:
            raise ValueError('Comparison identity is not retained for this entry')
        row = rows.iloc[0]
        return {'selected': row, 'selection_key': selection_key,
                'selected_evaluable': bool(pd.notna(row['score']) and row['tanimoto_neighbour'] >= view['cutoff'])}

    def snapshot(view, provenance, inspected=None):
        """JSON export of the actual setting, structures, scores, and provenance."""
        inspected = inspect_structure(view) if inspected is None else inspected
        selected = inspected['selected']
        cols = ['rank', 'cid', 'is_assigned', 'inchikey', 'smiles', 'smiles_std', 'tanimoto_neighbour', 'nn_name', 'nn_inchikey', 'score', 'raw_score', 'in_training', 'in_domain', 'weight', 'structure_note']
        rows = json.loads(view['candidates'][cols].to_json(orient='records', double_precision=15))
        payload = dict(schema_version=2, provenance=provenance,
            setting=dict(feature_id=view['feature']['feature_id'], cutoff=view['cutoff'],
                         identity_assumption=view['assumption']),
            result={key: view[key] for key in (
                'covered_weight', 'uncovered_weight', 'n_retained', 'n_evaluable',
                'n_weighted_evaluable', 'scenario_structure_coverage', 'conditional_score',
                'published_scenario_average', 'alternative_scenario_average',
                'listed_formula_pool_size', 'retained_pool_fraction', 'sampled',
                'assigned_evaluable', 'n_assigned_evaluable', 'n_evaluable_training_identities')},
            inspected_structure=dict(selection_key=inspected['selection_key'], rank=int(selected['rank']),
                cid=int(selected['cid']) if pd.notna(selected['cid']) else None,
                smiles=selected['smiles'], smiles_std=selected['smiles_std'],
                nn_smiles=selected['nn_smiles'], nn_name=selected['nn_name'], nn_inchikey=selected['nn_inchikey'],
                similarity=float(selected['tanimoto_neighbour']) if pd.notna(selected['tanimoto_neighbour']) else None,
                estimated_efsa_label_probability=float(selected['score']) if pd.notna(selected['score']) else None,
                in_training=bool(selected['in_training']),
                prediction_scope='fitted training identity; not a held-out assessment' if selected['in_training'] else 'query identity absent from fitted training set',
                meets_selected_similarity_rule=inspected['selected_evaluable']),
            interpretation=dict(
                structure_score='Estimated probability of an aggregated positive EFSA label, conditional on this structure.',
                scenario_average='Scenario average over retained structures that have scores and meet the selected similarity rule; not the probability that a GC-MS peak is genotoxic.',
                cutoff='exploratory similarity criterion; not a validated reliability boundary',
                weight='hypothetical scenario weights; not measured identity confidence',
                uncovered='excluded weight has no assigned hazard score; zero coverage returns null',
                pool='listed PubChem formula pool; not all chemically possible identities or probability of true-identity omission',
                sampling='assigned identity forcibly retained plus sampled alternatives; no unbiased full-pool expectation claimed',
                training_membership='training identities are fitted estimates, not independently held-out predictions',
                standard_confirmation='identity confirmation preserved in both scenarios; does not validate genotoxicity'),
            scenario_average_over_retained_structures=view['conditional_score'],
            structural_alerts=selected_alerts(selected), candidates=rows)
        return json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False).encode('utf-8')

    def selected_alerts(row):
        from genotox_food_migrants import candidate_view, chemistry
        valid = pd.notna(row['score'])
        smiles = row['smiles_std'] if valid else row['smiles']
        parsed = isinstance(smiles, str) and Chem.MolFromSmiles(smiles) is not None
        hits = candidate_view.alert_hits(smiles) if parsed else []
        return {'representation': 'standardized model input' if valid else 'retained raw structure', 'smiles': smiles, 'parsed': bool(parsed), 'rule_count': len(chemistry.DNA_REACTIVE_ALERTS), 'matched_names': [h[0] for h in hits] if parsed else None, 'matched_atoms': sorted({a for h in hits for a in h[1]})}
    from types import SimpleNamespace
    return SimpleNamespace(STUDY=STUDY, HOOK_FEATURE=HOOK_FEATURE, prepare=prepare, evaluate=evaluate, inspect_structure=inspect_structure, compare_structure=compare_structure, snapshot=snapshot, selected_alerts=selected_alerts)
