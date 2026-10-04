"""Reproduce internal evaluation with matched estimators and auditable splits.

Run with the pinned requirements: python scripts/validate_calibration.py
The similarity rule is selected using nested outer-TRAIN OOF predictions only.
"""
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import rdBase
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from genotox_food_migrants import chemistry, data, labels, models, splitting
SEEDS = list(range(5))
THRESHOLDS = np.round(np.arange(0, .91, .05), 2)
PROTOCOLS = {'scaffold_identity': None, 'scaffold_acyclic_series_070': .70}
DEST = ROOT / 'docs'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metrics(y, p):
    two = len(np.unique(y)) == 2
    return dict(n_test=len(y), pos_test=int(sum(y)), prevalence=float(np.mean(y)),
                pr_auc=float(average_precision_score(y, p)) if two else None,
                roc_auc=float(roc_auc_score(y, p)) if two else None,
                brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
                ece=models.ece(y, p))


def cutoff_metrics(y, p, similarities):
    rows = []
    for threshold in THRESHOLDS:
        keep = similarities >= threshold
        row = dict(threshold=float(threshold), n_retained=int(keep.sum()),
                   coverage=float(keep.mean()), n_positive=int(y[keep].sum()))
        row.update(metrics(y[keep], p[keep]) if keep.any() else
                   dict(pr_auc=None, roc_auc=None, brier=None, log_loss=None, ece=None))
        rows.append(row)
    return rows


def select_training_cutoff(X, y, scaffolds, groups, fingerprints, seed):
    """Every selection observation is absent from its classifier AND calibrator."""
    cv = models.calibration_splits(y, scaffolds, seed=seed + 100, groups=groups)
    p, similarity = np.full(len(y), np.nan), np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    for fold, (fit, heldout) in enumerate(cv):
        fitted, _ = models.fit_calibrated(X[fit], y[fit], scaffolds[fit],
                                         seed=seed, groups=groups[fit])
        p[heldout] = fitted.predict_proba(X[heldout])[:, 1]
        similarity[heldout], _ = chemistry.nearest_neighbor(
            [fingerprints[i] for i in heldout], [fingerprints[i] for i in fit])
        fold_id[heldout] = fold
    if not np.isfinite(p).all():
        raise RuntimeError('Incomplete training selection predictions')
    curve = pd.DataFrame(cutoff_metrics(y, p, similarity))
    eligible = curve.loc[(curve.coverage >= .5) & (curve.n_retained >= 50) & (curve.n_positive >= 5)]
    chosen = float(eligible.sort_values(['brier', 'threshold']).iloc[0].threshold)
    return chosen, p, similarity, fold_id, curve


def make_figures(predictions, results, curves):
    colors = {'lgbm_raw_ensemble': '#a86a31', 'lgbm_sigmoid_ensemble': '#126b78',
              'logistic_sigmoid_ensemble': '#7757a2'}
    titles={'lgbm_raw_ensemble':'LightGBM raw ensemble', 'lgbm_sigmoid_ensemble':'LightGBM sigmoid ensemble',
            'logistic_sigmoid_ensemble':'Logistic sigmoid ensemble'}
    for protocol in PROTOCOLS:
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        for col, (name, color) in enumerate(colors.items()):
            for seed in SEEDS:
                frame = predictions.query('protocol == @protocol and model == @name and seed == @seed')
                rel = models.reliability_curve(frame.y, frame.prediction).query('n > 0')
                axes[0,col].plot(rel.p_mean, rel.frac_positives, '-', alpha=.45, color=color, linewidth=.9)
                axes[0,col].scatter(rel.p_mean,rel.frac_positives,s=5+2*np.sqrt(rel.n),alpha=.6,color=color)
                precision, recall, _ = precision_recall_curve(frame.y, frame.prediction)
                axes[1,col].plot(recall,precision,color=color,alpha=.55,linewidth=1)
            axes[0,col].plot([0,1],[0,1],'--',color='#aaa',linewidth=1)
            axes[0,col].set(xlabel='Mean estimated P(EFSA positive label)',title=titles[name],xlim=(0,1),ylim=(0,1))
            axes[1,col].set(xlabel='Recall',xlim=(0,1),ylim=(0,1))
            for row in [0,1]:axes[row,col].grid(alpha=.15)
        axes[0,0].set_ylabel('Observed positive label fraction')
        axes[1,0].set_ylabel('Precision')
        fig.suptitle('Reliability (top) and precision–recall (bottom): five outer splits per panel. Markers scale with bin count.',fontsize=11)
        fig.savefig(DEST/f'calibration_{protocol}.png',dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, metric in zip(axes, ['brier', 'log_loss', 'pr_auc']):
        for policy_idx, protocol in enumerate(PROTOCOLS):
            for model_idx, name in enumerate(colors):
                values = results.query('protocol == @protocol and model == @name')[metric].to_numpy()
                x = model_idx + .19 * (policy_idx - .5)
                ax.scatter(x + np.linspace(-.04, .04, len(values)), values, color=colors[name],
                           marker='o' if policy_idx == 0 else 'x', alpha=.65)
                ax.plot([x - .06, x + .06], [np.median(values)] * 2, color=colors[name], linewidth=3)
        ax.set(title=metric + (' (lower is better)' if metric != 'pr_auc' else ' (average precision)'))
        ax.set_xticks(range(3), ['LGBM raw\nensemble', 'LGBM sigmoid\nensemble', 'Logistic sigmoid\nensemble'], fontsize=8)
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('Five seeds: dots = individual acyclic groups; crosses = acyclic-series robustness')
    fig.savefig(DEST / 'validation_metrics.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for protocol in PROTOCOLS:
        for seed in SEEDS:
            frame = curves.query('protocol == @protocol and seed == @seed')
            axes[0].plot(frame.threshold, frame.coverage, alpha=.45, label=protocol if seed == 0 else None)
            axes[1].plot(frame.threshold, frame.brier, alpha=.45, label=protocol if seed == 0 else None)
    axes[0].set(ylabel='Fraction of outer test structures retained', ylim=(0, 1))
    axes[1].set(ylabel='Brier of retained outer test structures')
    for ax in axes:
        ax.set(xlabel='Exploratory nearest-training Tanimoto cutoff')
        ax.axvline(.4, ls='--', color='#aaa', linewidth=1)
        ax.grid(alpha=.15)
        ax.legend(fontsize=7)
    fig.suptitle('Held-out sensitivity: 0.40 is not a validated reliability boundary')
    fig.savefig(DEST / 'similarity_validation.png', dpi=180)
    plt.close(fig)


def main():
    DEST.mkdir(exist_ok=True)
    out = DEST / 'splits'
    out.mkdir(exist_ok=True)
    print('Preparing standardized identities with pinned dependencies', flush=True)
    with rdBase.BlockLogs():
        raw = labels.aggregate_by_substance(data.with_structure_only(data.load_efsa()),
                                            key='inchikey', rule='any_positive', ambiguous='exclude')
        standardized = chemistry.standardize_table(raw)
        parents = labels.aggregate_standardized(standardized, rule='any_positive')
        train = parents.loc[parents.y.notna()].reset_index(drop=True)
        fps = chemistry.fingerprints(train.smiles_std, radius=2, n_bits=2048)
        X = chemistry.matrix(fps)
        scaffolds = train.smiles_std.map(chemistry.scaffold).to_numpy()
    keys, y = train.inchikey_std.to_numpy(), train.y.astype(int).to_numpy()
    train.assign(scaffold=scaffolds).to_parquet(DEST / 'modelable_identities.parquet', index=False)
    rows, frames, audits, cutoff_rows, selected_rows, reliability_rows = [], [], [], [], [], []
    parameters = {}
    for protocol, series_threshold in PROTOCOLS.items():
        groups = splitting.identity_groups(scaffolds, keys, fps, series_threshold)
        save_json(DEST / f'groups_{protocol}.json', dict(zip(keys, groups)))
        for seed, tr, te in splitting.multiseed_splits(groups, SEEDS):
            print(f'{protocol}: seed {seed}, {len(tr)} train / {len(te)} test', flush=True)
            if len(np.unique(y[tr])) != 2 or len(np.unique(y[te])) != 2:
                raise RuntimeError('Both labels required in outer train and test')
            overlaps = [set(keys[tr]) & set(keys[te]),
                        (set(scaffolds[tr]) - {''}) & (set(scaffolds[te]) - {''}),
                        set(groups[tr]) & set(groups[te])]
            if any(overlaps):
                raise RuntimeError('Train/test identity or group leakage')
            inner = models.calibration_splits(y[tr], scaffolds[tr], seed=seed, groups=groups[tr])
            save_json(out / f'{protocol}_{seed}.json', dict(protocol=protocol, seed=seed,
                train_identities=keys[tr].tolist(), test_identities=keys[te].tolist(),
                inner_folds=[dict(fold=i, fit_identities=keys[tr][fit].tolist(),
                                  calibration_identities=keys[tr][cal].tolist())
                             for i, (fit, cal) in enumerate(inner)]))
            similarity, _ = chemistry.nearest_neighbor([fps[i] for i in te], [fps[i] for i in tr])
            acyclic_train, acyclic_test = [i for i in tr if not scaffolds[i]], [i for i in te if not scaffolds[i]]
            related_count = 0
            if acyclic_train and acyclic_test:
                related, _ = chemistry.nearest_neighbor([fps[i] for i in acyclic_test], [fps[i] for i in acyclic_train])
                related_count = int(sum(related >= .70))
            if series_threshold is not None and related_count:
                raise RuntimeError('An acyclic Tc >= .70 series crossed a robustness split')
            audits.append(dict(protocol=protocol, seed=seed, n_train=len(tr), n_test=len(te),
                identity_overlap=len(overlaps[0]), ring_scaffold_overlap=len(overlaps[1]), group_overlap=len(overlaps[2]),
                n_acyclic_test=len(acyclic_test), n_acyclic_test_with_train_neighbour_ge_070=related_count))
            raw_single = models.build('lgbm', seed).set_params(n_jobs=2).fit(X[tr], y[tr])
            probabilities = {'prior': np.full(len(te), y[tr].mean()),
                             'lgbm_raw_single': raw_single.predict_proba(X[te])[:, 1]}
            for base in ('lgbm', 'logistic'):
                fitted, _ = models.fit_calibrated(X[tr], y[tr], scaffolds[tr], seed=seed,
                                                 groups=groups[tr], model_name=base)
                parameters[base] = fitted.estimator.get_params()
                probabilities[f'{base}_raw_ensemble'] = models.matched_raw_probability(fitted, X[te])
                probabilities[f'{base}_sigmoid_ensemble'] = fitted.predict_proba(X[te])[:, 1]
            for model, p in probabilities.items():
                rows.append(dict(protocol=protocol, seed=seed, model=model, **metrics(y[te], p)))
                frames.append(pd.DataFrame(dict(protocol=protocol, seed=seed, model=model,
                    inchikey_std=keys[te], y=y[te], prediction=p, nearest_training_tanimoto=similarity)))
                reliability_rows.append(models.reliability_curve(y[te], p).assign(protocol=protocol, seed=seed, model=model))
            cutoff_rows.extend(dict(protocol=protocol, seed=seed, **r) for r in
                               cutoff_metrics(y[te], probabilities['lgbm_sigmoid_ensemble'], similarity))
            print('  Selecting cutoff from nested outer-TRAIN OOF predictions only', flush=True)
            chosen, oof, sim_oof, fold_id, curve = select_training_cutoff(
                X[tr], y[tr], scaffolds[tr], groups[tr], [fps[i] for i in tr], seed)
            pd.DataFrame(dict(inchikey_std=keys[tr], y=y[tr], prediction=oof,
                              nearest_fit_tanimoto=sim_oof, oof_fold=fold_id)).to_parquet(
                                  out / f'{protocol}_{seed}_cutoff_training_oof.parquet', index=False)
            curve.to_csv(out / f'{protocol}_{seed}_cutoff_training_curve.csv', index=False)
            keep = similarity >= chosen
            selected_rows.append(dict(protocol=protocol, seed=seed, chosen_cutoff=chosen,
                selection_scope='outer training OOF only', coverage=float(keep.mean()),
                **(metrics(y[te][keep], probabilities['lgbm_sigmoid_ensemble'][keep]) if keep.any() else dict(n_test=0))))
    results, predictions, curves = pd.DataFrame(rows), pd.concat(frames, ignore_index=True), pd.DataFrame(cutoff_rows)
    results.to_csv(DEST / 'calibration_validation.csv', index=False)
    predictions.to_parquet(DEST / 'heldout_predictions.parquet', index=False)
    pd.DataFrame(audits).to_csv(DEST / 'split_overlap_audit.csv', index=False)
    curves.to_csv(DEST / 'similarity_validation.csv', index=False)
    pd.DataFrame(selected_rows).to_csv(DEST / 'training_selected_cutoffs.csv', index=False)
    pd.concat(reliability_rows, ignore_index=True).to_csv(DEST / 'reliability_bins.csv', index=False)
    summaries = []
    for (protocol, model), frame in results.groupby(['protocol', 'model'], sort=False):
        record = dict(protocol=protocol, model=model, n_splits=len(frame))
        for column in ('pr_auc', 'roc_auc', 'brier', 'log_loss', 'ece'):
            q1, median, q3 = frame[column].quantile([.25, .5, .75])
            record[column] = dict(median=float(median), q1=float(q1), q3=float(q3),
                                  minimum=float(frame[column].min()), maximum=float(frame[column].max()))
        summaries.append(record)
    versions = {name: importlib.metadata.version(name) for name in (
        'lightgbm', 'rdkit', 'numpy', 'pandas', 'scikit-learn', 'scipy', 'matplotlib', 'marimo', 'pyarrow')}
    report = dict(schema_version=2, endpoint='aggregated any-positive EFSA conclusion label',
        claim='transparent compound-identity uncertainty screening demonstration',
        n_identities=len(train), n_positive=int(y.sum()), seeds=SEEDS, default_protocol='scaffold_identity',
        protocols={'scaffold_identity': 'Disjoint ring scaffolds; acyclic parents grouped individually with stable InChIKey IDs.',
                   'scaffold_acyclic_series_070': 'Disjoint ring scaffolds and acyclic ECFP4 Tc >= .70 connected components; structure-only robustness protocol.'},
        comparison='Matched raw and sigmoid ensembles reuse the exact same three fitted classifiers.',
        calibration='Three StratifiedGroupKFold classifier/calibrator pairs, sigmoid, ensemble=True.',
        cutoff_selection='Minimum training-OOF Brier, >=50% coverage, >=50 retained and >=5 positives. No outer test tuning.',
        interactive_cutoff='0.40 is exploratory, not a validated confidence boundary.',
        external_validation={'status': 'not performed', 'reason': 'No independently sourced compatible endpoint supplied. FCM standard confirmation concerns identity only.'},
        efsa_release=json.loads((DEST / 'source_release_audit.json').read_text()) if (DEST / 'source_release_audit.json').exists() else {'status': 'unverified'},
        versions=versions, python=platform.python_version(), parameters=parameters, summary=summaries,
        limitations=['Seeds have overlapping test identities; spread is not an independent-sample confidence interval.',
                     'ECE uses ten equal-width bins and depends on binning.',
                     'Filtering changes the test population; lower retained Brier alone is not proof of reliability.',
                     'EFSA conclusions are not a newly validated genotoxicity assay.'])
    save_json(DEST / 'calibration_method.json', report)
    save_json(DEST / 'environment.json', dict(python=sys.version, platform=platform.platform(), versions=versions))
    (DEST / 'resolved_environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
    make_figures(predictions, results, curves)
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in (ROOT / 'src', ROOT / 'scripts', ROOT / 'data') for p in folder.rglob('*')
              if p.is_file() and '__pycache__' not in p.parts}
    save_json(DEST / 'input_code_sha256.json', hashes)
    lines = ['# Generated internal validation', '', 'Reproduce: pinned requirements + `python scripts/validate_calibration.py`.',
             '', 'Endpoint: aggregated positive EFSA label conditional on a supported standardized structure.',
             'Independent compatible endpoint validation has not been performed. Label content matches the verified 2023 OpenFoodTox export (DOI 10.5281/zenodo.8120114); historical download date and enrichment logs were not recorded.',
             '', 'Raw/sigmoid ensembles use the same fitted classifiers. Values: median [Q1–Q3] across five outer seeds.', '']
    for protocol in PROTOCOLS:
        lines += [f'## {protocol}', '', '| Model | AP (PR area) | Brier | Log loss | ECE |', '|---|---:|---:|---:|---:|']
        for record in summaries:
            if record['protocol'] == protocol:
                cells = [f"{record[c]['median']:.4f} [{record[c]['q1']:.4f}–{record[c]['q3']:.4f}]" for c in ('pr_auc', 'brier', 'log_loss', 'ece')]
                lines.append('| ' + record['model'] + ' | ' + ' | '.join(cells) + ' |')
        lines += ['', f'![Reliability and PR](calibration_{protocol}.png)', '']
    lines += ['![Split variability](validation_metrics.png)', '', '![Similarity sensitivity](similarity_validation.png)', '',
              'Saved: held-out predictions, train/test and inner fold identity lists, overlap audits, training-only cutoff OOF scores, and chosen-cutoff held-out metrics.']
    (DEST / 'VALIDATION.md').write_text('\n'.join(lines) + '\n')
    print('Validation complete', flush=True)


if __name__ == '__main__':
    main()
