"""Check saved numeric evidence and fold memberships without refitting."""
import json,sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from validate_calibration import metrics
predictions=pd.read_parquet(ROOT/'docs/heldout_predictions.parquet')
results=pd.read_csv(ROOT/'docs/calibration_validation.csv')
cutoffs=pd.read_csv(ROOT/'docs/training_selected_cutoffs.csv')
checks=0
for row in results.to_dict('records'):
    subset=predictions.loc[(predictions.protocol==row['protocol']) &
                           (predictions.seed==row['seed']) & (predictions.model==row['model'])]
    assert not subset.inchikey_std.duplicated().any()
    measured=metrics(subset.y.to_numpy(),subset.prediction.to_numpy())
    for key,value in measured.items():
        assert np.isclose(row[key],value,atol=1e-12,rtol=0), (key,row,value)
    split=json.loads((ROOT/'docs/splits'/f"{row['protocol']}_{row['seed']}.json").read_text())
    training,test=set(split['train_identities']),set(split['test_identities'])
    assert not training & test
    assert test==set(subset.inchikey_std)
    calibrated=[]
    groups=json.loads((ROOT/'docs'/f"groups_{row['protocol']}.json").read_text())
    assert not {groups[k] for k in training} & {groups[k] for k in test}
    for fold in split['inner_folds']:
        fit,cal=set(fold['fit_identities']),set(fold['calibration_identities'])
        assert not fit&cal and fit|cal==training
        assert not {groups[k] for k in fit} & {groups[k] for k in cal}
        assert not test&(fit|cal)
        calibrated+=fold['calibration_identities']
    assert len(calibrated)==len(set(calibrated))==len(training)
    checks+=1
for row in cutoffs.to_dict('records'):
    curve=pd.read_csv(ROOT/'docs/splits'/f"{row['protocol']}_{row['seed']}_cutoff_training_curve.csv")
    eligible=curve.loc[(curve.coverage>=.5)&(curve.n_retained>=50)&(curve.n_positive>=5)]
    chosen=eligible.sort_values(['brier','threshold']).iloc[0].threshold
    assert np.isclose(chosen,row['chosen_cutoff'])
    oof=pd.read_parquet(ROOT/'docs/splits'/f"{row['protocol']}_{row['seed']}_cutoff_training_oof.parquet")
    split=json.loads((ROOT/'docs/splits'/f"{row['protocol']}_{row['seed']}.json").read_text())
    assert set(oof.inchikey_std)==set(split['train_identities'])
    subset=predictions.loc[(predictions.protocol==row['protocol']) & (predictions.seed==row['seed']) &
                           (predictions.model=='lgbm_sigmoid_ensemble')]
    keep=subset.nearest_training_tanimoto>=chosen
    measured=metrics(subset.loc[keep,'y'].to_numpy(),subset.loc[keep,'prediction'].to_numpy())
    for key,value in measured.items():assert np.isclose(row[key],value,atol=1e-12,rtol=0)
print(json.dumps(dict(passed=True,metric_rows_checked=checks,cutoff_rows_checked=len(cutoffs),
                     predictions=len(predictions),identities_disjoint=True,calibration_groups_disjoint=True,
                     cutoff_selection_uses_training_only=True),indent=2))
