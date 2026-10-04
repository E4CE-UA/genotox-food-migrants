"""Recompute the documented demonstration from canonical package functions."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from genotox_food_migrants.demo_engine import build_demo
from genotox_food_migrants.demo_view import build_demo_view

def main():
    demo=build_demo(ROOT/'notebooks/what_if_wrong_compound.py')
    scored,features,provenance=demo.prepare()
    results={}
    for mode in ['published','same_formula']:
        for cutoff in [.4,.5,.9]:
            view=demo.evaluate(scored,features,demo.HOOK_FEATURE,cutoff,mode)
            results[f'{mode}_{cutoff}']={k:view[k] for k in ['n_retained','n_evaluable','conditional_score',
                'covered_weight','listed_formula_pool_size','retained_pool_fraction']}
            json.loads(demo.snapshot(view,provenance))
            assert 'Scenario average over retained structures' in build_demo_view().scenario_summary(view)
    report=dict(results=results,pool_audit=provenance['sampled_pool_audit'],
        assigned_in_training=bool(demo.evaluate(scored,features,demo.HOOK_FEATURE,.4,'published')['assigned']['in_training']),
        model_fit_token=provenance['model_fit_token'])
    (ROOT/'docs/case_results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
