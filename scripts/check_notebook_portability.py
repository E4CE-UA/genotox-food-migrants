"""Run the generated standalone notebook bootstrap away from the checkout.

The frozen package must be selected even when another package was already loaded.
"""
import ast,copy,json,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import genotox_food_migrants as previously_loaded
NOTEBOOK=ROOT/'notebooks/what_if_wrong_compound.py'
source=NOTEBOOK.read_text()
first=next(node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef))
node=copy.deepcopy(first);node.decorator_list=[]
with tempfile.TemporaryDirectory(prefix='genotox_portability_') as folder:
    copy_path=Path(folder)/'what_if_wrong_compound.py';copy_path.write_text(source)
    namespace={'__file__':str(copy_path)}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(copy_path),'exec'),namespace)
    namespace['_']()
    import genotox_food_migrants as frozen
    from genotox_food_migrants.demo_engine import build_demo
    assert 'genotox_frozen_' in str(frozen.__file__)
    assert frozen.__file__ != previously_loaded.__file__
    demo=build_demo(copy_path);scored,features,provenance=demo.prepare()
    expected=json.loads((ROOT/'docs/case_results.json').read_text())
    checks=[]
    for key,wanted in expected['results'].items():
        mode,cutoff=key.rsplit('_',1)
        view=demo.evaluate(scored,features,demo.HOOK_FEATURE,float(cutoff),mode)
        actual=view['conditional_score'];target=wanted['conditional_score']
        assert actual is None if target is None else abs(actual-target)<1e-10
        assert view['n_evaluable']==wanted['n_evaluable']
        checks.append(key)
    assert provenance['n_training_substances']==1892
    assert provenance['model_fit_token']==expected['model_fit_token']
    print(json.dumps(dict(passed=True,standalone_snapshot=True,previous_namespace_replaced=True,
                         checks=checks,training_identities=provenance['n_training_substances']),indent=2))
