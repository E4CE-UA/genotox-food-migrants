"""Generate README numbers from the current validation and case outputs."""
import json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
report=json.loads((ROOT/'docs/calibration_method.json').read_text())
case=json.loads((ROOT/'docs/case_results.json').read_text())
lines=['## Generated internal performance','', '| Protocol / model | AP | Brier | Log loss | ECE |','|---|---:|---:|---:|---:|']
for entry in report['summary']:
    if entry['model'] not in ['lgbm_raw_ensemble','lgbm_sigmoid_ensemble','logistic_sigmoid_ensemble']:continue
    cells=[f"{entry[c]['median']:.4f} [{entry[c]['q1']:.4f}–{entry[c]['q3']:.4f}]" for c in ['pr_auc','brier','log_loss','ece']]
    lines.append('| '+entry['protocol']+' / '+entry['model']+' | '+' | '.join(cells)+' |')
lines += ['', 'Median [Q1–Q3] across five outer seeds; internal evaluation. All individual results and figures: `docs/VALIDATION.md`.']
r=case['results'];p=r['published_0.4'];a=r['same_formula_0.4'];b=r['same_formula_0.5']
case_lines=[f"Generated with {report['n_identities']:,} standardized training parents and {report['n_positive']} positive labels. Assigned identity in training: **{case['assigned_in_training']}**.",'',
 '| Identity scenario | Tc rule | Included scenario weight | Structures meeting rule | Scenario average |',
 '|---|---:|---:|---:|---:|']
for label,key in [('Assigned only','published_0.4'),('20 equal-weight structures','same_formula_0.4'),('20 equal-weight structures','same_formula_0.5'),('20 equal-weight structures','same_formula_0.9')]:
    v=r[key];avg=f"{100*v['conditional_score']:.2f}%" if v['conditional_score'] is not None else 'Unavailable'
    case_lines.append(f"| {label} | {key.split('_')[-1]} | {100*v['covered_weight']:.0f}% | {v['n_evaluable']}/{v['n_retained']} | {avg} |")
case_lines += ['',f"Retained set: **{p['n_retained']}/{p['listed_formula_pool_size']:,} listed formula structures ({100*p['retained_pool_fraction']:.2f}%)**. All {case['pool_audit']['n_sampled_assignments']} sets are sampled; median retained fraction **{100*case['pool_audit']['median_retained_fraction']:.2f}%**. This is distinct from coverage of the displayed scenario."]
p=ROOT/'README.md';text=p.read_text()
for key,content in [('validation','\n'.join(lines)),('case','\n'.join(case_lines))]:
    text,n=re.subn(f'<!-- generated-{key}:start -->.*?<!-- generated-{key}:end -->',f'<!-- generated-{key}:start -->\n{content}\n<!-- generated-{key}:end -->',text,flags=re.S)
    if n!=1:raise RuntimeError('README markers missing')
p.write_text(text)
