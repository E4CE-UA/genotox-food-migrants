"""Verify frozen official label export against both derived input tables."""
import pandas as pd,re,json,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
raw_path=ROOT/'data/source_exports/Genotoxicity_KJ_2023.xlsx'
source=pd.read_excel(raw_path)
augmented=pd.read_excel(ROOT/'data/genotoxicity_cas_cid.xlsx')
resolved=pd.read_csv(ROOT/'data/efsa_estructuras.csv')
common=['Substance','Author','Year','OutputID','Genotoxicity']
def decoded(v):
    if pd.isna(v):return '<MISSING>'
    if isinstance(v,(int,float)):return format(v,'.15g')
    return re.sub(r'_x([0-9A-Fa-f]{4})_',lambda m:chr(int(m.group(1),16)),str(v)).strip()
def records(frame):
    return [tuple(decoded(value) for value in row) for row in frame[common].itertuples(index=False,name=None)]
a,b,c=records(source),records(augmented),records(resolved)
matched_ab=sum((Counter(a)&Counter(b)).values())
matched_ac=sum((Counter(a)&Counter(c)).values())
assert Counter(a)==Counter(b)==Counter(c)
result=dict(schema_version=1,status='content_matched_reference_export',
    reference_export=dict(doi='10.5281/zenodo.8120114',version=6,publication_date='2023-09-13',
                          filename=raw_path.name,url='https://zenodo.org/records/8120114/files/Genotoxicity_KJ_2023.xlsx?download=1',
                          zenodo_md5='cc6ad4a2fe03965874910fb19b588655',
                          downloaded_md5=hashlib.md5(raw_path.read_bytes()).hexdigest(),
                          sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),retrieved_on='2026-10-04'),
    verification=dict(fields=common,normalization='One-pass Excel _xHHHH_ decode, trim strings, numeric values to 15 significant digits, explicit missing marker',
                      official_rows=len(a),augmented_rows=len(b),resolved_rows=len(c),
                      label_row_multiset_matches_augmented=matched_ab,label_row_multiset_matches_resolved=matched_ac,
                      augmented_preserves_row_order=a==b,resolved_preserves_row_order=a==c),
    historical_download_release='not recorded; content equality does not identify the original download date',
    structure_join=dict(input='data/genotoxicity_cas_cid.xlsx',output='data/efsa_estructuras.csv',script='src/resolver.py',
                       resolution_precedence=['prepopulated PubChem_CID','validated CAS to PubChem CID','decoded substance name to PubChem CID'],
                       row_alignment_verified=True,endpoint_fields_preserved=True,
                       counts=resolved.fuente_cid.fillna('unresolved').value_counts().to_dict(),
                       limitations=['Prepopulated CAS and PubChem_CID acquisition logs are not bundled.',
                                    'A name-resolved CID is not analytical confirmation of identity.',
                                    'Source and resolved structures are screened by the common model representation policy.']))
(ROOT/'docs/source_release_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
