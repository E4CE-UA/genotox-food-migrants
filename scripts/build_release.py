"""Generate a checksummed frozen local package in the complete marimo notebook.

No hand-maintained fallback functions and no moving Git dependency. Rebuild after
changing canonical modules, bundled inputs or generated validation outputs.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/what_if_wrong_compound.py'


def package_files():
    result = {}
    for path in sorted((ROOT / 'src').glob('*.py')):
        result['genotox_food_migrants/' + path.name] = path.read_bytes()
    for source, destination in [('data', '_data'), ('docs', '_validation')]:
        for path in sorted((ROOT / source).rglob('*')):
            if (path.is_file() and '__pycache__' not in path.parts
                and not set(path.parts) & {'ui_checks', 'previous_v0_1'}
                and path.name not in {'release_manifest.json', 'verification_v0_2.json'}):
                result['genotox_food_migrants/' + destination + '/' + str(path.relative_to(ROOT / source))] = path.read_bytes()
    return result


def build():
    files = package_files()
    manifest = dict(package='genotox-food-migrants', version='0.2.0', format=1,
                    source_policy='Frozen local package generated from canonical modules and bundled data',
                    sha256={name: hashlib.sha256(content).hexdigest() for name, content in files.items()})
    files['genotox_food_migrants/_release_manifest.json'] = json.dumps(manifest, sort_keys=True, indent=2).encode()
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, content in sorted(files.items()):
            member = zipfile.ZipInfo(name, date_time=(2026, 10, 4, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o644 << 16
            archive.writestr(member, content, compresslevel=9)
    blob = stream.getvalue()
    digest = hashlib.sha256(blob).hexdigest()
    source = NOTEBOOK.read_text()
    source, n = re.subn(r'    _FROZEN_PACKAGE_B64 = "[A-Za-z0-9+/=]*"',
                       '    _FROZEN_PACKAGE_B64 = "' + base64.b64encode(blob).decode() + '"', source)
    source, n_hash = re.subn(r'    _FROZEN_PACKAGE_SHA256 = "[a-f0-9]*"',
                            '    _FROZEN_PACKAGE_SHA256 = "' + digest + '"', source)
    if n != 1 or n_hash != 1:
        raise RuntimeError('Notebook bootstrap markers missing or duplicated')
    ast.parse(source)
    NOTEBOOK.write_text(source)
    return dict(capsule_sha256=digest, capsule_bytes=len(blob), n_members=len(files),
                notebook_sha256=hashlib.sha256(source.encode()).hexdigest(), package_version='0.2.0')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    receipt = build()
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))
