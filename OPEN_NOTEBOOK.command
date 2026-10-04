#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if command -v uvx >/dev/null 2>&1; then
    exec uvx --python 3.12 --from 'marimo==0.24.2' marimo edit --sandbox notebooks/what_if_wrong_compound.py
fi
if ! command -v python3.12 >/dev/null 2>&1; then
    echo 'Necesitas Python 3.12 o uv para abrir este entorno fijado. Consulta START_HERE_ES.md.'
    exit 1
fi
if [ ! -x .venv-v0.2/bin/python ]; then
    python3.12 -m venv .venv-v0.2
fi
.venv-v0.2/bin/python -m pip install -r requirements.txt
exec .venv-v0.2/bin/marimo edit notebooks/what_if_wrong_compound.py
