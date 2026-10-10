#!/bin/sh
set -eu

cd "$(dirname "$0")"

PYTHON=
for candidate in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v "$candidate" >/dev/null 2>&1 &&
        "$candidate" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1; then
        PYTHON=$candidate
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "Python 3.9 à 3.12 est requis. Installez une version compatible puis relancez." >&2
    exit 1
fi

if [ ! -x .venv/bin/python ]; then
    if ! "$PYTHON" -m venv .venv; then
        echo "Création de l'environnement impossible. Installez le module avec :" >&2
        echo "sudo apt install python3-venv" >&2
        exit 1
    fi
elif ! .venv/bin/python -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 13) else 1)' >/dev/null 2>&1; then
    echo "L'environnement .venv utilise une version incompatible de Python." >&2
    echo "Supprimez uniquement le dossier .venv puis relancez ce script." >&2
    exit 1
fi

.venv/bin/python -m pip install -r requirements.txt

exec .venv/bin/python dessin_camera.py
