#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TASK_ROOT"
if [[ $(uname -s) != Linux ]]; then echo 'This setup script is for Linux. No files changed.' >&2; exit 1; fi
PYTHON=${SIRCUS_PYTHON:-python3.12}
command -v "$PYTHON" >/dev/null || { echo 'Install Python3.12 and python3.12-venv first; see docs/UBUNTU_SETUP.md.' >&2; exit 1; }
"$PYTHON" -m venv sircus/simulation/.venv
sircus/simulation/.venv/bin/python -m pip install -r sircus/simulation/requirements-viam.txt
sircus/simulation/.venv/bin/python sircus/two-arm/build_config.py
mkdir -p sircus/two-arm/data
printf '%s\n' 'Python environment and local Viam config ready.' 'Training can run now without Viam. For the scene/dashboard, install Linux viam-server and run ./start-local.sh.'
