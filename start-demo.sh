#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
TASK_PYTHON=${SIRCUS_PYTHON:-$TASK_ROOT/sircus/simulation/.venv/bin/python}
if ! command -v "$TASK_PYTHON" >/dev/null; then
  echo 'Set SIRCUS_PYTHON to the project Python 3.12 environment or run setup-ubuntu.sh.' >&2
  exit 1
fi
# An independent, loopback-only numerical preview. No cloud/hardware/audio service.
exec "$TASK_PYTHON" - "$TASK_ROOT" "${SIRCUS_DEMO_PORT:-18772}" <<'PY'
import os
import sys
from pathlib import Path
root, port = Path(sys.argv[1]), str(int(sys.argv[2]))
for key in tuple(os.environ):
    if key.startswith(('SIRCUS_AUDIO_', 'VIAM_')):
        os.environ.pop(key)
print(f'Sircus demo: http://127.0.0.1:{port}/demo.html', flush=True)
os.execv(sys.executable, [sys.executable, str(root/'sircus/simulation/server.py'),
    '--offline', '--show', '--physical-mode', 'disabled', '--port', port,
    '--show-data-root', str(root/'.runtime/demo/show'),
    '--workflow-data-root', str(root/'.runtime/demo/workflow')])
PY
