#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$TASK_ROOT/sircus/two-arm"
VIAM_BIN=${SIRCUS_VIAM_SERVER:-viam-server}
if ! command -v "$VIAM_BIN" >/dev/null; then
 if [[ -x /opt/viam/bin/viam-server ]]; then VIAM_BIN=/opt/viam/bin/viam-server
 else echo 'Install the Linux Viam runtime first; see docs/UBUNTU_SETUP.md.' >&2; exit 1; fi
fi
../simulation/.venv/bin/python - <<'PY'
import socket
from build_config import expected_config
import json
from pathlib import Path
with socket.socket() as s:
 if s.connect_ex(('127.0.0.1',8083))==0:raise SystemExit('Port8083 already occupied; reuse or inspect the existing process.')
Path('viam-config.test.json').write_text(json.dumps(expected_config(test=True),indent=2)+'\n')
Path('data').mkdir(exist_ok=True)
PY
exec "$VIAM_BIN" --config "$PWD/viam-config.test.json" --log-file "$PWD/data/viam-local-test.log"
