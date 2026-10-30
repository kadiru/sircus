#!/bin/sh
# Start the standalone local simulation; the Viam module has a separate launcher.
set -eu
SIRCUS_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$SIRCUS_ROOT"
SIRCUS_SERVER="$SIRCUS_ROOT/.runtime/vendor/viam-server/1.14.3/bin/viam-server"
if [ ! -x "$SIRCUS_SERVER" ]; then
  echo "Missing local viam-server binary: $SIRCUS_SERVER" >&2
  exit 1
fi
if [ ! -f "$SIRCUS_ROOT/viam-config.local.json" ]; then
  "$SIRCUS_ROOT/.venv/bin/python" "$SIRCUS_ROOT/build_viam_config.py"
fi
"$SIRCUS_ROOT/.venv/bin/python" - <<'PY'
import socket
from viam_module import check_simulation_config
check_simulation_config()
with socket.socket() as connection:
    if connection.connect_ex(('127.0.0.1', 8080)) == 0:
        raise SystemExit('Port 8080 is already serving; reuse the existing simulation.')
PY
mkdir -p "$SIRCUS_ROOT/data"
export DYLD_LIBRARY_PATH="$SIRCUS_ROOT/.runtime/vendor/x264/r3222/lib:/opt/homebrew/opt/nlopt-static/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
exec "$SIRCUS_SERVER" --config "$SIRCUS_ROOT/viam-config.local.json" --log-file "$SIRCUS_ROOT/data/viam-server.log"
