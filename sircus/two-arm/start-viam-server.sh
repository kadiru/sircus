#!/bin/sh
set -eu
SIRCUS_TWO_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SIRCUS_TWO_RUNTIME="$SIRCUS_TWO_ROOT/../simulation"
cd "$SIRCUS_TWO_ROOT"
"$SIRCUS_TWO_RUNTIME/.venv/bin/python" - <<'PY'
import socket
with socket.socket() as connection:
    if connection.connect_ex(('127.0.0.1', 8082)) == 0:
        raise SystemExit('Port 8082 is already serving; reuse the existing cloud simulation server.')
PY
"$SIRCUS_TWO_RUNTIME/.venv/bin/python" cloud_guard.py
export DYLD_LIBRARY_PATH="$SIRCUS_TWO_RUNTIME/.runtime/vendor/x264/r3222/lib:/opt/homebrew/opt/nlopt-static/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
exec "$SIRCUS_TWO_RUNTIME/.runtime/vendor/viam-server/1.14.3/bin/viam-server" --config "$SIRCUS_TWO_ROOT/../viam-cloud/.private/viam-connection.json" --log-file "$SIRCUS_TWO_ROOT/data/viam-server.log"
