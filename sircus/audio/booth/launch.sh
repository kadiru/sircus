#!/bin/sh
set -eu
cd "$(dirname "$0")"
mkdir -p runtime/module-cache
TASK_ROOT=$(CDPATH= cd ../../.. && pwd)
AUDIO_PYTHON=${SIRCUS_PYTHON:-$TASK_ROOT/.venv/bin/python}
if [ ! -x "$AUDIO_PYTHON" ]; then AUDIO_PYTHON=/opt/homebrew/bin/python3.12; fi
if [ ! -x "$AUDIO_PYTHON" ]; then AUDIO_PYTHON=python3.12; fi
if [ ! -x runtime/sircus-native ] || [ src/Native.swift -nt runtime/sircus-native ]; then
  xcrun swiftc -swift-version 5 -module-cache-path runtime/module-cache src/Native.swift -o runtime/sircus-native
fi
export PYTHONDONTWRITEBYTECODE=1
exec "$AUDIO_PYTHON" src/bridge.py "$@"
