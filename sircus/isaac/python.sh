#!/usr/bin/env bash
# Run inside the isolated Sircus container; never sources a hardware environment.
set -euo pipefail
export PYTHONEXE=/workspace/venv/bin/python
export PYTHONPATH=/workspace/venv/lib/python3.11/site-packages:/workspace/vendor/IsaacLab/source/isaaclab:/workspace/source/sircus/isaac${PYTHONPATH:+:$PYTHONPATH}
export OMNI_KIT_ALLOW_ROOT=1
# No Nucleus/cloud cache service is needed for these local assets.
export OMNICLIENT_HUB_MODE=disabled
exec /isaac-sim/python.sh "$@"
