#!/usr/bin/env bash
# Run inside the isolated Sircus container; never sources a hardware environment.
set -euo pipefail
export PYTHONEXE=/workspace/venv/bin/python
SIRCUS_SOURCE_ROOT=${SIRCUS_ISAAC_SOURCE:-/workspace/source/sircus/isaac}
export PYTHONPATH=/workspace/venv/lib/python3.11/site-packages:/workspace/vendor/IsaacLab/source/isaaclab:$SIRCUS_SOURCE_ROOT${PYTHONPATH:+:$PYTHONPATH}
export OMNI_KIT_ALLOW_ROOT=1
# No Nucleus/cloud cache service is needed for these local assets.
export OMNICLIENT_HUB_MODE=disabled
exec /isaac-sim/python.sh "$@"
