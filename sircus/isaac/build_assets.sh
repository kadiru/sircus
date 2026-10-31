#!/usr/bin/env bash
# Inside Sircus container (as root); assets remain in ignored local runtime storage.
set -euo pipefail
runner=/workspace/source/sircus/isaac/python.sh
"$runner" /workspace/source/sircus/isaac/prepare_assets.py
for role in board_arm dropper; do
  "$runner" /workspace/vendor/IsaacLab/scripts/tools/convert_urdf.py \
    "/workspace/assets/$role.urdf" "/workspace/assets/$role/$role.usd" \
    --fix-base --headless --device cuda:0
  test -s "/workspace/assets/$role/$role.usd"
done
