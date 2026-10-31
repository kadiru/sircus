#!/usr/bin/env bash
# Ubuntu host setup. Requires an existing Isaac Sim 5.1 binary installation and
# already-downloaded NVIDIA Isaac container image. No existing containers changed.
set -euo pipefail
: "${SIRCUS_GPU_UUID:?Set the UUID of the dedicated GPU from nvidia-smi}"
SIRCUS_ROOT=${SIRCUS_ROOT:-"$HOME/sircus-isaac"}
SIRCUS_SIM_ROOT=${SIRCUS_SIM_ROOT:-"$HOME/isaac-sim"}
SIRCUS_CONTAINER=${SIRCUS_CONTAINER:-sircus_isaac_4090}
SIRCUS_IMAGE=${SIRCUS_IMAGE:-nvcr.io/nvidia/isaac-sim:6.0.1}
SIRCUS_SOURCE=$(cd "$(dirname "$0")/../.." && pwd)
case "$(cat "$SIRCUS_SIM_ROOT/VERSION")" in 5.1.*) ;; *) echo 'This baseline requires mounted Isaac Sim 5.1.' >&2; exit 1;; esac
if docker container inspect "$SIRCUS_CONTAINER" >/dev/null 2>&1; then
  echo "Container already exists: $SIRCUS_CONTAINER. Inspect/reuse it; this script will not replace it." >&2; exit 1
fi
if [ -e "$SIRCUS_ROOT/vendor/IsaacLab" ] || [ -e "$SIRCUS_ROOT/vendor/xarm_ros2" ]; then
  echo 'Vendor checkout already exists; refusing to overwrite it.' >&2; exit 1
fi
mkdir -p "$SIRCUS_ROOT"/{vendor,cache,kit-cache,kit-data,logs,output,source}
git clone --depth 1 --branch v2.3.2 --filter=blob:none https://github.com/isaac-sim/IsaacLab.git "$SIRCUS_ROOT/vendor/IsaacLab"
test "$(git -C "$SIRCUS_ROOT/vendor/IsaacLab" rev-parse HEAD)" = 37ddf626871758333d6ed89cf64ad702aef127d0
git init "$SIRCUS_ROOT/vendor/xarm_ros2"
git -C "$SIRCUS_ROOT/vendor/xarm_ros2" remote add origin https://github.com/xArm-Developer/xarm_ros2.git
git -C "$SIRCUS_ROOT/vendor/xarm_ros2" sparse-checkout init --cone
git -C "$SIRCUS_ROOT/vendor/xarm_ros2" sparse-checkout set xarm_description
git -C "$SIRCUS_ROOT/vendor/xarm_ros2" fetch --depth 1 --filter=blob:none origin ac736c92b02de7ca7e0be6f737d77d46707464d8
git -C "$SIRCUS_ROOT/vendor/xarm_ros2" checkout --detach FETCH_HEAD
mkdir -p "$SIRCUS_ROOT/source/sircus/isaac"
cp -R "$SIRCUS_SOURCE/sircus/isaac/." "$SIRCUS_ROOT/source/sircus/isaac/"
docker run -d --name "$SIRCUS_CONTAINER" --gpus "device=$SIRCUS_GPU_UUID" --cpus 12 --memory 28g --shm-size 4g --restart no \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=N -e NVIDIA_DRIVER_CAPABILITIES=all -e OMNI_KIT_ALLOW_ROOT=1 \
  -v "$SIRCUS_SIM_ROOT:/isaac-sim:ro" -v "$SIRCUS_ROOT:/workspace" \
  -v "$SIRCUS_ROOT/cache:/root/.cache" -v "$SIRCUS_ROOT/kit-cache:/isaac-sim/kit/cache" \
  -v "$SIRCUS_ROOT/kit-data:/isaac-sim/kit/data" -v "$SIRCUS_ROOT/logs:/root/.nvidia-omniverse/logs" \
  --entrypoint bash "$SIRCUS_IMAGE" -lc 'sleep infinity'
docker exec -u 0 "$SIRCUS_CONTAINER" /isaac-sim/python.sh -m venv --system-site-packages /workspace/venv
docker exec -u 0 "$SIRCUS_CONTAINER" ln -s /isaac-sim /workspace/vendor/IsaacLab/_isaac_sim
docker exec -u 0 "$SIRCUS_CONTAINER" /workspace/source/sircus/isaac/python.sh -m pip install --no-cache-dir --no-build-isolation -c /workspace/source/sircus/isaac/constraints.txt \
  -e /workspace/vendor/IsaacLab/source/isaaclab \
  -e /workspace/vendor/IsaacLab/source/isaaclab_assets \
  -e /workspace/vendor/IsaacLab/source/isaaclab_tasks \
  -e /workspace/vendor/IsaacLab/source/isaaclab_rl \
  xacro==2.1.1 fastapi==0.121.0 docstring-parser==0.16 protobuf==5.29.5 onnx==1.19.1 pytest==8.4.2
