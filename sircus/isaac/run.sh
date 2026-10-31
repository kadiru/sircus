#!/usr/bin/env bash
# Run on Ubuntu host. Commands operate ONLY on this dedicated Sircus container.
set -euo pipefail
container=${SIRCUS_CONTAINER:-sircus_isaac_4090}
runner=/workspace/source/sircus/isaac/python.sh
scene=/workspace/source/sircus/isaac/launch_scene.py
action=${1:-status}
idle() {
  docker exec -u 0 "$container" bash -lc 'if pgrep -f "^/workspace/venv/bin/python /workspace/source/sircus/isaac/launch_scene.py" >/dev/null; then echo "A Sircus scene is already running; stop it explicitly first."; exit 1; fi'
}
case "$action" in
  train)
    episodes=${2:-256}; seed=${3:-42}
    [[ "$episodes" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ && "$episodes" -gt 0 ]] || { echo 'Use positive episode count and nonnegative seed'; exit 2; }
    idle
    docker exec -d -u 0 "$container" bash -lc 'exec flock -n /workspace/scene.lock /workspace/source/sircus/isaac/python.sh /workspace/source/sircus/isaac/launch_scene.py --headless --enable_cameras --device cuda:0 --rl-episodes "$1" --rl-seed "$2" > /workspace/logs/rl-training.log 2>&1' _ "$episodes" "$seed"
    echo 'RL starting. See output/training.json and logs/rl-training.log.' ;;
  evaluate)
    checkpoint=${2:?Supply checkpoint path inside the Sircus container}; trials=${3:-8}
    [[ "$trials" =~ ^[0-9]+$ && "$trials" -gt 0 ]] || exit 2
    idle
    docker exec -u 0 "$container" flock -n /workspace/scene.lock "$runner" "$scene" --headless --enable_cameras --device cuda:0 --rl-episodes "$trials" --rl-checkpoint "$checkpoint" ;;
  build-assets) docker exec -u 0 "$container" /workspace/source/sircus/isaac/build_assets.sh ;;
  smoke) docker exec -u 0 "$container" flock -n /workspace/scene.lock "$runner" "$scene" --headless --enable_cameras --device cuda:0 --frames 180 --capture-once --verify-physics
    docker exec -u 0 "$container" /workspace/venv/bin/python -c 'import json; r=json.load(open("/workspace/output/result.json")); assert r["passed"] and r["physics_verified"],r; print(r)' ;;
  start)
    docker exec -u 0 "$container" bash -lc 'if pgrep -f "^/workspace/venv/bin/python /workspace/source/sircus/isaac/launch_scene.py" >/dev/null; then echo "Sircus scene is already running"; exit 1; fi'
    docker exec -d -u 0 "$container" bash -lc 'exec flock -n /workspace/scene.lock /workspace/source/sircus/isaac/python.sh /workspace/source/sircus/isaac/launch_scene.py --headless --enable_cameras --device cuda:0 --frames 0 > /workspace/logs/live.log 2>&1'
    echo 'Sircus scene starting; check status and logs/live.log.' ;;
  status)
    docker inspect --format '{{.Name}} {{.State.Status}} GPU={{json .HostConfig.DeviceRequests}}' "$container"
    docker exec -u 0 "$container" sh -c 'pgrep -af "^/workspace/venv/bin/python /workspace/source/sircus/isaac/launch_scene.py" || true'
    docker exec -u 0 "$container" sh -c 'test ! -f /workspace/output/status.json || cat /workspace/output/status.json' ;;
  stop)
    docker exec -u 0 "$container" sh -c 'pkill -TERM -f "^/workspace/venv/bin/python /workspace/source/sircus/isaac/launch_scene.py" || test "$?" = 1'
    for attempt in {1..10}; do
      if ! docker exec -u 0 "$container" pgrep -f '^/workspace/venv/bin/python /workspace/source/sircus/isaac/launch_scene.py' >/dev/null; then break; fi
      sleep 1
    done
    docker exec -u 0 "$container" sh -c 'pkill -KILL -f "^/workspace/venv/bin/python /workspace/source/sircus/isaac/launch_scene.py" || test "$?" = 1'
    echo 'Only the Sircus scene was stopped; container and other workloads remain.' ;;
  *) echo 'Usage: run.sh {build-assets|smoke|start|status|stop|train [episodes seed]|evaluate checkpoint [trials]}' >&2; exit 2 ;;
esac
