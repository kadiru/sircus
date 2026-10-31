#!/usr/bin/env bash
# Ubuntu host: independent CPU observation workers, no simulator/hardware control.
set -euo pipefail
container=${SIRCUS_CONTAINER:-sircus_isaac_4090}
action=${1:-status}
for worker in perception_worker perception_evaluate; do
  pattern="^/workspace/venv/bin/python /workspace/source/sircus/isaac/${worker}.py --output /workspace/output$"
  case "$action" in
    start)
      if docker exec -u 0 "$container" pgrep -f "$pattern" >/dev/null; then
        echo "$worker already running"
      else
        docker exec -d -u 0 -e OPENBLAS_NUM_THREADS=1 -e OMP_NUM_THREADS=1 "$container" bash -lc \
          'exec /workspace/source/sircus/isaac/python.sh "/workspace/source/sircus/isaac/$1.py" --output /workspace/output > "/workspace/logs/$1.log" 2>&1' _ "$worker"
      fi ;;
    stop) docker exec -u 0 "$container" sh -c 'pkill -TERM -f "$1" || test "$?" = 1' _ "$pattern" ;;
    status) docker exec -u 0 "$container" sh -c 'pgrep -af "$1" || true' _ "$pattern" ;;
    *) echo 'Usage: perception.sh {start|stop|status}' >&2; exit 2 ;;
  esac
done
