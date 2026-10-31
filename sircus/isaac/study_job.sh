#!/usr/bin/env bash
# Bounded single-GPU job. Run only in an explicitly provisioned simulation container.
set -euo pipefail
: "${SIRCUS_ISAAC_SOURCE:?Point to an immutable copied source directory}"
layouts=${1:-1000}
episodes=${2:-4096}
seed=${3:-42}
[[ "$layouts" =~ ^[0-9]+$ && "$episodes" =~ ^[0-9]+$ && "$seed" =~ ^[0-9]+$ ]]
mkdir -p /workspace/logs
exec 9>/workspace/scene.lock
flock -n 9 || exit 75
set +e
timeout --signal=TERM --kill-after=30s 6h \
 "$SIRCUS_ISAAC_SOURCE/python.sh" "$SIRCUS_ISAAC_SOURCE/launch_scene.py" \
 --headless --enable_cameras --device cuda:0 --general-episodes "$episodes" \
 --train-layouts "$layouts" --rl-seed "$seed" \
 > "/workspace/logs/study-n${layouts}-seed${seed}.log" 2>&1
rc=$?
set -e
if [[ $rc != 0 ]]; then
 python3 - "$rc" <<'PY'
import datetime,json,pathlib,sys
p=pathlib.Path('/workspace/output/learning-study.json')
v=json.loads(p.read_text()) if p.exists() else {}
v.update(phase='failed',exit_code=int(sys.argv[1]),timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat())
t=p.with_suffix('.tmp');t.write_text(json.dumps(v));t.replace(p)
PY
fi
exit "$rc"
