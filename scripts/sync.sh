#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
action=${1:-status}
case "$action" in
 status) git fetch origin; git status --short --branch; git log -1 --oneline ;;
 pull|push)
  [[ -z $(git status --porcelain) ]] || { echo 'Working tree has changes. Commit/review them first; nothing was changed.' >&2; exit 1; }
  branch=$(git symbolic-ref --quiet --short HEAD) || { echo 'Detached HEAD; choose a task branch first.' >&2; exit 1; }
  if [[ "$action" == pull ]]; then git pull --ff-only; else git push -u origin "$branch"; fi ;;
 *) echo 'Usage: scripts/sync.sh status|pull|push' >&2; exit 2 ;;
esac
