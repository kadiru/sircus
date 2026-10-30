#!/bin/sh
set -eu
# Executed by viam-server only after separate installation/commissioning approval.
# Viam passes its module socket arguments through unchanged.
DEPLOYMENT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
SIRCUS_REPO=$(CDPATH= cd -- "$DEPLOYMENT_DIR/../.." && pwd)
cd "$SIRCUS_REPO"
exec "${SIRCUS_MODULE_PYTHON:-/opt/sircus/.venv/bin/python}" -m sircus.deployment.viam_module "$@"
