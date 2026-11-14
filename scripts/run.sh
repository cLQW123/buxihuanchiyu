#!/usr/bin/env bash
set -euo pipefail
task_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$task_repo/build/activate.sh"
exec python "$SGCUP_ROOT/scripts/run.py" "$@"
