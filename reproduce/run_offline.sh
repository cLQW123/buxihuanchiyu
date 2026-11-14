#!/usr/bin/env bash
# 兼容旧入口名称，统一使用当前完整离线验证流程。
set -euo pipefail
task_repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
exec bash "$task_repo/reproduce/validate_offline.sh" "$@"
