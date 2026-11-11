#!/usr/bin/env bash
# Run in the network-enabled preparation environment, then transfer the checkout.
set -euo pipefail
sgcup_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
git -C "$sgcup_root" submodule update --init \
    Projects/InfiniCore Projects/InfiniLM Projects/ninetoothed Projects/ntops
git -C "$sgcup_root/Projects/InfiniCore" submodule update --init third_party/spdlog
git -C "$sgcup_root/Projects/InfiniLM" submodule update --init third_party/spdlog third_party/json
python - "$sgcup_root" <<'PY'
import json
from pathlib import Path
import subprocess
import sys

root = Path(sys.argv[1])
lock = json.loads((root / 'source-lock.json').read_text())
for name, record in lock['projects'].items():
    actual = subprocess.check_output(
        ['git', '-C', str(root / 'Projects' / name), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == record['commit'], (name, actual, record['commit'])
print('PINNED_SOURCE_CHECKOUT_READY')
PY
