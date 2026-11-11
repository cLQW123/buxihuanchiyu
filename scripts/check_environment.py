"""Fail early when the offline competition dependencies do not match the lock."""
import importlib.metadata
import json
from pathlib import Path
import sys

import torch

root = Path(__file__).resolve().parent.parent
expected = json.loads((root / 'dependencies/runtime-lock.json').read_text())
records = {}
for name, version in expected['environment']['packages'].items():
    if name == 'ntops':
        continue
    actual = importlib.metadata.version(name)
    assert actual == version, (name, actual, version)
    records[name] = actual
assert sys.version_info[:2] == (3, 11), sys.version
assert torch.version.hip and torch.cuda.device_count() == 8
from diffusers.modular_pipelines import minimax_h3
print('OFFLINE_COMPETITION_ENVIRONMENT_PASSED', json.dumps({
    'python': sys.version, 'hip': torch.version.hip, 'packages': records,
    'devices': [torch.cuda.get_device_name(index) for index in range(8)],
    'diffusers_module': minimax_h3.__file__}), flush=True)
