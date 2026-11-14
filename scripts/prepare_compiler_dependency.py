"""Expose only the unchanged official Triton package, without its parent overlays."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--triton-source', type=Path,
                    default=Path(__file__).resolve().parent.parent/'inputs/vendor-triton/triton',
                    help='默认使用公开Release解包的完整厂商Triton')
args = parser.parse_args()
source = args.triton_source.resolve(strict=True)
initial = source / '__init__.py'
content = initial.read_bytes()
version = None
for node in ast.parse(content).body:
    if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and
            target.id == '__version__' for target in node.targets):
        version = ast.literal_eval(node.value)
        break
if not isinstance(version, str) or version.split('+')[0] != '3.5.1':
    raise RuntimeError('The competition Triton package must report version 3.5.1: ' + repr(version))
for name in ('compiler', 'runtime', 'language'):
    if not (source / name).is_dir():
        raise RuntimeError('Incomplete official Triton package: ' + str(source / name))
libraries = source.parent / 'triton.libs'
metadata = sorted(path for path in source.parent.glob('triton-3.5.1*.dist-info') if path.is_dir())
if not libraries.is_dir() or len(metadata) != 1:
    raise RuntimeError('Require official sibling triton.libs and one matching triton-3.5.1*.dist-info directory')
root = Path(__file__).resolve().parent.parent
directory = root / 'build/competition-python'
os.umask(0o007)
directory.mkdir(parents=True, exist_ok=True, mode=0o770)
links = {}
for original in (source, libraries, metadata[0]):
    name = 'triton' if original == source else original.name
    link = directory / name
    if link.is_symlink() and link.resolve() == original.resolve():
        pass
    elif link.exists() or link.is_symlink():
        raise RuntimeError('Refusing to replace an existing compiler dependency: ' + str(link))
    else:
        link.symlink_to(original, target_is_directory=True)
    links[name] = str(link)
assert initial.read_bytes() == content, 'The source package changed during preparation'
print('OFFICIAL_COMPILER_DEPENDENCY_PREPARED', json.dumps({
    'source': str(source), 'links': links, 'version': version,
    'init_sha256': hashlib.sha256(content).hexdigest()}), flush=True)
