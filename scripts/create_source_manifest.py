"""Hash tracked implementation files without recursively hashing evidence itself."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent.parent
lock = json.loads((root / 'source-lock.json').read_text())
repositories = [(root, '')]
repositories += [(root / 'Projects' / name, 'Projects/' + name + '/')
                 for name in lock['projects']]
repositories += [(root / 'Projects' / project / relative,
                  'Projects/' + project + '/' + relative + '/')
                 for project, relative in (
                     ('InfiniCore', 'third_party/spdlog'),
                     ('InfiniLM', 'third_party/spdlog'),
                     ('InfiniLM', 'third_party/json'))]
files = {}
for repo, prefix in repositories:
    names = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z']).decode().split('\0')
    for name in names:
        if not name:
            continue
        if not prefix and (name == 'source-lock.json' or name.startswith(('evidence/', 'outputs/', 'examples/'))):
            continue
        path = repo / name
        if path.is_file():
            files[prefix + name] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
destination = root / lock['source_manifest']
destination.parent.mkdir(exist_ok=True)
destination.write_text(json.dumps(dict(sorted(files.items())), indent=2) + '\n')
lock['source_manifest_sha256'] = hashlib.file_digest(destination.open('rb'), 'sha256').hexdigest()
lock['source_manifest_file_count'] = len(files)
lock['source_manifest_scope'] = 'tracked implementation, dependency locks and documentation; source-lock and evidence excluded to avoid recursive hashes'
(root / 'source-lock.json').write_text(json.dumps(lock, indent=2) + '\n')
print('TRACKED_SOURCE_MANIFEST_WRITTEN', len(files), lock['source_manifest_sha256'])
