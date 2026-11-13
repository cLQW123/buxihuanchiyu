"""Verify a pristine pinned Git checkout and record its exact source files."""
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent.parent


def git(directory, *args):
    return subprocess.check_output(['git', '-C', str(directory), *args], text=True).strip()


def main():
    lock = json.loads((ROOT / 'source-lock.json').read_text())
    commits = {'main': git(ROOT, 'rev-parse', 'HEAD')}
    repositories = [(ROOT, '')]
    for name, record in lock['projects'].items():
        repo = ROOT / 'Projects' / name
        actual = git(repo, 'rev-parse', 'HEAD')
        assert actual == record['commit'], (name, actual, record['commit'])
        commits[name] = actual
        repositories.append((repo, 'Projects/' + name + '/'))
    for project, relative in (('InfiniCore', 'third_party/spdlog'),
                              ('InfiniLM', 'third_party/spdlog'),
                              ('InfiniLM', 'third_party/json')):
        parent = ROOT / 'Projects' / project
        repo = parent / relative
        actual = git(repo, 'rev-parse', 'HEAD')
        assert actual == git(parent, 'rev-parse', 'HEAD:' + relative)
        commits[project + '/' + relative] = actual
        repositories.append((repo, 'Projects/' + project + '/' + relative + '/'))
    files = {}
    for repo, prefix in repositories:
        assert not git(repo, 'status', '--porcelain'), 'Use a pristine source checkout: ' + str(repo)
        for name in git(repo, 'ls-files', '-z').split('\0'):
            if not name:
                continue
            if not prefix and name.startswith(('outputs/', 'examples/')):
                continue
            if not prefix and name.startswith('evidence/') and name != lock['source_manifest']:
                continue
            path = repo / name
            if path.is_file():
                files[prefix + name] = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    expected = json.loads((ROOT / lock['source_manifest']).read_text())
    for name, sha in expected.items():
        assert files.get(name) == sha, name
    receipt = {'scope': 'pristine tracked pinned checkout; no build products',
               'publication_status': 'git-checkout-exact-commit',
               'integration_commit': commits['main'], 'commits': commits, 'files': files}
    (ROOT / 'SOURCE_PROVENANCE.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('EXACT_GIT_CHECKOUT_SOURCE_VERIFIED', commits['main'], len(files), flush=True)


if __name__ == '__main__':
    main()
