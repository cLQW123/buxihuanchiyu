"""Fetch an exact published main commit and export only its tracked sources.

Network preparation runs on the local machine; the exported archive builds
offline in a fresh exclusive eight-card competition allocation.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


def export_archive(checkout, destination, expected_commit,
                   publication_status='public-commit-verified'):
    lock = json.loads((checkout / 'source-lock.json').read_text())
    repositories = [(checkout, '')]
    commits = {'main': expected_commit}
    for name, record in lock['projects'].items():
        repo = checkout / 'Projects' / name
        actual = git(repo, 'rev-parse', 'HEAD')
        assert actual == record['commit'], (name, actual, record['commit'])
        assert not git(repo, 'status', '--porcelain'), name
        repositories.append((repo, 'Projects/' + name + '/'))
        commits[name] = actual
    for project, nested in (('InfiniCore', 'third_party/spdlog'),
                            ('InfiniLM', 'third_party/spdlog'),
                            ('InfiniLM', 'third_party/json')):
        path = checkout / 'Projects' / project / nested
        pin = git(checkout / 'Projects' / project, 'rev-parse', 'HEAD:' + nested)
        assert git(path, 'rev-parse', 'HEAD') == pin, str(path)
        repositories.append((path, 'Projects/' + project + '/' + nested + '/'))
        commits[project + '/' + nested] = pin

    expected = json.loads((checkout / lock['source_manifest']).read_text())
    for relative, sha in expected.items():
        path = checkout / relative
        assert path.is_file(), relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha, relative

    exported = {}
    with tarfile.open(destination, 'w:gz', compresslevel=6) as output:
        for repo, prefix in repositories:
            raw = subprocess.check_output(['git', '-C', str(repo), 'archive',
                                           '--format=tar', '--prefix=' + prefix, 'HEAD'])
            with tarfile.open(fileobj=io.BytesIO(raw), mode='r:') as archive:
                for member in archive:
                    name = Path(member.name)
                    assert not name.is_absolute() and '..' not in name.parts, member.name
                    assert '.git' not in name.parts and '__pycache__' not in name.parts, member.name
                    if not prefix and name.parts:
                        if name.parts[0] in ('outputs', 'examples'):
                            continue
                        if name.parts[0] == 'evidence' and member.name != lock['source_manifest']:
                            continue
                    assert name.suffix.lower() not in ('.so', '.o', '.a', '.hsaco', '.cubin', '.pyc'), member.name
                    if member.isfile():
                        content = archive.extractfile(member).read()
                        exported[member.name] = hashlib.sha256(content).hexdigest()
                        output.addfile(member, io.BytesIO(content))
                    else:
                        output.addfile(member)
        provenance = json.dumps({'scope': 'tracked pinned source only; no build products',
                                 'publication_status': publication_status,
                                 'integration_commit': expected_commit,
                                 'excluded_main_content': ['outputs/', 'examples/', 'evidence/ except the source manifest'],
                                 'commits': commits, 'files': exported}, indent=2).encode()
        entry = tarfile.TarInfo('SOURCE_PROVENANCE.json')
        entry.size = len(provenance)
        entry.mode = 0o644
        output.addfile(entry, io.BytesIO(provenance))
    result = {'integration_commit': expected_commit, 'publication_status': publication_status,
              'archive': str(destination.resolve()),
              'bytes': destination.stat().st_size,
              'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
              'commits': commits, 'tracked_file_count': len(exported)}
    destination.with_suffix(destination.suffix + '.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', default='https://github.com/cLQW123/buxihuanchiyu.git')
    parser.add_argument('--commit', required=True)
    parser.add_argument('--checkout', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    assert re.fullmatch('[0-9a-f]{40}', args.commit), 'Use an exact full commit hash'
    assert not args.checkout.exists(), 'Use a new checkout directory'
    assert not args.archive.exists(), 'Preserve existing source archives'
    args.checkout.parent.mkdir(parents=True, exist_ok=True)
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout',
                    '--single-branch', '--branch', 'main', args.repository,
                    str(args.checkout)], check=True)
    subprocess.run(['git', '-C', str(args.checkout), 'checkout', '--detach', args.commit], check=True)
    assert git(args.checkout, 'rev-parse', 'HEAD') == args.commit
    subprocess.run(['bash', 'scripts/prepare_sources.sh'], cwd=args.checkout, check=True)
    assert not git(args.checkout, 'status', '--porcelain')
    print(json.dumps(export_archive(args.checkout, args.archive, args.commit), indent=2))


if __name__ == '__main__':
    main()
