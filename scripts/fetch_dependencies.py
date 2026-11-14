"""Prepare pinned offline dependency archives on a network-enabled machine."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request
from check_inputs import bundled_first_frame


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def download_asset(item, manifest, temporary, transport):
    routes = ('direct', 'api') if transport == 'auto' else (transport,)
    failures = []
    for route in routes:
        try:
            headers = {'User-Agent': 'sgcup-offline-dependency-preparation',
                       'Accept': 'application/octet-stream'}
            if route == 'direct':
                url = ('https://github.com/' + manifest['repository'] + '/releases/download/'
                       + manifest['release_tag'] + '/' + item['file'])
            else:
                endpoint = ('https://api.github.com/repos/' + manifest['repository']
                            + '/releases/tags/' + manifest['release_tag'])
                request = urllib.request.Request(endpoint, headers={
                    'User-Agent': headers['User-Agent'], 'Accept': 'application/vnd.github+json'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    release = json.load(response)
                if release.get('draft'):
                    raise RuntimeError('Dependency release is not public')
                asset = next((entry for entry in release['assets'] if entry['name'] == item['file']), None)
                if asset is None or asset['size'] != item['bytes']:
                    raise RuntimeError('Public release asset is missing or has an unexpected size')
                url = asset['url']
                headers['X-GitHub-Api-Version'] = '2022-11-28'
            print('DOWNLOADING_OFFLINE_DEPENDENCY', item['file'], route, flush=True)
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response, temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream, length=4 * 1024 * 1024)
            if temporary.stat().st_size != item['bytes'] or digest(temporary) != item['sha256']:
                raise RuntimeError('Downloaded asset failed SHA256 verification')
            return
        except (OSError, RuntimeError, ValueError, StopIteration) as error:
            failures.append(route + ': ' + str(error))
            temporary.unlink(missing_ok=True)
            print('DEPENDENCY_DOWNLOAD_ROUTE_FAILED', item['file'], route, str(error), flush=True)
    raise RuntimeError('All public dependency download routes failed: ' + '; '.join(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'inputs')
    parser.add_argument('--archive-dir', type=Path, help='Use already downloaded release assets')
    parser.add_argument('--offline', action='store_true', help='Fail instead of downloading missing archives')
    parser.add_argument('--transport', choices=('auto', 'direct', 'api'), default='auto',
                        help='auto优先公开Release直链，失败后尝试API；两条路线均校验固定SHA256')
    args = parser.parse_args()
    bundled_first_frame()
    output = args.output_dir.resolve()
    archives = (args.archive_dir or output / 'archives').resolve()
    output.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / 'dependencies/artifacts.json').read_text())
    receipt = []
    for item in manifest['artifacts']:
        path = archives / item['file']
        if not path.is_file():
            if args.offline:
                raise RuntimeError('Missing offline asset: ' + str(path))
            temporary = path.with_name(path.name + '.partial')
            download_asset(item, manifest, temporary, args.transport)
            temporary.replace(path)
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise RuntimeError('Dependency asset differs from the pinned manifest: ' + str(path))
        with tarfile.open(path) as package:
            for member in package:
                name = Path(member.name)
                if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                    raise RuntimeError('Unsafe dependency archive entry: ' + member.name)
            package.extractall(output, filter='data')
        receipt.append(item)
    lock = json.loads((ROOT / 'dependencies/offline-lock.json').read_text())
    for item in lock['wheels']:
        assert digest(output / 'wheelhouse' / item['file']) == item['sha256'], item['file']
    vendor = json.loads((output / 'vendor-triton/VENDOR_PROVENANCE.json').read_text())
    for name, item in vendor['files'].items():
        assert digest(output / 'vendor-triton' / name) == item['sha256'], name
    tool = json.loads((output / 'vendor-xmake/VENDOR_PROVENANCE.json').read_text())
    for name, item in tool['files'].items():
        assert digest(output / 'vendor-xmake' / name) == item['sha256'], name
    (output / 'dependency-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('OFFLINE_DEPENDENCIES_PREPARED', len(lock['wheels']), len(vendor['files']), flush=True)


if __name__ == '__main__':
    main()
