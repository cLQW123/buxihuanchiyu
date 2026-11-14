"""在联网准备机打包完整新工程，避免macOS元数据污染Linux工具树。"""
import argparse
import hashlib
from pathlib import Path
import tarfile

from check_inputs import bundled_first_frame


ROOT = Path(__file__).resolve().parent.parent


def package_checkout(root, destination):
    root = root.resolve(strict=True)
    destination = destination.resolve()
    if destination.is_relative_to(root):
        raise RuntimeError('输出tar必须位于源码目录之外')
    if destination.exists():
        raise RuntimeError('保留已有归档，请指定新的输出文件: ' + str(destination))
    for name in ('build', 'outputs'):
        if (root / name).exists():
            raise RuntimeError('请从没有旧build/outputs的全新检出目录打包')
    for name in ('SOURCE_PROVENANCE.json', '.git', 'inputs/dependency-receipt.json',
                 'inputs/wheelhouse', 'inputs/vendor-triton', 'inputs/vendor-xmake',
                 'assets/competition-first-frame.png'):
        if not (root / name).exists():
            raise RuntimeError('请先完成prepare_sources及fetch_dependencies: ' + name)

    def keep(member):
        relative = Path(member.name).relative_to('buxihuanchiyu')
        if any(part.startswith('._') or part in ('.DS_Store', '__MACOSX', '__pycache__')
               for part in relative.parts):
            return None
        if relative.parts[:2] == ('inputs', 'archives') or relative.suffix == '.pyc':
            return None
        path = root / relative
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise RuntimeError('归档不能依赖源码目录外的符号链接: ' + str(relative))
        member.uid = member.gid = 0
        member.uname = member.gname = ''
        return member

    temporary = destination.with_name(destination.name + '.partial')
    try:
        with tarfile.open(temporary, 'w', format=tarfile.PAX_FORMAT) as output:
            output.add(root, arcname='buxihuanchiyu', filter=keep)
        with tarfile.open(temporary) as archive:
            names = archive.getnames()
            assert 'buxihuanchiyu/assets/competition-first-frame.png' in names
            assert not any(Path(name).name.startswith('._') for name in names)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    bundled_first_frame()
    path = package_checkout(ROOT, args.output)
    with path.open('rb') as stream:
        sha = hashlib.file_digest(stream, 'sha256').hexdigest()
    print('PORTABLE_OFFLINE_PACKAGE_READY', path, path.stat().st_size, sha, flush=True)


if __name__ == '__main__':
    main()
