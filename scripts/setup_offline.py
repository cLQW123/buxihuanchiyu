"""Create an isolated Python environment from verified wheels and vendor runtimes."""
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig


ROOT = Path(__file__).resolve().parent.parent


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def vendor_distribution(directory, name, expected):
    matches = [dist for dist in metadata.distributions(path=[str(directory)])
               if dist.metadata['Name'].lower().replace('_', '-') == name]
    if len(matches) != 1 or matches[0].version != expected:
        raise RuntimeError(f'Expected one vendor {name}=={expected} in {directory}')
    return matches[0]


def link_vendor(dist, packages, destination):
    records = []
    source = Path(dist.locate_file('')).resolve()
    names = set(packages)
    names.update(path.parts[0] for path in dist.files or []
                 if path.parts and path.parts[0].endswith('.dist-info'))
    for name in sorted(names):
        original = source / name
        if not original.exists():
            raise RuntimeError('Missing vendor runtime entry: ' + str(original))
        target = destination / name
        if target.exists() or target.is_symlink():
            raise RuntimeError('Vendor target already exists: ' + str(target))
        target.symlink_to(original, target_is_directory=original.is_dir())
        records.append({'name': name, 'source': str(original), 'link': str(target)})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wheelhouse', required=True, type=Path)
    parser.add_argument('--vendor-torch-site', required=True, type=Path)
    parser.add_argument('--vendor-triton-site', required=True, type=Path)
    parser.add_argument('--dtk-root', required=True, type=Path)
    parser.add_argument('--gcc-root', required=True, type=Path)
    parser.add_argument('--xmake-root', required=True, type=Path,
                        help='Installed xmake tree with bin/xmake and share/xmake')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 11) or sys.platform != 'linux':
        raise RuntimeError('Run setup with competition Linux CPython 3.11')
    for name, value in vars(args).items():
        setattr(args, name, value.resolve(strict=True))
    lock = json.loads((ROOT / 'dependencies/offline-lock.json').read_text())
    wheel_paths = []
    for item in lock['wheels']:
        if Path(item['file']).name != item['file']:
            raise RuntimeError('Invalid wheel basename in lock')
        path = args.wheelhouse / item['file']
        if not path.is_file() or sha256(path) != item['sha256']:
            raise RuntimeError('Missing or altered offline wheel: ' + str(path))
        wheel_paths.append(path)
    runtime = json.loads((ROOT / 'dependencies/runtime-lock.json').read_text())
    versions = runtime['environment']['packages']
    torch_dist = vendor_distribution(args.vendor_torch_site, 'torch', versions['torch'])
    triton_dist = vendor_distribution(args.vendor_triton_site, 'triton', versions['triton'])
    for path in (args.dtk_root / 'env.sh', args.gcc_root / 'bin/g++',
                 args.xmake_root / 'bin/xmake', args.xmake_root / 'share/xmake'):
        if not path.exists():
            raise RuntimeError('Missing competition toolchain entry: ' + str(path))
    build = ROOT / 'build'
    venv = build / 'venv'
    if venv.exists():
        raise RuntimeError('Use a fresh checkout/build directory; refusing to replace ' + str(venv))
    build.mkdir(exist_ok=True)
    subprocess.run([sys.executable, '-m', 'venv', str(venv)], check=True)
    python = venv / 'bin/python'
    site = Path(subprocess.check_output([str(python), '-c',
                'import sysconfig; print(sysconfig.get_path("purelib"))'], text=True).strip())
    torch_names = ['torch', 'torchgen', 'functorch']
    if (args.vendor_torch_site / 'torch.libs').exists():
        torch_names.append('torch.libs')
    torch_links = link_vendor(torch_dist, torch_names, site)
    triton_links = link_vendor(triton_dist, ['triton', 'triton.libs'], site)
    clean = os.environ.copy()
    clean.pop('PYTHONPATH', None)
    clean['PYTHONNOUSERSITE'] = '1'
    clean['PIP_NO_INDEX'] = '1'
    clean['PIP_DISABLE_PIP_VERSION_CHECK'] = '1'
    subprocess.run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps',
                    *map(str, wheel_paths)], env=clean, check=True)
    subprocess.run([str(python), '-m', 'pip', 'check'], env=clean, check=True)
    tools = build / 'tools'
    shutil.copytree(args.xmake_root, tools)
    quote = shlex.quote
    activation = '\n'.join([
        '#!/usr/bin/env bash',
        '# Generated from explicit competition dependency paths; no inherited Python overlays.',
        'export DTK_ROOT=' + quote(str(args.dtk_root)),
        'export CMAKE_PREFIX_PATH="" LD_LIBRARY_PATH="" LIBRARY_PATH=""',
        'export C_INCLUDE_PATH="" CPLUS_INCLUDE_PATH=""',
        'source "$DTK_ROOT/env.sh"',
        'source ' + quote(str(venv / 'bin/activate')),
        'unset PYTHONPATH SGCUP_ROOT SGCUP_PREFIX INFINI_ROOT SGCUP_COMPILER_PYTHONPATH',
        'unset SGCUP_TP_MATMUL_KERNEL_DIR SGCUP_CACHE_DIR NINETOOTHED_CACHE_DIR',
        'export PYTHONNOUSERSITE=1',
        'export SGCUP_COMPILER_PYTHONPATH=' + quote(str(site)),
        'export HYGON_ARCH=gfx936 SGCUP_ARCH=gfx936 XMAKE_ROOT=y',
        'export HYGON_CUDA_HOME="$DTK_ROOT/cuda/cuda"',
        'export XMAKE_PROGRAM_DIR=' + quote(str(tools / 'share/xmake')),
        'export SGCUP_XMAKE=' + quote(str(tools / 'bin/xmake')),
        'export PATH=' + quote(str(tools / 'bin') + ':' + str(args.gcc_root / 'bin')) + ':"$PATH"',
        'export CC=' + quote(str(args.gcc_root / 'bin/gcc')),
        'export CXX=' + quote(str(args.gcc_root / 'bin/g++')),
        'export SGCUP_VENDOR_TORCH_LIB=' + quote(str(args.vendor_torch_site / 'torch/lib')),
        'export LD_LIBRARY_PATH="$DTK_ROOT/lib:$DTK_ROOT/hip/lib:$HYGON_CUDA_HOME/lib64:$SGCUP_VENDOR_TORCH_LIB:$LD_LIBRARY_PATH"',
        'for sgcup_lib in lib64 isl/lib gmp/lib mpfr/lib mpc/lib; do',
        '  export LD_LIBRARY_PATH=' + quote(str(args.gcc_root)) + '/"$sgcup_lib:$LD_LIBRARY_PATH"',
        'done',
        'export LIBRARY_PATH="$DTK_ROOT/lib:$DTK_ROOT/hip/lib:$HYGON_CUDA_HOME/lib64:$LIBRARY_PATH"',
        'export CPLUS_INCLUDE_PATH="$DTK_ROOT/include:$HYGON_CUDA_HOME/include:$CPLUS_INCLUDE_PATH"',
        'export PYTORCH_INCLUDE_DIR=' + quote(str(args.vendor_torch_site / 'torch/include')),
        'export FA_SO_PATH=' + quote(str(site / 'flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so')),
        'export TMPDIR=' + quote(str(build / 'tmp')),
        'export HF_HOME=' + quote(str(build / 'cache/huggingface')),
        'export XDG_CACHE_HOME=' + quote(str(build / 'cache/xdg')),
        'export PIP_CACHE_DIR=' + quote(str(build / 'cache/pip')),
        'mkdir -p "$TMPDIR" "$HF_HOME" "$XDG_CACHE_HOME" "$PIP_CACHE_DIR"',
        'export NCCL_IB_DISABLE=1 NCCL_NET=Socket',
        'export SGCUP_DIAGNOSTIC_LATENTS=0 SGCUP_BUILD_JOBS=32',
        'source ' + quote(str(ROOT / 'scripts/env.sh')),
        'unset sgcup_lib',
        '',
    ])
    (build / 'activate.sh').write_text(activation)
    receipt = {
        'python': str(python), 'system_site_packages': False,
        'vendor_torch': torch_links, 'vendor_triton': triton_links,
        'offline_lock_sha256': sha256(ROOT / 'dependencies/offline-lock.json'),
        'wheel_count': len(wheel_paths), 'toolchain': {
            key: str(getattr(args, key)) for key in ('dtk_root', 'gcc_root', 'xmake_root')},
    }
    (build / 'offline-environment.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print('ISOLATED_OFFLINE_PYTHON_READY', json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
