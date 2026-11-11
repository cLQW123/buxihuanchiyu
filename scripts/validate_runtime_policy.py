"""Check fixed backend selection and hash this build's required libraries.

This is a provenance check, not a replacement for GPU numerical validation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def collect_runtime_policy(root, prefix):
    root, prefix = Path(root).resolve(), Path(prefix).resolve()
    if prefix != (root / 'build/install').resolve():
        raise RuntimeError('The submitted install prefix must be this source build: ' + str(prefix))
    switches = {name: os.environ.get(name) for name in
                ('INFINILM_HYGON_NINETOOTHED_FLASH', 'SGCUP_TP_FP32')}
    if switches != {'INFINILM_HYGON_NINETOOTHED_FLASH': '1', 'SGCUP_TP_FP32': '0'}:
        raise RuntimeError('Source scripts/env.sh to select the submitted numerical paths: ' + repr(switches))
    if os.environ.get('SGCUP_TP_MATMUL_KERNEL_DIR'):
        raise RuntimeError('The disabled optional FP32 path must not retain an external kernel directory')
    for name, relative in (('SGCUP_H3_ROPE_KERNEL_DIR', 'build/h3-rope'),
                           ('SGCUP_QWEN_RMS_KERNEL_DIR', 'build/qwen-rms')):
        if Path(os.environ.get(name, '/missing-sgcup-kernel-directory')).resolve() != (root / relative).resolve():
            raise RuntimeError(name + ' must select this source build')

    paths = [prefix / 'lib' / name for name in
             ('libinfinirt.so', 'libinfiniop.so', 'libinfiniccl.so',
              'libinfinicore_cpp_api.so', 'mha_ninetoothed.so')]
    paths += [root / relative for relative in
              ('build/python/_infinilm.so',
               'Projects/InfiniCore/python/infinicore/lib/_infinicore.so',
               'build/h3-rope/minimax_h3_rope_cache_16_10000_F64_BF16_v65.so',
               'build/qwen-rms/sgcup_qwen_rms_v140_128.so',
               'build/qwen-rms/sgcup_qwen_rms_v140_5120.so')]
    generated = root / 'Projects/InfiniCore/build/ninetoothed'
    for operation in ('kv_caching', 'flash_attention', 'swiglu'):
        kernels = sorted(generated.glob(operation + '_*.so'))
        if not kernels:
            raise RuntimeError('Missing this build\'s NineToothed AOT kernels: ' + operation)
        paths += kernels

    libraries = {}
    for path in paths:
        actual = path.resolve()
        if not actual.is_relative_to(root) or not actual.is_file() or actual.stat().st_size == 0:
            raise RuntimeError('A required library is missing or outside this source build: ' + str(path))
        digest = hashlib.sha256()
        with actual.open('rb') as content:
            for chunk in iter(lambda: content.read(1024 * 1024), b''):
                digest.update(chunk)
        libraries[str(path.relative_to(root))] = {
            'path': str(actual), 'bytes': actual.stat().st_size, 'sha256': digest.hexdigest()}
    return {'scope': 'fixed backend selection and source-build library provenance',
            'switches': switches,
            'attention': 'NineToothed Hygon flash attention and MHA registration plugin',
            'tensor_parallel_linear': 'original InfiniCore GEMM followed by allreduce',
            'optional_fp32_matmul': 'disabled',
            'pythonpath': os.environ.get('PYTHONPATH', '').split(os.pathsep),
            'libraries': libraries}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    receipt = collect_runtime_policy(os.environ['SGCUP_ROOT'], os.environ['INFINI_ROOT'])
    if args.output:
        args.output.write_text(json.dumps(receipt, indent=2))
    print('SUBMITTED_RUNTIME_POLICY_VERIFIED', json.dumps(receipt), flush=True)
