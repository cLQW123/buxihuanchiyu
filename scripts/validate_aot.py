"""Validate the generated rotary and Qwen RMS AOT libraries before inference."""
import ctypes
import hashlib
import inspect
import json
import os
from pathlib import Path

import torch
from infiniop.ops.minimax_h3_rope_cache.ninetoothed import rope_cache


class TensorArgument(ctypes.Structure):
    _fields_ = [('data', ctypes.c_void_p),
                ('shape', ctypes.POINTER(ctypes.c_uint64)),
                ('strides', ctypes.POINTER(ctypes.c_int64))]


def tensor_argument(tensor):
    shape = (ctypes.c_uint64 * tensor.ndim)(*tensor.shape)
    strides = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
    return TensorArgument(tensor.data_ptr(), shape, strides), (shape, strides)


def launcher(directory, name, nargs):
    library = ctypes.CDLL(str(directory / (name + '.so')))
    launch = getattr(library, 'launch_' + name)
    launch.argtypes = [ctypes.c_void_p] + [TensorArgument] * nargs
    launch.restype = ctypes.c_int
    return library, launch


root = Path(os.environ['SGCUP_ROOT'])
torch.manual_seed(42)
directory = root / 'build/h3-rope'
name = 'minimax_h3_rope_cache_16_10000_F64_BF16_v65'
library, launch = launcher(directory, name, 4)
unused = TensorArgument(None, None, None)
cases = []
for tokens in (1, 17, 257):
    for strided in (False, True):
        positions = torch.randn((tokens, 6 if strided else 3), dtype=torch.float64, device='cuda') * 128
        positions = positions[:, ::2] if strided else positions
        output = torch.empty((tokens, 2, 3, 16), dtype=torch.bfloat16, device='cuda')
        args = [tensor_argument(tensor) for tensor in (positions, output)]
        status = launch(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                        *(arg[0] for arg in args), unused, unused)
        torch.cuda.synchronize()
        if status:
            raise RuntimeError('Rotary launch status ' + str(status))
        inverse = 1.0 / (10000.0 ** (torch.arange(0, 32, 2, dtype=torch.float32, device='cuda') / 32))
        phase = (positions.float().unsqueeze(-1) * inverse.view(1, 1, -1)).flatten(1)
        reference = torch.cat((phase.cos(), phase.sin()), dim=-1).to(torch.bfloat16)
        actual = output.view(tokens, 96)
        torch.testing.assert_close(actual, reference, atol=0.004, rtol=0.004)
        cases.append({'tokens': tokens, 'strided': strided, 'pass': True,
                      'max_abs_error': (actual.float() - reference.float()).abs().max().item()})
        print('ROTARY_AOT_CASE', cases[-1], flush=True)
source = Path(inspect.getsourcefile(rope_cache.application))
(directory / 'validation.json').write_text(json.dumps({
    'kernel': name, 'scope': 'F64 positions, BF16 output, frequency dimension16, theta10000',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'cases': cases,
    'device': torch.cuda.get_device_name(0)}, indent=2))

directory = root / 'build/qwen-rms'
cases = []
for width in (128, 5120):
    library, launch = launcher(directory, f'sgcup_qwen_rms_v140_{width}', 4)
    for gain in (1, 128):
        x = torch.randn((37, width * 2), device='cuda', dtype=torch.bfloat16)[:, ::2] * gain
        weight = torch.randn(width, device='cuda', dtype=torch.bfloat16)
        output = torch.empty_like(x)
        eps = ctypes.c_double(1e-6)
        args = [tensor_argument(tensor) for tensor in (x, weight, output)]
        status = launch(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                        *(arg[0] for arg in args), TensorArgument(ctypes.addressof(eps), None, None))
        torch.cuda.synchronize()
        if status:
            raise RuntimeError('RMS launch status ' + str(status))
        normalized = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps.value)).to(x.dtype)
        reference = (normalized.float() * weight.float()).to(x.dtype)
        torch.testing.assert_close(output, reference, atol=0.016, rtol=0.01)
        cases.append({'width': width, 'gain': gain, 'pass': True,
                      'max_abs_error': (output.float() - reference.float()).abs().max().item()})
        print('QWEN_RMS_AOT_CASE', cases[-1], flush=True)
(directory / 'validation.json').write_text(json.dumps({'cases': cases}, indent=2))
print('REBUILT_AOT_VALIDATION_PASSED', flush=True)
