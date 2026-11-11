"""Check the rebuilt native dispatch on every allocated accelerator."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch

root = Path(os.environ['SGCUP_ROOT']).resolve()
prefix = Path(os.environ['INFINI_ROOT']).resolve()
for name in ('libinfinirt.so', 'libinfiniop.so', 'libinfinicore_cpp_api.so',
             'mha_ninetoothed.so'):
    ctypes.CDLL(str(prefix / 'lib' / name), mode=ctypes.RTLD_GLOBAL)
import infinicore as core
from infinicore.lib import _infinicore
from infinicore.utils import to_infinicore_dtype
import _infinilm

modules = {
    'InfiniCore': Path(_infinicore.__file__).resolve(),
    'InfiniLM': Path(_infinilm.__file__).resolve(),
}
assert modules['InfiniCore'] == (root / 'Projects/InfiniCore/python/infinicore/lib/_infinicore.so').resolve(), modules
assert modules['InfiniLM'] == (root / 'build/python/_infinilm.so').resolve(), modules
assert torch.cuda.device_count() == 8
records = []


def storage(tensor):
    return core.strided_from_blob(
        tensor.data_ptr(), list(tensor.shape), list(tensor.stride()),
        dtype=to_infinicore_dtype(tensor.dtype),
        device=core.device('cuda', tensor.device.index))


def check(name, actual, expected, device, *, exact=False):
    core.sync_stream()
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, expected, atol=0 if exact else .02,
                               rtol=0 if exact else .02)
    record = {'name': name, 'device': device, 'shape': list(actual.shape),
              'max_abs_error': (actual.float() - expected.float()).abs().max().item(),
              'exact': exact, 'pass': True}
    records.append(record)
    print('REBUILT_NATIVE_CASE', json.dumps(record), flush=True)


with torch.inference_mode():
    for device in range(8):
        torch.cuda.set_device(device)
        core.set_device(core.device('cuda', device))
        torch_device = torch.device('cuda', device)
        torch.manual_seed(42 + device)
        for width in (64, 72, 128):
            for causal in (False, True):
                q = torch.randn((1, 17, 4, width), device=torch_device, dtype=torch.bfloat16)
                k = torch.randn((1, 17, 2, width), device=torch_device, dtype=torch.bfloat16)
                v = torch.randn_like(k)
                output = torch.empty_like(q)
                repeated_k = k.repeat_interleave(2, dim=2).transpose(1, 2)
                repeated_v = v.repeat_interleave(2, dim=2).transpose(1, 2)
                expected = torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2), repeated_k, repeated_v,
                    is_causal=causal).transpose(1, 2)
                torch.cuda.synchronize(device)
                core.mha(storage(q), storage(k), storage(v), scale=width ** -.5,
                         is_causal=causal, out=storage(output))
                check(f'mha-head{width}-causal{causal}', output, expected, device)
        for length in (1, 7):
            kc = torch.randn((2, 2, 48, 128), device=torch_device, dtype=torch.bfloat16)
            vc = torch.randn_like(kc)
            k = torch.randn((2, 2, length, 128), device=torch_device, dtype=torch.bfloat16)
            v = torch.randn_like(k)
            past = torch.tensor([0, 31], device=torch_device, dtype=torch.int64)
            kr, vr = kc.clone(), vc.clone()
            for batch, offset in enumerate((0, 31)):
                kr[batch, :, offset:offset + length] = k[batch]
                vr[batch, :, offset:offset + length] = v[batch]
            torch.cuda.synchronize(device)
            core.kv_caching(*(storage(t) for t in (kc, vc, k, v, past)))
            check(f'kv-key-len{length}', kc, kr, device, exact=True)
            check(f'kv-value-len{length}', vc, vr, device, exact=True)
        up = torch.randn((1, 17, 512), device=torch_device, dtype=torch.bfloat16)
        gate = torch.randn_like(up)
        output = torch.empty_like(up)
        expected = up * torch.nn.functional.silu(gate)
        torch.cuda.synchronize(device)
        core.nn.functional.swiglu(storage(up), storage(gate), out=storage(output))
        check('swiglu', output, expected, device)

manifest = {'scope': 'rebuilt native dispatch on all eight devices',
            'modules': {name: {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name, path in modules.items()},
            'cases': records, 'pass': True}
(root / 'build/runtime-validation.json').write_text(json.dumps(manifest, indent=2))
print('EIGHT_DEVICE_REBUILT_RUNTIME_PASSED', len(records), flush=True)
