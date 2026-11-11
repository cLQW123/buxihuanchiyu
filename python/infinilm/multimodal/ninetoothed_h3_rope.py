"""Storage-only Torch bridge to the validated NineToothed H3 rotary kernel.

The generated kernel performs all frequency and trigonometric arithmetic.
This module is prepared for integration; callers must configure a validated
kernel directory before selecting it in the model adapter.
"""
import ctypes
import functools
import json
import os
from pathlib import Path


class _TensorArgument(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p),
                ("shape", ctypes.POINTER(ctypes.c_uint64)),
                ("strides", ctypes.POINTER(ctypes.c_int64))]


@functools.cache
def _load_kernel(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "validation.json").read_text())
    if len(manifest.get("cases", ())) != 6 or not all(case.get("pass") for case in manifest["cases"]):
        raise RuntimeError("H3 rotary kernel lacks its six required numeric checks")
    input_kind = {
        "F32 positions, BF16 output, frequency dimension16, theta10000": "float32",
        "F64 positions, BF16 output, frequency dimension16, theta10000": "float64",
    }.get(manifest.get("scope"))
    if input_kind is None:
        raise RuntimeError("H3 rotary kernel has a different validated configuration")
    name = manifest["kernel"]
    if Path(name).name != name:
        raise RuntimeError("Invalid H3 rotary kernel name")
    library = ctypes.CDLL(str(directory / (name + ".so")))
    launch = getattr(library, "launch_" + name)
    launch.argtypes = [ctypes.c_void_p] + [_TensorArgument] * 4
    launch.restype = ctypes.c_int
    return library, launch, input_kind


def rotary_cache(position_ids, rope_freq_dim=16, rope_theta=10000.0, dtype=None):
    import torch

    if dtype is None:
        dtype = torch.bfloat16
    if rope_freq_dim != 16 or rope_theta != 10000.0 or dtype != torch.bfloat16:
        raise NotImplementedError("Validated H3 rotary configuration is dim16/theta10000/BF16")
    if position_ids.ndim != 2 or position_ids.shape[1] != 3:
        raise ValueError("position_ids must have shape [tokens, 3]")
    if position_ids.dtype not in (torch.float32, torch.float64) or position_ids.device.type != "cuda":
        raise TypeError("Validated H3 rotary input is an F32 or F64 accelerator tensor")
    directory = os.environ.get("SGCUP_H3_ROPE_KERNEL_DIR")
    if not directory:
        raise RuntimeError("SGCUP_H3_ROPE_KERNEL_DIR must point to validated generated kernels")
    library, launch, input_kind = _load_kernel(directory)
    if position_ids.dtype != getattr(torch, input_kind):
        raise TypeError(f"H3 rotary kernel expects {input_kind}; got {position_ids.dtype}")
    output = torch.empty((position_ids.shape[0], 2, 3, 16),
                         dtype=dtype, device=position_ids.device)
    if position_ids.shape[0] == 0:
        return output.view(0, 96)
    buffers = []
    arguments = []
    for tensor in (position_ids, output):
        shape = (ctypes.c_uint64 * tensor.ndim)(*tensor.shape)
        strides = (ctypes.c_int64 * tensor.ndim)(*tensor.stride())
        buffers.extend((shape, strides))
        arguments.append(_TensorArgument(tensor.data_ptr(), shape, strides))
    ignored = _TensorArgument(None, None, None)
    with torch.cuda.device(position_ids.device):
        status = launch(ctypes.c_void_p(torch.cuda.current_stream().cuda_stream),
                        *arguments, ignored, ignored)
        if status:
            raise RuntimeError(f"H3 rotary kernel launch failed with status {status}")
        # The native engine consumes this result on its own runtime streams.
        torch.cuda.synchronize(position_ids.device)
    return output.view(position_ids.shape[0], 96)
