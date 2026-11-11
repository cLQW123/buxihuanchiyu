"""Candidate SwiGLU matching BF16 SiLU rounding before multiplication."""
import functools

import ninetoothed.language as ntl
from ninetoothed import Tensor
from ntops.kernels.element_wise import arrangement


def application(output, up, gate):
    gate_f32 = ntl.cast(gate, ntl.float32)
    activated = (ntl.sigmoid(gate_f32) * gate_f32).to(gate.dtype)
    output = ntl.cast(activated, ntl.float32) * ntl.cast(up, ntl.float32)


def premake(ndim, dtype=None, block_size=None):
    tensors = tuple(Tensor(ndim, dtype=dtype) for _ in range(3))
    return functools.partial(arrangement, block_size=block_size), application, tensors
