"""NineToothed H3 continuous three-axis rotary-cache candidate.

Output storage is [tokens, cos_or_sin, axis, frequency], which is a view of
InfiniLM's [tokens, 6 * rope_freq_dim] cache. This candidate is not yet enabled
in the native adapter; target-device numeric validation is required first.
"""
import functools

import ninetoothed as nt
import ninetoothed.language as ntl
from ninetoothed.language import libdevice


def arrangement(position_ids, output, rope_freq_dim, rope_theta):
    positions = position_ids.tile((1, -1))
    positions.dtype = positions.dtype.squeeze(0)
    positions = positions.unsqueeze(1).unsqueeze(3)

    cache = output.tile((1, 1, -1, -1)).tile((1, -1, 1, 1))
    cache.dtype = cache.dtype.squeeze((0, 2, 3))
    cache.dtype.dtype = cache.dtype.dtype.squeeze((0, 1))
    return positions, cache, rope_freq_dim, rope_theta


def application(position_ids, output, rope_freq_dim, rope_theta):
    frequency_index = ntl.arange(0, rope_freq_dim).to(ntl.float32)
    theta = ntl.full((), rope_theta, dtype=ntl.float32)
    inverse_frequency = 1.0 / libdevice.pow(theta, frequency_index / rope_freq_dim)
    phase = ntl.cast(position_ids, ntl.float32)[:, None] * inverse_frequency
    output[0] = libdevice.cos(phase)
    output[1] = libdevice.sin(phase)


def premake(rope_freq_dim=16, rope_theta=10000.0, position_dtype=None, dtype=None):
    if rope_freq_dim <= 0 or rope_freq_dim & (rope_freq_dim - 1):
        raise ValueError("rope_freq_dim must be a positive power of two")
    if rope_theta <= 0:
        raise ValueError("rope_theta must be positive")
    position_ids = nt.Tensor(2, dtype=position_dtype, other=0)
    position_ids.shape = (position_ids.shape[0], 3)
    output = nt.Tensor(4, dtype=dtype, shape_options=(None, None, None, {"constexpr": True}))
    output.shape = (output.shape[0], 2, 3, rope_freq_dim)
    frequency_count = nt.Tensor(0, constexpr=True, value=rope_freq_dim)
    theta = nt.Tensor(0, constexpr=True, value=float(rope_theta))
    return arrangement, application, (position_ids, output, frequency_count, theta)
