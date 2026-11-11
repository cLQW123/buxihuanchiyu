"""NineToothed group normalization with FP32 reduction and explicit stats.

Storage is supplied as [batch, groups, channels_per_group * spatial] views.
Affine tensors use [groups, channels_per_group]; their broadcast is symbolic.
"""
import functools

import ninetoothed as nt
import ninetoothed.language as ntl
from ntops.kernels.reduction import arrangement as reduce_arrangement


def arrangement(input, weight, bias, output, mean, rstd, eps, *, block_size):
    spatial = input.shape[2] // weight.shape[1]

    def affine(tensor):
        return tensor[None, :, :, None].expand(
            (input.shape[0], -1, -1, spatial)
        ).flatten(start_dim=2)

    inputs = reduce_arrangement(input, affine(weight), affine(bias), output,
                                dim=-1, block_size=block_size)

    def statistic(tensor):
        tiled = tensor.tile((1, 1, 1))
        tiled.dtype = tiled.dtype.squeeze((0, 1))
        return tiled

    return *inputs, statistic(mean), statistic(rstd), eps


def application(input, weight, bias, output, mean, rstd, eps):
    count = input.source.shape[-1]
    total = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for block in range(input.shape[0]):
        total += ntl.cast(input[block], ntl.float32)
    group_mean = ntl.sum(total, 0) / count
    squares = ntl.zeros(input.dtype.shape, dtype=ntl.float32)
    for block in range(input.shape[0]):
        delta = ntl.cast(input[block], ntl.float32) - group_mean
        delta = ntl.where(input[block].offsets(-1) < count, delta, 0)
        squares += delta * delta
    group_rstd = ntl.rsqrt(ntl.sum(squares, 0) / count + eps)
    mean = group_mean
    rstd = group_rstd
    for block in range(input.shape[0]):
        normalized = (ntl.cast(input[block], ntl.float32) - group_mean) * group_rstd
        output[block] = normalized * weight[block] + bias[block]


def premake(dtype, eps=1e-6, block_size=512):
    if eps <= 0 or block_size <= 0:
        raise ValueError("GroupNorm requires positive epsilon and block size")
    tensors = (nt.Tensor(3, dtype=dtype, other=0),
               nt.Tensor(2, dtype=dtype), nt.Tensor(2, dtype=dtype),
               nt.Tensor(3, dtype=dtype), nt.Tensor(shape=(None, None, 1), dtype=nt.float32),
               nt.Tensor(shape=(None, None, 1), dtype=nt.float32), nt.Tensor(0, constexpr=True, value=eps))
    return (functools.partial(arrangement, block_size=block_size),
            application, tensors)
