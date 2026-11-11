"""NineToothed dense Conv3D candidate for the H3 video VAE.

This extends the pinned NTops Conv2D im2col arrangement to three spatial axes.
It is not selected in formal inference until target numerical tests pass.
"""
import copy
import functools

import ninetoothed as nt
import ninetoothed.language as ntl
from ntops.kernels import mm


def arrangement(input, weight, bias, output, input_precision, *, stride,
                padding, dilation, block_size_m, block_size_n, block_size_k):
    matrix = functools.partial(mm.arrangement, block_size_m=block_size_m,
                               block_size_n=block_size_n, block_size_k=block_size_k)
    columns = input.pad(((0, 0), (0, 0), *((p, p) for p in padding)))
    columns = columns.tile((1, *weight.shape[1:]),
                           strides=(-1, -1, *stride), dilation=(1, 1, *dilation),
                           floor_mode=True)
    columns = columns.squeeze(1)
    columns.dtype = columns.dtype.squeeze(0)
    columns = columns.ravel().flatten(end_dim=4).flatten(start_dim=1)
    filters = weight.flatten(start_dim=1).permute((1, 0))
    affine = bias[None, :, None, None, None].expand(
        (output.shape[0], -1, *output.shape[2:]))
    affine = affine.permute((0, 2, 3, 4, 1)).flatten(end_dim=4)
    result = output.permute((0, 2, 3, 4, 1)).flatten(end_dim=4)
    _, _, affine, _ = matrix(copy.deepcopy(columns), copy.deepcopy(filters),
                             affine, copy.deepcopy(input_precision))
    columns, filters, result, precision = matrix(columns, filters, result, input_precision)
    return columns, filters, affine, result, precision


def application(input, weight, bias, output, input_precision):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for block in range(input.shape[0]):
        accumulator += ntl.dot(input[block], weight[block])
    output = accumulator + bias


def application_bias_bf16(input, weight, bias, output, input_precision):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for block in range(input.shape[0]):
        accumulator += ntl.dot(input[block], weight[block])
    rounded = ntl.cast(accumulator, ntl.bfloat16)
    output = ntl.cast(rounded, ntl.float32) + bias


def application_bias_f16(input, weight, bias, output, input_precision):
    accumulator = ntl.zeros(output.shape, dtype=ntl.float32)
    for block in range(input.shape[0]):
        accumulator += ntl.dot(input[block], weight[block])
    rounded = ntl.cast(accumulator, ntl.float16)
    output = ntl.cast(rounded, ntl.float32) + bias


def premake(stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1),
            dtype=None, block_size_m=16, block_size_n=16, block_size_k=32,
            round_before_bias=False):
    if len(stride) != 3 or len(padding) != 3 or len(dilation) != 3:
        raise ValueError("Conv3D geometry requires three spatial dimensions")
    if any(x <= 0 for x in (*stride, *dilation)) or any(x < 0 for x in padding):
        raise ValueError("Invalid Conv3D stride, padding, or dilation")
    arrange = functools.partial(arrangement, stride=stride, padding=padding,
                                dilation=dilation, block_size_m=block_size_m,
                                block_size_n=block_size_n, block_size_k=block_size_k)
    tensors = (nt.Tensor(5, dtype=dtype, other=0),
               nt.Tensor(5, dtype=dtype, other=0),
               nt.Tensor(1, dtype=dtype, other=0),
               nt.Tensor(5, dtype=dtype),
               nt.Tensor(0, constexpr=True, value=2))
    selected = application
    if round_before_bias:
        if dtype == nt.bfloat16:
            selected = application_bias_bf16
        elif dtype == nt.float16:
            selected = application_bias_f16
        elif dtype != nt.float32:
            raise ValueError("Bias rounding requires a concrete floating dtype")
    return arrange, selected, tensors
