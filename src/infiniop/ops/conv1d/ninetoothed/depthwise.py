"""NineToothed depthwise Conv1D without one launch per channel."""
import functools

import ninetoothed as nt
import ninetoothed.language as ntl


def arrangement(input, weight, bias, output, *, kernel_size, stride, pad_left,
                pad_right, dilation, transpose, kernel_last, kernel_period,
                block_size):
    if transpose:
        weight = weight[:, :, kernel_last::-kernel_period]
    windows = input.pad(((0, 0), (0, 0), (pad_left, pad_right)))
    windows = windows.tile((1, 1, kernel_size), strides=(-1, -1, stride),
                           dilation=(1, 1, dilation), floor_mode=True)
    windows = windows.ravel().squeeze((3, 4))
    windows = windows.tile((1, 1, block_size, kernel_size)).squeeze(3)
    windows.dtype = windows.dtype.squeeze((0, 1))
    filters = weight[:, 0, :][None, :, None, :].expand(
        (input.shape[0], -1, output.shape[2], -1))
    filters = filters.tile((1, 1, block_size, kernel_size)).squeeze(3)
    filters.dtype = filters.dtype.squeeze((0, 1))
    affine = bias[None, :, None].expand((output.shape[0], -1, output.shape[2]))
    affine = affine.tile((1, 1, block_size))
    affine.dtype = affine.dtype.squeeze((0, 1))
    result = output.tile((1, 1, block_size))
    result.dtype = result.dtype.squeeze((0, 1))
    return windows, filters, affine, result


def application(input, weight, bias, output):
    products = ntl.cast(input, ntl.float32) * ntl.cast(weight, ntl.float32)
    output = ntl.sum(products, 1) + ntl.cast(bias, ntl.float32)


def application_bias_bf16(input, weight, bias, output):
    products = ntl.cast(input, ntl.float32) * ntl.cast(weight, ntl.float32)
    rounded = ntl.cast(ntl.sum(products, 1), ntl.bfloat16)
    output = ntl.cast(rounded, ntl.float32) + ntl.cast(bias, ntl.float32)


def premake(kernel_size, dtype, stride=1, pad_left=0, pad_right=0,
            dilation=1, transpose=False, kernel_last=0, kernel_period=1,
            block_size=64, round_before_bias=False):
    if min(kernel_size, stride, dilation, kernel_period, block_size) <= 0:
        raise ValueError("Depthwise convolution geometry must be positive")
    tensors = (nt.Tensor(3, dtype=dtype, other=0),
               nt.Tensor(3, dtype=dtype, other=0),
               nt.Tensor(1, dtype=dtype, other=0), nt.Tensor(3, dtype=dtype))
    config = dict(kernel_size=kernel_size, stride=stride, pad_left=pad_left,
                  pad_right=pad_right, dilation=dilation, transpose=transpose,
                  kernel_last=kernel_last, kernel_period=kernel_period,
                  block_size=block_size)
    selected = application_bias_bf16 if round_before_bias and dtype == nt.bfloat16 else application
    return functools.partial(arrangement, **config), selected, tensors
