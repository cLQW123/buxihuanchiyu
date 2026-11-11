"""NineToothed RMSNorm/LayerNorm with independently typed affine storage."""
import functools

import ninetoothed as nt
import ninetoothed.language as ntl


def arrangement(input, weight, bias, output, eps, centered, *, width):
    def rows(tensor):
        tiled = tensor.tile((1, width))
        tiled.dtype = tiled.dtype.squeeze(0)
        return tiled
    w = weight[None, :].expand((input.shape[0], -1))
    b = bias[None, :].expand((input.shape[0], -1))
    return rows(input), rows(w), rows(b), rows(output), eps, centered


def application(input, weight, bias, output, eps, centered):
    values = ntl.cast(input, ntl.float32)
    count = input.source.shape[-1]
    if centered:
        mean = ntl.sum(values, 0) / count
        centered_values = values - mean
        centered_values = ntl.where(input.offsets(-1) < count, centered_values, 0)
        variance = ntl.sum(centered_values * centered_values, 0) / count
        normalized = centered_values * ntl.rsqrt(variance + ntl.cast(eps, ntl.float32))
    else:
        variance = ntl.sum(values * values, 0) / count
        normalized = values * ntl.rsqrt(variance + ntl.cast(eps, ntl.float32))
    output = normalized * ntl.cast(weight, ntl.float32) + ntl.cast(bias, ntl.float32)


def premake(width, centered, input_dtype, weight_dtype=None, bias_dtype=None, output_dtype=None):
    weight_dtype = weight_dtype or input_dtype
    bias_dtype = bias_dtype or input_dtype
    output_dtype = output_dtype or input_dtype
    tensors = (nt.Tensor(shape=(None, width), dtype=input_dtype, other=0),
               nt.Tensor(shape=(width,), dtype=weight_dtype, other=0),
               nt.Tensor(shape=(width,), dtype=bias_dtype, other=0),
               nt.Tensor(shape=(None, width), dtype=output_dtype),
               nt.Tensor(0, dtype=nt.float64),
               nt.Tensor(0, constexpr=True, value=int(centered)))
    return functools.partial(arrangement, width=width), application, tensors
