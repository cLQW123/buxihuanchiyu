"""Experimental strict NineToothed dispatch for the H3 audio VAE decoder."""
import math

import torch

from infiniop.ops.conv1d.ninetoothed.conv1d import premake as conv1d_premake
from infiniop.ops.conv1d.ninetoothed.depthwise import premake as depthwise_premake
from infiniop.ops.vae_support.ninetoothed.support import premake_weight_norm
from .ninetoothed_vae_encoder import (
    NineToothedVAEEncoderMode,
    _DTYPES,
    _kernel,
    pointwise,
)


class NineToothedVAEAudioMode(NineToothedVAEEncoderMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if name == "aten._weight_norm.default":
            self.operations[name] += 1
            weight, gain = args[:2]
            if gain.dtype != weight.dtype:
                raise NotImplementedError("Weight-norm gain and weight must have matching dtypes")
            dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
            output = torch.empty_like(weight)
            _kernel(premake_weight_norm, weight.ndim, dim=dim,
                    dtype=_DTYPES[weight.dtype])(weight, gain, output)
            return output
        if name not in ("aten.conv1d.default", "aten.conv_transpose1d.default"):
            return super().__torch_dispatch__(func, types, args, kwargs)
        self.operations[name] += 1
        transpose = name == "aten.conv_transpose1d.default"
        input, weight = args[:2]
        bias = args[2] if len(args) > 2 else kwargs.get("bias")
        stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
        padding = args[4] if len(args) > 4 else kwargs.get("padding", 0)
        if transpose:
            output_padding = args[5] if len(args) > 5 else kwargs.get("output_padding", 0)
            groups = args[6] if len(args) > 6 else kwargs.get("groups", 1)
            dilation = args[7] if len(args) > 7 else kwargs.get("dilation", 1)
        else:
            dilation = args[5] if len(args) > 5 else kwargs.get("dilation", 1)
            groups = args[6] if len(args) > 6 else kwargs.get("groups", 1)
            output_padding = 0

        def scalar(value):
            return value if isinstance(value, int) else value[0]
        stride, padding, dilation, output_padding = map(
            scalar, (stride, padding, dilation, output_padding))
        dtype = (torch.get_autocast_dtype("cuda")
                 if torch.is_autocast_enabled("cuda") else input.dtype)
        if input.dtype != dtype:
            input = pointwise(input, 0, dtype=dtype)
        if weight.dtype != dtype:
            weight = pointwise(weight, 0, dtype=dtype)
        if bias is not None and bias.dtype != dtype:
            bias = pointwise(bias, 0, dtype=dtype)
        batch, channels, length = input.shape
        kernel_size = weight.shape[-1]
        output_channels = weight.shape[1] * groups if transpose else weight.shape[0]
        output_length = ((length - 1) * stride - 2 * padding + dilation * (kernel_size - 1)
                         + output_padding + 1 if transpose else
                         (length + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1)
        output = torch.empty((batch, output_channels, output_length),
                             device=input.device, dtype=dtype)
        if bias is None:
            bias = torch.zeros(output_channels, device=input.device, dtype=dtype)
        if groups == channels == output_channels and weight.shape[1] == 1:
            if not transpose:
                _kernel(depthwise_premake, kernel_size, _DTYPES[dtype],
                        stride=stride, pad_left=padding, pad_right=padding,
                        dilation=dilation, round_before_bias=True)(input, weight, bias, output)
            else:
                divisor = math.gcd(stride, dilation)
                for phase in range(stride):
                    target = output[..., phase::stride]
                    if target.numel() == 0:
                        continue
                    indices = [k for k in range(kernel_size)
                               if (phase + padding - k * dilation) % stride == 0]
                    if not indices:
                        target.copy_(bias[None, :, None].expand(target.shape))
                        continue
                    last = indices[-1]
                    left = -(phase + padding - last * dilation) // stride
                    step = dilation // divisor
                    right = target.shape[-1] - length - left + step * (len(indices) - 1)
                    _kernel(depthwise_premake, len(indices), _DTYPES[dtype],
                            stride=1, pad_left=left, pad_right=right, dilation=step,
                            transpose=True, kernel_last=last,
                            kernel_period=stride // divisor, round_before_bias=True)(input, weight, bias, target)
            return output
        for group in range(groups):
            x = input[:, group * (channels // groups):(group + 1) * (channels // groups)]
            y = output[:, group * (output_channels // groups):(group + 1) * (output_channels // groups)]
            b = bias[group * (output_channels // groups):(group + 1) * (output_channels // groups)]
            if not transpose:
                w = weight[group * (output_channels // groups):(group + 1) * (output_channels // groups)]
                _kernel(conv1d_premake, _DTYPES[dtype], stride=stride,
                        pad_left=padding, pad_right=padding, dilation=dilation,
                        round_before_bias=True)(x, w, b, y, 2)
                continue
            w = weight[group * (channels // groups):(group + 1) * (channels // groups)]
            divisor = math.gcd(stride, dilation)
            period = stride // divisor
            for phase in range(stride):
                indices = [k for k in range(kernel_size)
                           if (phase + padding - k * dilation) % stride == 0]
                target = y[..., phase::stride]
                if target.numel() == 0:
                    continue
                if not indices:
                    target.copy_(b[None, :, None].expand(target.shape))
                    continue
                last = indices[-1]
                pad_left = -(phase + padding - last * dilation) // stride
                step = dilation // divisor
                pad_right = target.shape[-1] - length - pad_left + step * (len(indices) - 1)
                _kernel(conv1d_premake, _DTYPES[dtype], stride=1,
                        pad_left=pad_left, pad_right=pad_right, dilation=step,
                        transpose=True, kernel_last=last, kernel_period=period,
                        round_before_bias=True)(x, w, b, target, 2)
        return output
