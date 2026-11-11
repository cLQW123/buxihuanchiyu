"""Experimental VAE encoder dispatch using NineToothed numerical kernels.

Opt-in component adapter. It fails on unsupported operations, so an incomplete
port cannot silently execute a Torch model kernel. Storage/view/constant
construction remains in Torch; numerical operations are handled explicitly.
"""
from collections import Counter
import functools
import os

import ninetoothed as nt
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from infiniop.ops.conv3d.ninetoothed.conv3d import premake as conv3d_premake
from infiniop.ops.group_norm.ninetoothed.group_norm import premake as group_norm_premake
from infiniop.ops.vae_support.ninetoothed.support import (
    premake_padding,
    premake_pointwise,
)

_DTYPES = {torch.float32: nt.float32, torch.bfloat16: nt.bfloat16,
           torch.float16: nt.float16}


@functools.cache
def _prepare_compiler():
    if os.getenv("NINETOOTHED_HYGON_LLVM_COMPAT") == "1":
        from ninetoothed._compile_triton import install_hygon_llvm_compatibility
        install_hygon_llvm_compatibility()


@functools.cache
def _kernel(premake, *args, **kwargs):
    _prepare_compiler()
    return nt.make(*premake(*args, **kwargs), num_warps=4, num_stages=1,
                   max_num_configs=1)


def pointwise(input, operation, other=None, *, dtype=None,
              lower=-float("inf"), upper=float("inf")):
    dtype = dtype or (torch.result_type(input, other) if other is not None else input.dtype)
    if other is None:
        other = input
    elif not isinstance(other, torch.Tensor):
        # Scalar arithmetic uses FP32 opmath before the result is stored.
        other = torch.full((), other, device=input.device, dtype=torch.float32)
    shape = torch.broadcast_shapes(input.shape, other.shape)
    scalar_result = len(shape) == 0
    if scalar_result:
        # The elementwise arrangement needs a launch dimension. Represent one
        # scalar as a single storage element, then restore its scalar view.
        shape = (1,)
        input, other = input.reshape(shape), other.reshape(shape)
    input, other = input.expand(shape), other.expand(shape)
    result = torch.empty(shape, dtype=dtype, device=input.device)
    _kernel(premake_pointwise, len(shape), operation,
            input_dtype=_DTYPES[input.dtype], other_dtype=_DTYPES[other.dtype],
            output_dtype=_DTYPES[dtype], lower=lower, upper=upper)(
                input, other, result, operation, lower, upper)
    return result.reshape(()) if scalar_result else result


class NineToothedVAEEncoderMode(TorchDispatchMode):
    """Use only with an encoder until component/model validation has passed."""

    _STORAGE = {
        "aten.alias.default", "aten.detach.default", "aten.slice.Tensor",
        "aten.select.int", "aten.permute.default", "aten.view.default",
        "aten.reshape.default", "aten._unsafe_view.default", "aten.expand.default",
        "aten.unsqueeze.default", "aten.squeeze.dim", "aten.squeeze.default",
        "aten.split.Tensor", "aten.split_with_sizes.default", "aten.chunk.default",
        "aten.cat.default", "aten.stack.default", "aten.contiguous.default",
        "aten.clone.default", "aten.empty_like.default", "aten.empty.memory_format",
        "aten.empty_strided.default", "aten.zeros_like.default", "aten.zeros.default",
        "aten.ones.default", "aten.ones_like.default", "aten.full.default",
        "aten.full_like.default", "aten.new_empty.default", "aten.new_zeros.default",
        "aten.arange.default", "aten.arange.start", "aten.arange.start_step",
        "aten.lift_fresh.default", "aten.repeat.default",
        "aten.unbind.int", "aten.flatten.using_ints", "aten.unflatten.int",
        "aten.tile.default", "aten.meshgrid.indexing",
    }

    def __init__(self):
        super().__init__()
        self.operations = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        self.operations[name] += 1
        if name == "aten.copy_.default":
            destination, source = args[:2]
            if destination.dtype == source.dtype:
                return func(*args, **kwargs)
            source = source.expand(destination.shape)
            _kernel(premake_pointwise, destination.ndim, 0,
                    input_dtype=_DTYPES[source.dtype],
                    other_dtype=_DTYPES[source.dtype],
                    output_dtype=_DTYPES[destination.dtype])(
                        source, source, destination, 0, -float("inf"), float("inf"))
            return destination
        if name in self._STORAGE:
            return func(*args, **kwargs)
        if name == "aten.to.dtype":
            x, dtype = args[:2]
            return x if x.dtype == dtype else pointwise(x, 0, dtype=dtype)
        if name == "aten._to_copy.default":
            x = args[0]
            dtype = kwargs.get("dtype", x.dtype)
            if kwargs.get("device", x.device) != x.device:
                raise NotImplementedError("Numerical mode requires preplaced VAE storage")
            return pointwise(x, 0, dtype=dtype)
        binary = {"aten.add.Tensor": 1, "aten.mul.Tensor": 2,
                  "aten.sub.Tensor": 3, "aten.div.Tensor": 4,
                  "aten.rsub.Scalar": 5, "aten.add.Scalar": 1,
                  "aten.mul.Scalar": 2, "aten.sub.Scalar": 3,
                  "aten.div.Scalar": 4}
        if name in binary:
            if kwargs.get("alpha", 1) != 1:
                raise NotImplementedError("Non-unit add/sub alpha")
            return pointwise(args[0], binary[name], args[1])
        unary = {"aten.exp.default": 6, "aten.sin.default": 7,
                 "aten.cos.default": 8, "aten.neg.default": 9,
                 "aten.silu.default": 10, "aten.square.default": 11,
                 "aten.reciprocal.default": 12}
        if name in unary:
            return pointwise(args[0], unary[name])
        if name == "aten.pow.Tensor_Scalar" and args[1] == 2:
            return pointwise(args[0], 11)
        if name == "aten.clamp.default":
            lower = args[1] if len(args) > 1 else kwargs.get("min")
            upper = args[2] if len(args) > 2 else kwargs.get("max")
            return pointwise(args[0], 13, lower=-float("inf") if lower is None else lower,
                             upper=float("inf") if upper is None else upper)
        if name == "aten.pad.default":
            x, pad = args[:2]
            mode = args[2] if len(args) > 2 else kwargs.get("mode", "constant")
            value = args[3] if len(args) > 3 else kwargs.get("value")
            if value not in (None, 0):
                raise NotImplementedError("Only zero constant padding is supported")
            full = [(0, 0)] * x.ndim
            for index in range(len(pad) // 2):
                full[-1-index] = tuple(pad[2*index:2*index+2])
            full = tuple(full)
            output = torch.empty(tuple(n + sum(p) for n, p in zip(x.shape, full)),
                                 device=x.device, dtype=x.dtype)
            _kernel(premake_padding, x.ndim, full, mode, _DTYPES[x.dtype])(x, output)
            return output
        if name == "aten.group_norm.default":
            x, groups, weight, bias = args[:4]
            eps = args[4] if len(args) > 4 else kwargs.get("eps", 1e-5)
            batch, channels = x.shape[:2]
            grouped = x.contiguous().view(batch, groups, -1)
            output = torch.empty_like(grouped)
            if weight is None:
                weight = torch.ones(channels, device=x.device, dtype=x.dtype)
            if bias is None:
                bias = torch.zeros(channels, device=x.device, dtype=x.dtype)
            if weight.dtype != x.dtype or bias.dtype != x.dtype:
                raise NotImplementedError("GroupNorm affine tensors must match the input dtype")
            mean = torch.empty((batch, groups, 1), device=x.device, dtype=torch.float32)
            rstd = torch.empty_like(mean)
            _kernel(group_norm_premake, _DTYPES[x.dtype], eps=eps)(
                grouped, weight.view(groups, -1), bias.view(groups, -1), output,
                mean, rstd, eps)
            return output.view(x.shape)
        if name == "aten.conv3d.default":
            x, weight = args[:2]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            stride = args[3] if len(args) > 3 else kwargs.get("stride", (1, 1, 1))
            padding = args[4] if len(args) > 4 else kwargs.get("padding", (0, 0, 0))
            dilation = args[5] if len(args) > 5 else kwargs.get("dilation", (1, 1, 1))
            groups = args[6] if len(args) > 6 else kwargs.get("groups", 1)
            def triple(value):
                return (value,) * 3 if isinstance(value, int) else tuple(value)
            stride, padding, dilation = map(triple, (stride, padding, dilation))
            if groups != 1 or weight.dtype != x.dtype:
                raise NotImplementedError("Dense same-dtype Conv3D required")
            spatial = tuple((n + 2*p - d*(k-1) - 1) // s + 1
                            for n, p, d, k, s in zip(x.shape[2:], padding,
                                                     dilation, weight.shape[2:], stride))
            output = torch.empty((x.shape[0], weight.shape[0], *spatial),
                                 device=x.device, dtype=x.dtype)
            if bias is None:
                bias = torch.zeros(weight.shape[0], device=x.device, dtype=x.dtype)
            _kernel(conv3d_premake, stride=stride, padding=padding, dilation=dilation,
                    dtype=_DTYPES[x.dtype], round_before_bias=True)(x, weight, bias, output, 2)
            return output
        raise NotImplementedError(f"VAE encoder operation is not ported: {name}")
