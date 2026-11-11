"""Experimental strict NineToothed execution for the H3 video VAE decoder."""
import math

import ninetoothed as nt
import torch

from infiniop.ops.flash_attention.ninetoothed.flash_attention_scaled import premake as attention_premake
from infiniop.ops.matmul.ninetoothed.matmul_accumulate_f32 import premake as matrix_premake
from infiniop.ops.vae_support.ninetoothed.normalization import premake as norm_premake
from .ninetoothed_vae_encoder import NineToothedVAEEncoderMode, _DTYPES, _kernel, pointwise


class NineToothedVAEVideoMode(NineToothedVAEEncoderMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = str(func)
        if name == "aten.dropout.default":
            self.operations[name] += 1
            probability = args[1] if len(args) > 1 else kwargs.get("p", 0.5)
            training = args[2] if len(args) > 2 else kwargs.get("train", True)
            if training and probability:
                raise NotImplementedError("Dropout must be inactive during VAE inference")
            return args[0]
        if name in ("aten.rms_norm.default", "aten.layer_norm.default"):
            self.operations[name] += 1
            x, shape = args[:2]
            width = math.prod(shape)
            centered = name == "aten.layer_norm.default"
            weight = args[2] if len(args) > 2 else kwargs.get("weight")
            if centered:
                bias = args[3] if len(args) > 3 else kwargs.get("bias")
                eps = args[4] if len(args) > 4 else kwargs.get("eps", 1e-5)
            else:
                bias = None
                eps = args[3] if len(args) > 3 else kwargs.get("eps")
                eps = torch.finfo(x.dtype).eps if eps is None else eps
            if weight is None:
                weight = torch.ones(width, device=x.device, dtype=x.dtype)
            if bias is None:
                bias = torch.zeros(width, device=x.device, dtype=x.dtype)
            output = torch.empty_like(x, memory_format=torch.contiguous_format)
            _kernel(norm_premake, width, centered, _DTYPES[x.dtype],
                    weight_dtype=_DTYPES[weight.dtype], bias_dtype=_DTYPES[bias.dtype],
                    output_dtype=_DTYPES[x.dtype])(
                        x.contiguous().view(-1, width), weight.reshape(width), bias.reshape(width),
                        output.view(-1, width), eps, int(centered))
            return output
        if name == "aten.linear.default":
            self.operations[name] += 1
            x, weight = args[:2]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            dtype = (torch.get_autocast_dtype("cuda")
                     if torch.is_autocast_enabled("cuda") else x.dtype)
            if dtype != torch.bfloat16:
                raise NotImplementedError("This linear candidate currently requires BF16 compute")
            if x.dtype != dtype:
                x = pointwise(x, 0, dtype=dtype)
            if weight.dtype != dtype:
                weight = pointwise(weight, 0, dtype=dtype)
            if bias is not None and bias.dtype != dtype:
                bias = pointwise(bias, 0, dtype=dtype)
            flat = x.contiguous().view(-1, weight.shape[1])
            accumulator = torch.empty((flat.shape[0], weight.shape[0]),
                                      device=x.device, dtype=torch.float32)
            _kernel(matrix_premake)(flat, weight.t(), accumulator, 2)
            if bias is not None:
                accumulator = pointwise(accumulator, 1, bias)
            output = pointwise(accumulator, 0, dtype=dtype)
            return output.view(*x.shape[:-1], weight.shape[0])
        if name == "aten.scaled_dot_product_attention.default":
            self.operations[name] += 1
            query, key, value = args[:3]
            mask = args[3] if len(args) > 3 else kwargs.get("attn_mask")
            probability = args[4] if len(args) > 4 else kwargs.get("dropout_p", 0)
            causal = args[5] if len(args) > 5 else kwargs.get("is_causal", False)
            scale = kwargs.get("scale")
            scale = query.shape[-1] ** -0.5 if scale is None else scale
            if (mask is not None or probability or query.dtype != torch.bfloat16 or
                key.dtype != query.dtype or value.dtype != query.dtype or
                query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or
                query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0] or
                key.shape[1] != value.shape[1] or query.shape[1] % key.shape[1] or
                query.shape[-1] != key.shape[-1] or query.shape[-1] != value.shape[-1]):
                raise NotImplementedError("Only mask-free BF16 inference attention is supported")
            output = torch.empty_like(query)
            lengths = torch.full((query.shape[0],), key.shape[-2],
                                  device=query.device, dtype=torch.int32)
            unused_mask = torch.empty((1, 1, 1, 1), device=query.device, dtype=query.dtype)
            _kernel(attention_premake, with_kv_cache=0, emb_dim=query.shape[-1],
                    is_causal=int(causal), with_attn_mask=0, causal_variant=1,
                    dtype=nt.bfloat16, block_size_m=32, block_size_n=32)(
                        query, key, value, lengths, unused_mask, int(causal), scale,
                        output, 0, 1)
            return output
        return super().__torch_dispatch__(func, types, args, kwargs)
