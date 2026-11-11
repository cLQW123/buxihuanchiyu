import os

import ninetoothed

import infiniop.ninetoothed.build

from . import flash_attention
from .flash_attention import CausalVariant


def build():

    env_vars_to_check = ["MACA_HOME", "MACA_PATH", "MACA_ROOT"]
    if any(var in os.environ for var in env_vars_to_check):
        return

    if os.getenv("INFINICORE_NINETOOTHED_PROFILE") == "fl2va":
        from .flash_attention_scaled import premake
        return infiniop.ninetoothed.build.build(
            premake,
            dict(with_kv_cache=(0,), emb_dim=(64, 72, 128), is_causal=(0, 1),
                 with_attn_mask=(0,), causal_variant=(2,),
                 dtype=(ninetoothed.bfloat16,), block_size_m=(32,), block_size_n=(32,)),
            caller="cuda", op_name="flash_attention",
            output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH)

    with_kv_cache_values = (0,)
    emb_dim_values = (16, 32, 64, 128, 256)
    is_causal_values = (0, 1)
    with_attn_mask_values = (0,)
    causal_variant_values = (CausalVariant.UPPER_LEFT, CausalVariant.LOWER_RIGHT)
    dtype_values = (ninetoothed.float16, ninetoothed.bfloat16, ninetoothed.float32)
    block_size_m_values = (256,)
    block_size_n_values = (64,)

    constexpr_param_grid = {
        "with_kv_cache": with_kv_cache_values,
        "emb_dim": emb_dim_values,
        "is_causal": is_causal_values,
        "with_attn_mask": with_attn_mask_values,
        "causal_variant": causal_variant_values,
        "dtype": dtype_values,
        "block_size_m": block_size_m_values,
        "block_size_n": block_size_n_values,
    }

    infiniop.ninetoothed.build.build(
        flash_attention.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="flash_attention",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
