import os

import ninetoothed

import infiniop.ninetoothed.build

from . import kv_caching


def build():
    dtype_by_name = {
        "float16": ninetoothed.float16,
        "bfloat16": ninetoothed.bfloat16,
        "float32": ninetoothed.float32,
    }
    requested_dtypes = os.getenv("INFINICORE_KV_CACHING_DTYPES")
    dtype_values = (
        tuple(dtype_by_name[name] for name in requested_dtypes.split(","))
        if requested_dtypes
        else tuple(dtype_by_name.values())
    )
    requested_dims = os.getenv("INFINICORE_KV_CACHING_EMB_DIMS")
    emb_dim_values = (
        tuple(int(dim) for dim in requested_dims.split(","))
        if requested_dims
        else (1, 16, 32, 64, 128, 256)
    )

    constexpr_param_grid = {
        "emb_dim": emb_dim_values,
        "dtype": dtype_values,
        "block_size_m": (64,),
        "block_size_n": (64,),
    }

    infiniop.ninetoothed.build.build(
        kv_caching.premake,
        constexpr_param_grid,
        caller="cuda",
        op_name="kv_caching",
        output_dir=infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH,
    )
