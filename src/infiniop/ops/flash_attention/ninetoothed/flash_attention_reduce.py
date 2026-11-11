"""NineToothed attention using FP32 reductions for the two block products."""
import ninetoothed.language as ntl
from .flash_attention_simple import premake as tile_premake


def application(query, key, value, total_kv_len, attn_mask, is_causal,
                scale, output, with_attn_mask, causal_variant):
    length = total_kv_len[0]
    q = (query * (1.4426950408889634 * scale)).to(query.dtype)
    accumulator = ntl.zeros((q.shape[0], q.shape[1]), dtype=ntl.float32)
    normalizer = ntl.full((q.shape[0],), 1, dtype=ntl.float32)
    maximum = ntl.full((q.shape[0],), float("-inf"), dtype=ntl.float32)
    for block in range((length + key.dtype.shape[0] - 1) // key.dtype.shape[0]):
        scores = ntl.sum(
            ntl.cast(q, ntl.float32)[:, None, :]
            * ntl.cast(key[block], ntl.float32)[None, :, :], 2)
        key_pos = key[block].offsets(-2)
        scores = ntl.where(key_pos[None, :] < length, scores, float("-inf"))
        if is_causal:
            query_pos = query.offsets(-2)
            if causal_variant == 2:
                causal_mask = query_pos[:, None] + length - query.source.shape[-2] >= key_pos[None, :]
            else:
                causal_mask = query_pos[:, None] >= key_pos[None, :]
            scores = ntl.where(causal_mask, scores, float("-inf"))
        next_maximum = ntl.maximum(maximum, ntl.max(scores, 1))
        weights = ntl.exp2(scores - next_maximum[:, None])
        alpha = ntl.exp2(maximum - next_maximum)
        contribution = ntl.sum(
            ntl.cast(weights.to(value[block].dtype), ntl.float32)[:, :, None]
            * ntl.cast(value[block], ntl.float32)[None, :, :], 1)
        accumulator = accumulator * alpha[:, None] + contribution
        normalizer = normalizer * alpha + ntl.sum(weights, 1)
        maximum = next_maximum
    output = accumulator / normalizer[:, None]


def premake(**config):
    arrangement, _, tensors = tile_premake(**config)
    return arrangement, application, tensors
