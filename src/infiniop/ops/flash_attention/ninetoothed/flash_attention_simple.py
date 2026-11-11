"""NineToothed attention with one query-head tile per program."""
import functools
import ninetoothed.language as ntl
from .flash_attention import premake as original_premake


def arrangement(query, key, value, total_kv_len, present_key, present_value,
                present_key_slot, present_value_slot, attn_mask, is_causal,
                scale, output, with_attn_mask, causal_variant, *,
                block_size_m, block_size_n):
    def arrange_q(tensor):
        result = tensor.tile((1, 1, block_size_m, -1))
        result.dtype = result.dtype.squeeze((0, 1))
        return result

    q = arrange_q(query)

    def arrange_kv(tensor):
        # Broadcast a KV head over its query-head group without materializing it.
        repeated = tensor.unsqueeze(2).expand(
            (-1, -1, query.shape[1] // key.shape[1], -1, -1)
        ).flatten(1, 3)
        result = repeated.tile((1, 1, block_size_n, -1)).tile((1, 1, -1, -1))
        result = result.expand((-1, -1, q.shape[2], -1))
        result.dtype = result.dtype.squeeze((0, 1, 3))
        result.dtype.dtype = result.dtype.dtype.squeeze((0, 1))
        return result

    lengths = total_kv_len.tile((1,)).unsqueeze(1).unsqueeze(2).unsqueeze(3)
    lengths = lengths.expand((-1, *q.shape[1:]))
    return (q, arrange_kv(key), arrange_kv(value), lengths, attn_mask,
            is_causal, scale, arrange_q(output), with_attn_mask, causal_variant)


def application(query, key, value, total_kv_len, attn_mask, is_causal,
                scale, output, with_attn_mask, causal_variant):
    length = total_kv_len[0]
    q = (query * (1.4426950408889634 * scale)).to(query.dtype)
    accumulator = ntl.zeros((q.shape[0], q.shape[1]), dtype=ntl.float32)
    normalizer = ntl.full((q.shape[0],), 1, dtype=ntl.float32)
    maximum = ntl.full((q.shape[0],), float("-inf"), dtype=ntl.float32)
    for block in range((length + key.dtype.shape[0] - 1) // key.dtype.shape[0]):
        scores = ntl.dot(q, ntl.trans(key[block]))
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
        accumulator = accumulator * alpha[:, None] + ntl.dot(weights.to(value[block].dtype), value[block])
        normalizer = normalizer * alpha + ntl.sum(weights, 1)
        maximum = next_maximum
    output = accumulator / normalizer[:, None]


def premake(**config):
    if config["with_kv_cache"] or config["with_attn_mask"]:
        raise ValueError("This attention variant requires a separate KV update and no explicit mask")
    _, _, tensors = original_premake(**config)
    return functools.partial(arrangement, block_size_m=config["block_size_m"],
                             block_size_n=config["block_size_n"]), application, tensors
