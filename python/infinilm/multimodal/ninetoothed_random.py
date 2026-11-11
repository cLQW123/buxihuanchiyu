"""NineToothed normal draws with the pinned PyTorch CPU stream layout."""
import math
import struct

import torch

from infiniop.ops.vae_support.ninetoothed.random import premake_normal, premake_stream
from .ninetoothed_vae_encoder import _DTYPES, _kernel


def randn_tensor(shape, generator=None, device=None, dtype=None, layout=None):
    device = torch.device(device or "cuda")
    dtype = dtype or torch.float32
    if isinstance(generator, list):
        if len(generator) != shape[0]:
            raise ValueError("Expected one CPU generator per batch member")
        return torch.cat([randn_tensor((1, *shape[1:]), item, device, dtype, layout)
                          for item in generator])
    if generator is None or generator.device.type != "cpu":
        raise NotImplementedError("The competition entry uses an explicit CPU generator")
    if device.type != "cuda" or dtype not in _DTYPES:
        raise NotImplementedError("NineToothed random requires CUDA/HIP and F32/F16/BF16")
    count = math.prod(shape)
    if count < 16:
        raise NotImplementedError("Scalar cached-normal RNG is outside the FL2VA tensor path")
    state_bytes = bytearray(generator.get_state().numpy().tobytes())
    if len(state_bytes) != 5056:
        raise RuntimeError("Unsupported CPU generator state ABI")
    _, left, seeded, cursor = struct.unpack_from("<QiiQ", state_bytes)
    if not seeded or not 1 <= left <= 624 or not 0 <= cursor <= 624:
        raise RuntimeError("Invalid CPU MT19937 state")
    state_words = struct.unpack_from("<624Q", state_bytes, 24)
    state_storage = bytearray(struct.pack("<624I", *state_words))
    state = torch.frombuffer(state_storage, dtype=torch.uint32).to(device)
    consumed = count + (16 if count % 16 else 0)
    raw = torch.empty(consumed, dtype=torch.uint32, device=device)
    initial_twist = left == 1
    _kernel(premake_stream, cursor, initial_twist)(state, raw, cursor, initial_twist)
    output = torch.empty(count, dtype=dtype, device=device)
    digits = {torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}[dtype]
    _kernel(premake_normal, count, _DTYPES[dtype])(raw, output, count, digits)
    # Copy the new stream state back into the same generator. This preserves
    # draw order when a later pipeline block uses the request generator.
    returned_words = struct.unpack("<624I", state.cpu().numpy().tobytes())
    struct.pack_into("<624Q", state_bytes, 24, *returned_words)
    final_cursor = ((0 if initial_twist else cursor) + consumed - 1) % 624 + 1
    struct.pack_into("<i", state_bytes, 8, 625 - final_cursor)
    struct.pack_into("<Q", state_bytes, 16, final_cursor)
    generator.set_state(torch.frombuffer(state_bytes, dtype=torch.uint8).clone())
    return output.reshape(shape)
