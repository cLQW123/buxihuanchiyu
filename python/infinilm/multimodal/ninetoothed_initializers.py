"""Load-time VAE buffers; no Torch numerical initializer is used."""
import functools
import math
import torch

from infiniop.ops.vae_support.ninetoothed.initializers import (
    premake_rotary, premake_filter, premake_normalize)
from .ninetoothed_vae_encoder import _kernel


def rotary_init(self, dim, theta=100.0, num_axes=3):
    torch.nn.Module.__init__(self)
    if dim % (2 * num_axes):
        raise ValueError('Rotary dimensions must divide evenly over the axes')
    count = dim // (2 * num_axes)
    output = torch.empty(count, device='cuda', dtype=torch.float32)
    _kernel(premake_rotary, count, float(theta))(output, count, float(theta))
    self.register_buffer('inv_freq', output.cpu(), persistent=False)


@functools.cache
def _filter_buffer(cutoff, half_width, kernel_size):
    if not 1 < kernel_size <= 128:
        raise NotImplementedError('H3 Kaiser filter supports 2..128 taps')
    attenuation = 2.285 * (kernel_size // 2 - 1) * math.pi * (4 * half_width) + 7.95
    if attenuation > 50:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21)
    else:
        beta = 0.0
    unscaled = torch.empty(kernel_size, device='cuda', dtype=torch.float32)
    output = torch.empty_like(unscaled)
    _kernel(premake_filter, kernel_size, float(cutoff), beta)(
        unscaled, kernel_size, float(cutoff), beta)
    _kernel(premake_normalize)(unscaled, output)
    return output.cpu().view(1, 1, kernel_size)


def kaiser_sinc_filter1d(cutoff, half_width, kernel_size):
    return _filter_buffer(cutoff, half_width, kernel_size).clone()
