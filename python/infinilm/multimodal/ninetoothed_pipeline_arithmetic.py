"""NineToothed arithmetic candidates for H3 pipeline boundaries.

These helpers preserve intermediate storage casts instead of fusing across
reference rounding boundaries. Integration, random-number generation, position
construction and the complete pipeline still need separate validation.
"""
import torch
import ninetoothed as nt

from infiniop.ops.vae_support.ninetoothed.support import premake_pointwise
from .ninetoothed_vae_encoder import _kernel, pointwise


def _place(value, device):
    return value if value.device == device else value.to(device=device)


def _check_float(*values):
    if any(value.dtype not in (torch.float16, torch.bfloat16, torch.float32)
           for value in values):
        raise NotImplementedError("Pipeline arithmetic currently supports F16/BF16/F32")


def encode_pixels(pixels, mean, std):
    if pixels.dtype == torch.uint8:
        converted = torch.empty(pixels.shape, device=pixels.device, dtype=torch.float32)
        _kernel(premake_pointwise, pixels.ndim, 0, input_dtype=nt.uint8,
                other_dtype=nt.uint8, output_dtype=nt.float32)(
                    pixels, pixels, converted, 0, -float("inf"), float("inf"))
    else:
        _check_float(pixels)
        converted = pointwise(pixels, 0, dtype=torch.float32)
    converted = pointwise(converted, 4, 255.0)
    return normalize_latents(converted, mean, std, device=pixels.device)


def normalize_latents(latents, mean, std, *, device):
    """Compute (latents - mean) / std on the accelerator, then restore device."""
    _check_float(latents, mean, std)
    original_device = latents.device
    result = pointwise(_place(latents, device), 3, _place(mean, device))
    result = pointwise(result, 4, _place(std, device))
    return _place(result, original_device)


def denormalize_latents(latents, mean, std):
    _check_float(latents, mean, std)
    result = pointwise(latents, 2, _place(std, latents.device))
    return pointwise(result, 1, _place(mean, latents.device))


def combine_posterior(mean, std, noise):
    """Combine supplied random noise; this helper does not implement RNG."""
    _check_float(mean, std, noise)
    scaled = pointwise(std, 2, _place(noise, std.device))
    return pointwise(mean, 1, scaled)


def decode_pixels(video, mean, std):
    _check_float(video, mean, std)
    video = pointwise(video, 0, dtype=torch.float32)
    video = pointwise(video, 2, _place(std, video.device))
    video = pointwise(video, 1, _place(mean, video.device))
    return pointwise(video, 13, lower=0.0, upper=1.0)


def scheduler_interpolate(sample, denoised, sigma, sigma_next):
    """Reference scheduler interpolation with separate FP32 operations."""
    _check_float(sample, denoised, sigma, sigma_next)
    device = sample.device
    current = pointwise(_place(sigma, device), 0, dtype=torch.float32)
    following = pointwise(_place(sigma_next, device), 0, dtype=torch.float32)
    ratio = pointwise(following, 4, current)
    residual_ratio = pointwise(ratio, 5, 1.0)
    current_sample = pointwise(sample, 0, dtype=torch.float32)
    prediction = pointwise(denoised, 0, dtype=torch.float32)
    left = pointwise(ratio, 2, current_sample)
    right = pointwise(residual_ratio, 2, prediction)
    output = pointwise(left, 1, right)
    return pointwise(output, 0, dtype=sample.dtype)


def scheduler_denoise(sample, model_output, timestep):
    _check_float(sample, model_output, timestep)
    timestep = pointwise(_place(timestep, sample.device), 0, dtype=sample.dtype)
    sigma = pointwise(timestep, 5, 1.0)
    while sigma.ndim < sample.ndim:
        sigma = sigma.unsqueeze(-1)
    scaled = pointwise(sigma, 2, model_output)
    return pointwise(sample, 1, scaled)
