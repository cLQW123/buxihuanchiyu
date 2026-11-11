"""NineToothed FL2VA boundary arithmetic, RNG, schedule and rotary layout.

Neural VAE dispatch and the native InfiniLM engine are installed separately.
"""
from collections import Counter
import functools
import hashlib
import inspect
import json
from pathlib import Path

import torch

from . import ninetoothed_pipeline_arithmetic as arithmetic
from .ninetoothed_vae_encoder import pointwise, _kernel
from .ninetoothed_random import randn_tensor
from .ninetoothed_positions import build_packed_sequence
from .ninetoothed_initializers import rotary_init, kaiser_sinc_filter1d
from infiniop.ops.vae_support.ninetoothed.grid import premake as premake_grid


def install_pipeline_arithmetic(evidence_path):
    from diffusers.modular_pipelines.minimax_h3 import encoders, decoders, before_denoise
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as video_vae
    from diffusers.models.autoencoders import autoencoder_kl_minimax_h3_audio as audio_vae
    from diffusers.schedulers.scheduling_minimax_h3 import (
        MiniMaxH3Scheduler,
        MiniMaxH3SchedulerOutput,
    )

    evidence = Path(evidence_path)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    replacements = []
    source_hashes = {}

    def save():
        evidence.write_text(json.dumps({
            "scope": "pipeline arithmetic integration; neural VAE installed separately",
            "rule_compliance": "requires full-generation and source-build evidence",
            "pending": ["full-generation validation", "initialization-constant audit"],
            "calls": dict(counts),
            "original_source_sha256": source_hashes,
        }, indent=2))

    def replace(owner, name, function, *, static=False):
        original = getattr(owner, name)
        descriptor = inspect.getattr_static(owner, name) if static else original
        label = owner.__name__ + "." + name
        source_hashes[label] = hashlib.sha256(inspect.getsource(original).encode()).hexdigest()

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            counts[label] += 1
            try:
                return function(*args, **kwargs)
            finally:
                save()

        setattr(owner, name, staticmethod(wrapped) if static else wrapped)
        replacements.append((owner, name, descriptor))

    def encode_condition(vae, pixels, pixel_mean, pixel_std, encode_seed=42):
        device = pixels.device
        pixel_mean = torch.tensor(pixel_mean, device=device).view(1, -1, 1, 1, 1)
        pixel_std = torch.tensor(pixel_std, device=device).view(1, -1, 1, 1, 1)
        pixels = arithmetic.encode_pixels(pixels, pixel_mean, pixel_std)
        posterior = vae.encode(pixels, return_dict=False)[0]
        noise = randn_tensor(posterior.mean.shape,
                             generator=torch.Generator().manual_seed(encode_seed),
                             device=posterior.mean.device, dtype=posterior.mean.dtype)
        counts["NineToothedRNG.posterior"] += 1
        latent = arithmetic.combine_posterior(posterior.mean, posterior.std, noise)
        latent = pointwise(latent, 0, dtype=torch.float16)
        latent = pointwise(latent, 0, dtype=torch.float32).cpu()
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        return arithmetic.normalize_latents(latent, mean, std, device=device)

    def video_decode(self, components, state):
        block = self.get_block_state(state)
        if block.output_type not in ("pil", "np", "pt"):
            raise ValueError("Video output_type must be 'pil', 'np' or 'pt'")
        device = components._execution_device
        mean = torch.tensor(components.vae.config.latents_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(components.vae.config.latents_std, device=device).view(1, -1, 1, 1, 1)
        latent = arithmetic.denormalize_latents(block.latents, mean, std)
        with torch.autocast(device_type=device.type, dtype=torch.float16,
                            enabled=device.type == "cuda"):
            video = components.vae.decode(latent, return_dict=False)[0]
        mean = torch.tensor(components.pixel_mean, device=device).view(1, -1, 1, 1, 1)
        std = torch.tensor(components.pixel_std, device=device).view(1, -1, 1, 1, 1)
        video = arithmetic.decode_pixels(video, mean, std)
        block.videos = components.video_processor.postprocess_video(video, output_type=block.output_type)
        self.set_block_state(state, block)
        return components, state

    def audio_decode(self, components, state):
        block = self.get_block_state(state)
        device = components._execution_device
        mean = torch.tensor(components.audio_vae.config.latents_mean, device=device).view(1, -1, 1)
        std = torch.tensor(components.audio_vae.config.latents_std, device=device).view(1, -1, 1)
        latent = arithmetic.denormalize_latents(block.audio_latents, mean, std)
        audio = components.audio_vae.decode(latent, return_dict=False)[0]
        block.audio = pointwise(audio, 0, dtype=torch.float32).permute(1, 0, 2)
        block.sampling_rate = components.audio_sampling_rate
        self.set_block_state(state, block)
        return components, state

    def scheduler_setup(self, num_inference_steps=None, device=None, sigmas=None):
        destination = torch.device(device or 'cpu')
        if sigmas is None:
            if num_inference_steps is None or num_inference_steps < 2:
                raise ValueError('set_timesteps requires at least two grid points')
            count = int(num_inference_steps)
            base = torch.empty(count, device='cuda', dtype=torch.float32)
            _kernel(premake_grid, count, wide_step=False)(base, 1.0, 0.0)
            numerator = pointwise(base, 2, self._shift)
            denominator = pointwise(base, 2, self._shift - 1)
            denominator = pointwise(denominator, 1, 1.0)
            shifted = pointwise(numerator, 4, denominator).cpu()
            # Collapse adjacent duplicate schedule entries as host control flow.
            values = shifted.tolist()
            values = [value for index, value in enumerate(values)
                      if index == 0 or value != values[index - 1]]
        else:
            values = torch.as_tensor(sigmas, dtype=torch.float32).flatten().tolist()
            if len(values) < 2 or values[-1] != 0.0 or any(
                    first <= second for first, second in zip(values, values[1:])):
                raise ValueError('sigmas must strictly decrease and end at zero')
        accelerator_sigmas = torch.tensor(values, device='cuda', dtype=torch.float32)
        timesteps = pointwise(accelerator_sigmas[:-1], 5, 1.0)
        self.sigmas = accelerator_sigmas.to(destination)
        self.timesteps = timesteps.to(destination)
        self.num_inference_steps = len(values) - 1
        self._step_index = None
        self._begin_index = None

    def scale_noise(self, sample, timestep, noise):
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, device=sample.device, dtype=sample.dtype)
        else:
            timestep = pointwise(timestep.to(sample.device), 0, dtype=sample.dtype)
        while timestep.ndim < sample.ndim:
            timestep = timestep.unsqueeze(-1)
        left = pointwise(timestep, 2, sample)
        residual = pointwise(timestep, 5, 1.0)
        right = pointwise(residual, 2, noise)
        return pointwise(left, 1, right)

    def scheduler_step(self, model_output, timestep, sample, return_dict=True):
        if self.num_inference_steps is None:
            raise ValueError("Call set_timesteps before step")
        if isinstance(timestep, int) or (isinstance(timestep, torch.Tensor)
                                        and not timestep.is_floating_point()):
            raise ValueError("Use an actual scheduler timestep, not an integer index")
        if self.step_index is None:
            self._step_index = (self.index_for_timestep(timestep)
                                if self._begin_index is None else self._begin_index)
        if not isinstance(timestep, torch.Tensor):
            timestep = torch.tensor(timestep, dtype=sample.dtype)
        denoised = arithmetic.scheduler_denoise(sample, model_output, timestep)
        result = arithmetic.scheduler_interpolate(
            sample, denoised, self.sigmas[self.step_index], self.sigmas[self.step_index + 1])
        self._step_index += 1
        return MiniMaxH3SchedulerOutput(prev_sample=result) if return_dict else (result,)

    replace(encoders, "encode_vae_condition", encode_condition)
    replace(decoders.MiniMaxH3VideoDecodeStep, "__call__", video_decode)
    replace(decoders.MiniMaxH3AudioDecodeStep, "__call__", audio_decode)
    replace(MiniMaxH3Scheduler, "step", scheduler_step)
    replace(MiniMaxH3Scheduler, "set_timesteps", scheduler_setup)
    replace(MiniMaxH3Scheduler, "scale_noise", scale_noise)
    replace(before_denoise, "randn_tensor", randn_tensor)
    replace(before_denoise.MiniMaxH3PrepareLayoutStep, "build_packed_sequence",
            build_packed_sequence, static=True)
    replace(video_vae.MiniMaxH3VideoRotaryPosEmbed, "__init__", rotary_init)
    replace(audio_vae, "kaiser_sinc_filter1d", kaiser_sinc_filter1d)
    save()
    return replacements
