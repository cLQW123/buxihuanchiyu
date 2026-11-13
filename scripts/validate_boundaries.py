"""Compare submitted auxiliary arithmetic with independent upstream functions."""
import json
import os
from pathlib import Path

import torch
from diffusers.modular_pipelines.minimax_h3 import before_denoise
from diffusers.models.autoencoders import autoencoder_kl_minimax_h3 as video_vae
from diffusers.models.autoencoders import autoencoder_kl_minimax_h3_audio as audio_vae
from diffusers.schedulers.scheduling_minimax_h3 import MiniMaxH3Scheduler
from diffusers.utils.torch_utils import randn_tensor as reference_randn
from infinilm.multimodal.ninetoothed_pipeline import install_pipeline_arithmetic
from infinilm.multimodal.ninetoothed_random import randn_tensor
from infinilm.multimodal.ninetoothed_initializers import rotary_init, kaiser_sinc_filter1d
from infinilm.multimodal.ninetoothed_positions import build_packed_sequence
from infinilm.multimodal import ninetoothed_pipeline_arithmetic as arithmetic
from infinilm.multimodal.ninetoothed_vae_encoder import NineToothedVAEEncoderMode

torch.cuda.set_device(7)
device=torch.device('cuda:7')
output=Path(os.environ.get('BOUNDARY_AUDIT_OUTPUT',Path(os.environ['SGCUP_ROOT'])/'build/boundary-validation'))
output.mkdir(parents=True,exist_ok=True)
records=[]

def check(label,actual,expected,*,atol=2e-6,rtol=2e-5):
    actual=actual.detach().cpu()
    expected=expected.detach().cpu()
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
    error=(actual.double()-expected.double()).abs().max().item() if actual.numel() else 0
    record={'name':label,'shape':list(actual.shape),'dtype':str(actual.dtype),
            'max_abs_error':error,'atol':atol,'rtol':rtol,'pass':True}
    records.append(record)
    (output/'cases.json').write_text(json.dumps(records,indent=2)+'\n')
    print('BOUNDARY_REFERENCE_CASE',json.dumps(record),flush=True)

with torch.inference_mode():
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        for shape in ((17,),(2,31),(1,24,31,58,48)):
            reference_generator=torch.Generator().manual_seed(42)
            actual_generator=torch.Generator().manual_seed(42)
            expected=reference_randn(shape,generator=reference_generator,device=device,dtype=dtype)
            actual=randn_tensor(shape,generator=actual_generator,device=device,dtype=dtype)
            tolerance=4e-6 if dtype==torch.float32 else (0.002 if dtype==torch.float16 else 0.016)
            check('rng-'+str(dtype)+'-'+str(shape),actual,expected,atol=tolerance,rtol=tolerance)
            check('rng-state-'+str(dtype)+'-'+str(shape),actual_generator.get_state(),reference_generator.get_state(),atol=0,rtol=0)
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        for count in (1,3,5,31,58,768):
            for start in (0,.5):
                expected=torch.arange(start,start+count,device=device,dtype=dtype)
                with NineToothedVAEEncoderMode():
                    actual=torch.arange(start,start+count,device=device,dtype=dtype)
                check(f'floating-vae-range-{dtype}-{count}-{start}',actual,expected,atol=0,rtol=0)
    for dim in (48,96):
        expected=video_vae.MiniMaxH3VideoRotaryPosEmbed(dim,theta=100.0).inv_freq
        candidate=torch.nn.Module()
        rotary_init(candidate,dim,theta=100.0)
        check('video-rotary-initializer-'+str(dim),candidate.inv_freq,expected,atol=2e-7,rtol=2e-6)
    # Released H3 uses 12 taps; 24/48 provide additional coverage. The fixed model
    # uses the default 12-tap anti-aliasing filters.
    for taps in (12,24,48):
        cutoff=0.25
        expected=audio_vae.kaiser_sinc_filter1d(cutoff,0.3,taps)
        actual=kaiser_sinc_filter1d(cutoff,0.3,taps)
        check('audio-filter-initializer-'+str(taps),actual,expected,atol=4e-6,rtol=2e-5)
    for frames,height,width,audio in ((7,8,12,32),(31,58,48,207),(49,58,48,300)):
        kwargs=dict(text_token_tags=torch.tensor([1,1,0,0,1]),num_latent_frames=frames,
                    latent_height=height,latent_width=width,num_audio_latents=audio,
                    patch_size=(1,2,2),audio_channels=2,audio_tag=2,video_tag=0,keyframe_anchors=('first',))
        expected=before_denoise.MiniMaxH3PrepareLayoutStep.build_packed_sequence(**kwargs)
        actual=build_packed_sequence(**kwargs)
        for index,(left,right) in enumerate(zip(actual,expected)):
            if isinstance(left,torch.Tensor):
                check(f'packed-layout-{frames}-{index}',left,right,atol=1e-12,rtol=1e-13)
            else:assert left==right
    mean=torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1,1)
    std=torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1,1)
    pixels=torch.randint(0,256,(1,3,1,928,768),device=device,dtype=torch.uint8)
    expected=(pixels.float()/255.0-mean)/std
    check('pixel-encode-actual-shape',arithmetic.encode_pixels(pixels,mean,std),expected)
    value=torch.randn((1,3,7,19,23),device=device,dtype=torch.bfloat16)
    check('pixel-decode',arithmetic.decode_pixels(value,mean,std),(value.float()*std+mean).clamp(0,1))
    references=[]
    for dtype in (torch.float32,torch.float16,torch.bfloat16):
        scheduler=MiniMaxH3Scheduler()
        references.append((dtype,scheduler,scheduler.set_timesteps,scheduler.step,scheduler.scale_noise))
    install_pipeline_arithmetic(output/'adapter-calls.json')
    for dtype,reference,set_reference,step_reference,noise_reference in references:
        candidate=MiniMaxH3Scheduler()
        set_reference(50,device=device)
        candidate.set_timesteps(50,device=device)
        check('schedule-sigmas-'+str(dtype),candidate.sigmas,reference.sigmas,atol=2e-7,rtol=2e-6)
        check('schedule-timesteps-'+str(dtype),candidate.timesteps,reference.timesteps,atol=2e-7,rtol=2e-6)
        assert len(candidate.timesteps)==49
        sample=torch.randn((2,24,7,9),device=device,dtype=dtype)
        noise=torch.randn_like(sample)
        tolerance=1e-5 if dtype==torch.float32 else (0.006 if dtype==torch.float16 else .05)
        check('condition-noise-'+str(dtype),candidate.scale_noise(sample,.9,noise),noise_reference(sample,.9,noise),atol=tolerance,rtol=tolerance)
        actual=sample.clone();expected=sample.clone()
        for index,t in enumerate(reference.timesteps):
            velocity=torch.randn_like(sample)
            expected=step_reference(velocity,t,expected).prev_sample
            actual=candidate.step(velocity,t,actual).prev_sample
            check(f'scheduler-{dtype}-step{index}',actual,expected,atol=tolerance,rtol=tolerance)
receipt={'pass':True,'case_count':len(records),'scope':'RNG and generator state, initialization buffers, fixed FL2VA position layout, pixel normalization and 49 scheduler updates compared with original upstream functions; VAE neural methods and native model require separate full-run evidence','cases':records}
(output/'validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('BOUNDARY_REFERENCE_VALIDATION_PASSED',len(records),flush=True)
