"""Run the unchanged official FL2VA entry with explicit NineToothed adapters."""
import ctypes
import functools
import faulthandler
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
from check_inputs import default_frame_arguments, first_frame_record

from validate_runtime_policy import collect_runtime_policy

faulthandler.enable()
sys.argv[1:]=default_frame_arguments(sys.argv[1:])
if os.environ.get('SGCUP_TRACE_TIMEOUT'):
    faulthandler.dump_traceback_later(int(os.environ['SGCUP_TRACE_TIMEOUT']), repeat=True)
import torch

root=Path(os.environ['SGCUP_ROOT']).resolve()
prefix=Path(os.environ['INFINI_ROOT']).resolve()
runtime_policy=collect_runtime_policy(root,prefix)
evidence=Path(os.environ.get('SGCUP_RUN_EVIDENCE',root/'outputs/latest'))
evidence.mkdir(parents=True,exist_ok=True)
if torch.cuda.device_count()!=8:
    raise RuntimeError('The submitted FL2VA path requires an exclusive eight-card allocation')
for name in ('libinfinirt.so','libinfiniop.so','libinfinicore_cpp_api.so','mha_ninetoothed.so'):
    ctypes.CDLL(str(prefix/'lib'/name),mode=ctypes.RTLD_GLOBAL)
from diffusers import AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio
from infinilm.multimodal.ninetoothed_vae import install_ninetoothed_vae
from infinilm.multimodal.ninetoothed_pipeline import install_pipeline_arithmetic

if os.environ.get('SGCUP_KERNEL_TRACE') == '1':
    from infinilm.multimodal.ninetoothed_trace import install_kernel_trace
    install_kernel_trace(evidence)
if os.environ.get('SGCUP_DIAGNOSTIC_LATENTS') == '1':
    from infinilm.multimodal.ninetoothed_trace import capture_audio_diagnostic_input
    capture_audio_diagnostic_input(AutoencoderKLMiniMaxH3Audio, evidence)

def precision_scope(original):
    @functools.wraps(original)
    def run(*args,**kwargs):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return original(*args,**kwargs)
    return run
for cls in (AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio):
    for name in ('encode','decode'):
        setattr(cls,name,precision_scope(getattr(cls,name)))
install_ninetoothed_vae(AutoencoderKLMiniMaxH3,AutoencoderKLMiniMaxH3Audio,evidence/'vae-ops.json')
install_pipeline_arithmetic(evidence/'boundary-calls.json')

# Host synchronization is part of the submitted path and its measured runtime.
from infinilm.infer_engine import InferEngine
original_forward=InferEngine.forward_raw
@functools.wraps(original_forward)
def fenced_forward(self,*args,**kwargs):
    for device in range(8):torch.cuda.synchronize(device)
    output=original_forward(self,*args,**kwargs)
    for device in range(8):torch.cuda.synchronize(device)
    return output
InferEngine.forward_raw=fenced_forward

entry=root/'Projects/InfiniLM/examples/minimax_h3.py'
expected='ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4'
assert hashlib.sha256(entry.read_bytes()).hexdigest()==expected,'The official entry was changed'
import _infinilm
actual=Path(_infinilm.__file__).resolve()
if actual!=(root/'build/python/_infinilm.so').resolve():
    raise RuntimeError('The native module must come from this source build: '+str(actual))
from infinicore.lib import _infinicore
core_module=Path(_infinicore.__file__).resolve()
if core_module!=(root/'Projects/InfiniCore/python/infinicore/lib/_infinicore.so').resolve():
    raise RuntimeError('The core module must come from this source build: '+str(core_module))
manifest={'argv':sys.argv[1:],'entry_sha256':expected,'native_sha256':hashlib.sha256(actual.read_bytes()).hexdigest(),
          'core_sha256':hashlib.sha256(core_module.read_bytes()).hexdigest(),
          'devices':8,'fallback':'unsupported neural numerical operators raise',
          'runtime_policy':runtime_policy}
if sys.argv[1:2]==['fl2va']:
    frame_arg=next((arg.split('=',1)[1] for arg in sys.argv if arg.startswith('--first-frame=')),None)
    if frame_arg is None:
        frame_arg=sys.argv[sys.argv.index('--first-frame')+1]
    manifest['first_frame']=first_frame_record(frame_arg)
manifest['source_modules'] = {}
for name, project in (('ninetoothed', 'ninetoothed'), ('ntops', 'ntops'),
                      ('infinilm', 'InfiniLM'), ('infiniop.ops.vae_support.ninetoothed.support', 'InfiniCore')):
    module_path = Path(importlib.import_module(name).__file__).resolve()
    if not module_path.is_relative_to((root / 'Projects' / project).resolve()):
        raise RuntimeError('Source module is outside this build: ' + str(module_path))
    manifest['source_modules'][name] = str(module_path)
(evidence/'run.json').write_text(json.dumps(manifest,indent=2))
print('SUBMITTED_SOURCE_RUNTIME',json.dumps(manifest),flush=True)
sys.argv[0]=str(entry)
try:
    runpy.run_path(str(entry),run_name='__main__')
finally:
    for device in range(8):torch.cuda.synchronize(device)
