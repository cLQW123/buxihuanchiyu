"""Optional host diagnostics for NineToothed compilation and launches."""
import functools
import json
import os
from pathlib import Path
import resource
import sys
import time

import torch


def install_kernel_trace(directory):
    from . import ninetoothed_vae_encoder as encoder

    directory = Path(directory)
    original = encoder._kernel
    seen = set()
    constructed = set()

    def record(item):
        item['max_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        item['time'] = time.time()
        temporary = directory / 'last-kernel.json.tmp'
        temporary.write_text(json.dumps(item))
        os.replace(temporary, directory / 'last-kernel.json')

    def traced(premake, *args, **kwargs):
        kernel = {'module': premake.__module__, 'name': premake.__name__,
                  'arguments': repr(args), 'keywords': repr(kwargs)}
        key = json.dumps(kernel, sort_keys=True)
        if key not in constructed:
            constructed.add(key)
            record({'phase': 'construct', 'kernel': kernel})
        function = original(premake, *args, **kwargs)

        @functools.wraps(function)
        def launch(*arguments):
            tensors = [{'shape': list(value.shape), 'stride': list(value.stride()),
                        'dtype': str(value.dtype), 'device': str(value.device)}
                       if isinstance(value, torch.Tensor) else repr(value)
                       for value in arguments]
            signature = json.dumps([kernel, tensors], sort_keys=True)
            item = {'phase': 'launch', 'kernel': kernel, 'tensors': tensors}
            first = signature not in seen
            if first:
                seen.add(signature)
                record(item)
                print('NINETOOTHED_KERNEL_TRACE', signature, flush=True)
            result = function(*arguments)
            if first:
                record({**item, 'phase': 'returned'})
            return result

        return launch

    for name, module in tuple(sys.modules.items()):
        if name.startswith(__package__ + '.ninetoothed_') and getattr(module, '_kernel', None) is original:
            module._kernel = traced


def capture_audio_diagnostic_input(audio_class, directory):
    original = audio_class.decode

    @functools.wraps(original)
    def decode(self, latent, *args, **kwargs):
        # This diagnostic tensor is never consumed by the submitted generator.
        torch.save(latent.detach().cpu(), Path(directory) / 'diagnostic-audio-input.pt')
        return original(self, latent, *args, **kwargs)

    audio_class.decode = decode
