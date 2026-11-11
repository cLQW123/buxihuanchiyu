"""Install strict NineToothed VAE execution without changing evaluator code."""
from collections import Counter
import functools
import json
from pathlib import Path
import time

from .ninetoothed_vae_encoder import NineToothedVAEEncoderMode
from .ninetoothed_vae_video import NineToothedVAEVideoMode
from .ninetoothed_vae_audio import NineToothedVAEAudioMode


def install_ninetoothed_vae(video_class, audio_class, evidence_path):
    """Wrap neural methods; fail on unsupported compute and retain audit data.

    Caller controls dtype/device placement and performs normal model loading.
    Media output and scheduler numerical coverage are audited separately.
    """
    evidence = Path(evidence_path)
    evidence.parent.mkdir(parents=True, exist_ok=True)
    counters = Counter()
    stages = []
    methods = []
    for model_class, name, mode_class in (
        (video_class, "encode", NineToothedVAEEncoderMode),
        (video_class, "decode", NineToothedVAEVideoMode),
        (audio_class, "decode", NineToothedVAEAudioMode),
    ):
        original = getattr(model_class, name)
        label = model_class.__name__ + "." + name

        def wrap(original, label, mode_class):
            @functools.wraps(original)
            def execute(*args, **kwargs):
                mode = mode_class()
                started = time.perf_counter()
                item = {"stage": label, "status": "started"}
                try:
                    with mode:
                        result = original(*args, **kwargs)
                    item["status"] = "completed"
                    return result
                except BaseException as error:
                    item["status"] = "failed"
                    item["error"] = repr(error)
                    raise
                finally:
                    item["host_seconds"] = time.perf_counter() - started
                    item["op_counts"] = dict(mode.operations)
                    counters.update(mode.operations)
                    stages.append(item)
                    evidence.write_text(json.dumps({
                        "scope": "neural VAE methods; scheduler and other boundaries audited separately",
                        "fallback": "unsupported numerical operations raise",
                        "stages": stages, "op_counts": dict(counters),
                    }, indent=2))
                    print("NINETOOTHED_VAE_STAGE", json.dumps(item), flush=True)
            return execute

        setattr(model_class, name, wrap(original, label, mode_class))
        methods.append((model_class, name, original))
    return methods
