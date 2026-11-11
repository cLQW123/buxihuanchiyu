"""Check decoded FL2VA media against the unchanged reference output."""
import argparse
import hashlib
import json
from pathlib import Path

import av


def inspect(path):
    with av.open(str(path)) as container:
        video = container.streams.video[0]
        metadata = {'width': video.width, 'height': video.height,
                    'fps': str(video.average_rate),
                    'codec': video.codec_context.name,
                    'pixel_format': video.codec_context.format.name,
                    'audio': [{'codec': stream.codec_context.name,
                               'sample_rate': stream.codec_context.sample_rate,
                               'channels': stream.codec_context.channels}
                              for stream in container.streams.audio]}
        metadata['frames'] = sum(1 for _ in container.decode(video))
    assert metadata['frames'] == 124, metadata
    assert metadata['audio'], 'The output must contain audio'
    with av.open(str(path)) as container:
        audio = container.streams.audio[0]
        metadata['decoded_audio_samples'] = sum(frame.samples for frame in container.decode(audio))
    assert metadata['decoded_audio_samples'] > 0, metadata
    return metadata


parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--video', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
reference, actual = inspect(args.reference), inspect(args.video)
assert actual == reference, {'reference': reference, 'actual': actual}
receipt = {'reference': reference, 'actual': actual, 'pass': True,
           'video_bytes': args.video.stat().st_size,
           'video_sha256': hashlib.sha256(args.video.read_bytes()).hexdigest()}
args.output.write_text(json.dumps(receipt, indent=2))
print('DECODED_MEDIA_SPECIFICATION_PASSED', json.dumps(actual), flush=True)
