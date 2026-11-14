"""Run the original official PyTorch entry in a separate, unpatched process."""
import argparse
import base64
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from check_inputs import resolve_first_frame, model_record


ROOT = Path(__file__).resolve().parent.parent
ENTRY_SHA = 'ca54a18a73a7d760d644ca4eb2c3b20678f88dfdbcfa26a62dcd3de617b31af4'
COMPARE_SHA = 'e143c28e4b3acb885662e0aa238aa00ca17cbcc82077f465f2ce57d0f1225590'


def digest(path):
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--first-frame', type=Path, help='默认使用仓库随附的统一首帧')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    entry = ROOT / 'Projects/InfiniLM/examples/minimax_h3.py'
    comparison = entry.with_name('minimax_h3_compare.py')
    assert digest(entry) == ENTRY_SHA, 'Official entry differs from the pinned upstream'
    assert digest(comparison) == COMPARE_SHA, 'Official comparison differs from the pinned upstream'
    dependency = json.loads((ROOT / 'dependencies/diffusers-source.json').read_text())
    dist = metadata.distribution('diffusers')
    assert dist.version == dependency['version']
    verified = 0
    for item in dist.files or []:
        if str(item).startswith('diffusers/') and item.suffix == '.py':
            assert item.hash is not None and item.hash.mode == 'sha256', str(item)
            expected = base64.urlsafe_b64decode(item.hash.value + '=' * (-len(item.hash.value) % 4))
            assert hashlib.file_digest(Path(dist.locate_file(item)).open('rb'), 'sha256').digest() == expected, str(item)
            verified += 1
    assert verified == dependency['python_files_verified_byte_identical_to_official_source']
    model = args.model_path.resolve(strict=True)
    model_record(model)
    first_frame = resolve_first_frame(args.first_frame)
    video = output / 'reference.mp4'
    command = [sys.executable, str(entry), 'fl2va', '--model-path', str(model),
               '--first-frame', str(first_frame), '--text', 'A dog falls asleep',
               '--gpus', '0,1,2,3,4,5,6,7', '--legacy',
               '--legacy-text-gpus', '0,1,2,3', '--legacy-transformer-gpus', '4,5,6,7',
               '--num-frames', '124', '--steps', '50', '--seed', '42', '--output', str(video)]
    record = {
        'kind': 'official-unmodified-execution',
        'description': 'Original official InfiniLM --legacy entry with unmodified pinned upstream Diffusers; no numerical execution adapters.',
        'official_entry_sha256': ENTRY_SHA, 'comparison_sha256': COMPARE_SHA,
        'diffusers': dependency, 'verified_dependency_python_files': verified,
        'first_frame_sha256': digest(first_frame), 'entry_modifications': [],
        'numerical_execution_wrappers': [], 'command': command,
        'reference_is_unmodified_execution': True, 'full_generation_completed': False,
        'organizer_acceptance_established': False,
    }
    receipt = output / 'reference-provenance.json'
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    started = time.time()
    with (output / 'generation.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        (output / 'generation.pid').write_text(str(process.pid) + '\n')
        status = process.wait()
    (output / 'generation.exit').write_text(str(status) + '\n')
    record.update(exit=status, process_wall_seconds=time.time() - started)
    if status == 0 and video.is_file():
        record.update(full_generation_completed=True, video_sha256=digest(video),
                      video_bytes=video.stat().st_size)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    if status or not record['full_generation_completed']:
        raise RuntimeError('Official reference generation failed; inspect ' + str(output / 'generation.log'))
    print('UNMODIFIED_OFFICIAL_REFERENCE_COMPLETED', json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
