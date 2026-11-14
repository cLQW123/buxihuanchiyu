"""校验随仓库提供的首帧，以及显式提供的比赛模型目录。"""
import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def bundled_first_frame():
    record = json.loads((ROOT / 'assets/competition-inputs.json').read_text())['first_frame']
    path = (ROOT / record['path']).resolve(strict=True)
    if not path.is_relative_to(ROOT / 'assets'):
        raise RuntimeError('首帧清单路径必须位于本仓库assets目录')
    if path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
        raise RuntimeError('仓库首帧文件缺失或已改变，请重新拉取固定公开版本')
    return path


def resolve_first_frame(value=None):
    default = bundled_first_frame()
    selected = value or os.environ.get('FIRST_FRAME') or default
    path = Path(selected).expanduser().resolve(strict=True)
    if not path.is_file():
        raise RuntimeError('首帧必须是可读图片文件: ' + str(path))
    return path


def first_frame_record(value=None):
    path = resolve_first_frame(value)
    return {'path': str(path), 'sha256': digest(path),
            'bytes': path.stat().st_size, 'bundled_competition_input': path == bundled_first_frame(),
            'requires_report_or_template': False}


def default_frame_arguments(arguments):
    arguments = list(arguments)
    if arguments and arguments[0] == 'fl2va':
        supplied = any(arg == '--first-frame' or arg.startswith('--first-frame=') for arg in arguments)
        if not supplied:
            arguments += ['--first-frame', str(resolve_first_frame())]
    return arguments


def model_record(value):
    model = Path(value).expanduser().resolve(strict=True)
    for name in ('model_index.json', 'text_encoder/config.json', 'transformer/config.json',
                 'vae/config.json', 'audio_vae/config.json',
                 'scheduler/scheduler_config.json', 'audio_scheduler/scheduler_config.json'):
        if not (model / name).is_file():
            raise RuntimeError('缺少配套Diffusers模型文件: ' + str(model / name))
    weights = set()
    for component in ('text_encoder', 'transformer', 'vae', 'audio_vae'):
        folder = model / component
        indices = sorted(folder.glob('*.safetensors.index.json'))
        if indices:
            for index in indices:
                names = set(json.loads(index.read_text())['weight_map'].values())
                for name in names:
                    path = (folder / name).resolve()
                    if not path.is_relative_to(folder.resolve()):
                        raise RuntimeError('模型索引包含目录外权重: ' + name)
                    weights.add(path)
        else:
            weights.update(folder.glob('*.safetensors'))
        if not any(path.is_relative_to(folder.resolve()) for path in weights):
            raise RuntimeError('缺少模型权重: ' + str(folder))
    for path in weights:
        if not path.is_file() or path.stat().st_size < 1024:
            raise RuntimeError('模型权重缺失或仍为未下载的LFS指针: ' + str(path))
    return {'path': str(model), 'weight_files': len(weights),
            'weight_bytes': sum(path.stat().st_size for path in weights),
            'scope': '仅检查目录、索引和文件存在性；不改写模型，也不替代官方权重哈希核验',
            'source': '赛事统一提供的MiniMax-H3 Base FL2VA Diffusers checkpoint'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-frame', type=Path)
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    record = {'pass': True, 'first_frame': first_frame_record(args.first_frame)}
    if args.model_path:
        record['model'] = model_record(args.model_path)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print('COMPETITION_INPUTS_VERIFIED', json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
