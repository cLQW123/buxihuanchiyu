"""检查完整构建与连续两例验证是否真正完成。"""
import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/offline-validation')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    receipt = json.loads((output/'validation.json').read_text())
    if not receipt['technical_validation_pass'] or not receipt['reference_code_unchanged']:
        raise RuntimeError('完整验证未通过，或参考代码来源不符')
    if receipt['reference_kind'] not in ('official-unmodified-execution', 'organizer-standard'):
        raise RuntimeError('参考必须来自原样官方执行或赛方标准输出')
    names = ['environment', 'build', 'runtime-policy']
    stages = receipt['stages']
    if any(item['stage'] == 'reference' for item in stages):
        names.append('reference')
    names += ['continuous', 'comparison', 'media', 'media-seed43']
    if [item['stage'] for item in stages] != names or any(item['exit'] for item in stages):
        raise RuntimeError('验证阶段缺失或非零退出')
    cases = receipt['cases']
    if [item['seed'] for item in cases] != [42, 43] or any(item['exit'] for item in cases):
        raise RuntimeError('两例连续生成未全部完成')
    for name in ('media.json', 'media-seed43.json'):
        media = json.loads((output/name).read_text())
        if not media['pass'] or media['actual'] != media['reference']:
            raise RuntimeError('媒体检查失败: ' + name)
    print('完整复现完成，平均SSIM百分比:', receipt['ssim_percent'], flush=True)


if __name__ == '__main__':
    main()
