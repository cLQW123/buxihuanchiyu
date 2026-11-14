"""默认样例、显式输入和损坏材料的行为检查；只使用标准库。"""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT/'scripts'))
import check_inputs
import fetch_dependencies
import package_offline


class InputMaterials(unittest.TestCase):
    def test_portable_package_excludes_mac_metadata_and_keeps_required_materials(self):
        import tarfile
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'source'
            root.mkdir()
            for name in ('.git', 'inputs/wheelhouse', 'inputs/vendor-triton', 'inputs/vendor-xmake', 'inputs/archives', 'assets'):
                (root/name).mkdir(parents=True, exist_ok=True)
            for name in ('SOURCE_PROVENANCE.json', '.git/HEAD', 'inputs/dependency-receipt.json',
                         'assets/competition-first-frame.png', 'inputs/wheelhouse/example.whl',
                         'inputs/vendor-xmake/tool.lua', 'inputs/vendor-xmake/._tool.lua',
                         '.DS_Store', 'inputs/archives/cache.tar'):
                (root/name).write_bytes(b'test')
            output = package_offline.package_checkout(root, Path(directory)/'portable.tar')
            with tarfile.open(output) as archive:
                names = archive.getnames()
                self.assertIn('buxihuanchiyu/.git/HEAD', names)
                self.assertIn('buxihuanchiyu/assets/competition-first-frame.png', names)
                self.assertIn('buxihuanchiyu/inputs/vendor-xmake/tool.lua', names)
                self.assertIn('buxihuanchiyu/inputs/wheelhouse/example.whl', names)
                self.assertNotIn('buxihuanchiyu/inputs/vendor-xmake/._tool.lua', names)
                self.assertNotIn('buxihuanchiyu/.DS_Store', names)
                self.assertNotIn('buxihuanchiyu/inputs/archives/cache.tar', names)
            (root/'build').mkdir()
            with self.assertRaises(RuntimeError):
                package_offline.package_checkout(root, Path(directory)/'stale.tar')

    def test_fresh_clone_needs_no_report(self):
        with patch.dict(os.environ, {}, clear=True):
            record = check_inputs.first_frame_record()
            self.assertTrue(record['bundled_competition_input'])
            self.assertFalse(record['requires_report_or_template'])
            self.assertEqual(record['sha256'], 'b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54')
            args = check_inputs.default_frame_arguments(['fl2va', '--seed', '42'])
            self.assertEqual(args[-2:], ['--first-frame', record['path']])

    def test_explicit_image_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory)/'other.png'
            other.write_bytes(check_inputs.bundled_first_frame().read_bytes())
            with patch.dict(os.environ, {'FIRST_FRAME': str(other)}):
                self.assertEqual(check_inputs.resolve_first_frame(), other.resolve())
                explicit = ['fl2va', '--first-frame=another.png']
                self.assertEqual(check_inputs.default_frame_arguments(explicit), explicit)

    def test_changed_bundled_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'assets').mkdir()
            (root/'assets/competition-inputs.json').write_bytes((ROOT/'assets/competition-inputs.json').read_bytes())
            (root/'assets/competition-first-frame.png').write_bytes(b'changed')
            with patch.object(check_inputs, 'ROOT', root), self.assertRaises(RuntimeError):
                check_inputs.resolve_first_frame()

    def test_model_index_missing_shard_and_lfs_pointer_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory)
            (model/'model_index.json').write_text('{}')
            for component in ('text_encoder', 'transformer', 'vae', 'audio_vae', 'scheduler', 'audio_scheduler'):
                folder = model/component
                folder.mkdir()
                name = 'scheduler_config.json' if 'scheduler' in component else 'config.json'
                (folder/name).write_text('{}')
                if 'scheduler' not in component:
                    (folder/'weights.safetensors').write_bytes(b'x'*2048)
            self.assertEqual(check_inputs.model_record(model)['weight_files'], 4)
            (model/'transformer/weights.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'missing.safetensors'}}))
            with self.assertRaises(RuntimeError):
                check_inputs.model_record(model)
            (model/'transformer/missing.safetensors').write_text('version https://git-lfs.github.com/spec/v1')
            with self.assertRaises(RuntimeError):
                check_inputs.model_record(model)

    def test_download_falls_back_after_direct_failure_and_checks_hash(self):
        import io
        from urllib.error import HTTPError
        payload = b'fixed public dependency'
        item = {'file': 'asset.tar', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}
        manifest = {'repository': 'test/repo', 'release_tag': 'fixed'}
        metadata = {'draft': False, 'assets': [{'name': item['file'], 'size': len(payload), 'url': 'https://api.github.com/assets/1'}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'partial'
            responses = [HTTPError('url', 403, 'denied', None, None), io.BytesIO(json.dumps(metadata).encode()), io.BytesIO(payload)]
            with patch('urllib.request.urlopen', side_effect=responses):
                fetch_dependencies.download_asset(item, manifest, path, 'auto')
            self.assertEqual(path.read_bytes(), payload)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'changed')):
                with self.assertRaises(RuntimeError):
                    fetch_dependencies.download_asset(item, manifest, path, 'direct')
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
