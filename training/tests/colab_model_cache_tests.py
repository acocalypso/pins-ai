"""Offline tiny-file checks for the optional Colab Drive base-model cache."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_model_cache import persist_base_model, reuse_base_model


class BaseModelCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.local = self.root / 'local'
        self.cache_root = self.root / 'drive' / 'base-models'
        self.local.mkdir()
        self.config = self.root / 'config.json'
        self.identity = {'model': 'test/model', 'revision': 'pinned-revision',
                         'license': 'Apache-2.0',
                         'weightsSha256': hashlib.sha256(b'weights').hexdigest()}
        self.config.write_text(json.dumps(self.identity), encoding='utf-8')
        files = {'model.safetensors': b'weights', 'config.json': b'{}',
                 'tokenizer.json': b'{}', 'tokenizer_config.json': b'{}'}
        for name, data in files.items():
            (self.local / name).write_bytes(data)
        self.provenance = {key: value for key, value in self.identity.items() if key != 'weightsSha256'}
        self.provenance.update(tokenizerOnly=False, files={
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
        self.write_provenance(self.local)

    def write_provenance(self, folder):
        (folder / 'base-provenance.json').write_text(json.dumps(self.provenance), encoding='utf-8')

    def persist(self):
        return persist_base_model(self.local, self.cache_root, self.config)

    def reuse(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return reuse_base_model(self.local, self.cache_root, self.config)

    def test_verified_local_needs_no_drive_or_download(self):
        self.assertEqual(self.reuse()['source'], 'local')
        self.assertFalse(self.cache_root.exists())

    def test_store_and_restore_exact_files_without_hf_metadata(self):
        (self.local / '.cache' / 'huggingface').mkdir(parents=True)
        (self.local / '.cache' / 'huggingface' / 'token').write_text('do not copy')
        cache = Path(self.persist()['cachePath'])
        self.assertFalse((cache / '.cache').exists())
        self.assertEqual(self.persist()['source'], 'existing')
        shutil.rmtree(self.local)
        self.assertEqual(self.reuse()['source'], 'drive')
        for path in cache.iterdir():
            self.assertEqual(path.read_bytes(), (self.local / path.name).read_bytes())

    def test_corrupt_local_recovers_from_verified_drive(self):
        self.persist()
        (self.local / 'model.safetensors').write_bytes(b'corrupt')
        self.assertEqual(self.reuse()['source'], 'drive')
        self.assertEqual((self.local / 'model.safetensors').read_bytes(), b'weights')

    def test_corrupt_drive_is_miss_and_does_not_overwrite_local(self):
        cache = Path(self.persist()['cachePath'])
        (cache / 'tokenizer.json').write_bytes(b'corrupt')
        (self.local / 'model.safetensors').write_bytes(b'local-invalid')
        self.assertIsNone(self.reuse())
        self.assertEqual((self.local / 'model.safetensors').read_bytes(), b'local-invalid')

    def test_drive_cannot_supply_its_own_weights_hash(self):
        cache = Path(self.persist()['cachePath'])
        (cache / 'model.safetensors').write_bytes(b'changed')
        self.provenance['files']['model.safetensors'] = hashlib.sha256(b'changed').hexdigest()
        self.write_provenance(cache)
        shutil.rmtree(self.local)
        self.assertIsNone(self.reuse())
        self.assertFalse(self.local.exists())

    def test_wrong_revision_partial_and_tokenizer_only_are_rejected(self):
        for key, value in [('revision', 'other'), ('tokenizerOnly', True), ('files', {})]:
            with self.subTest(key=key):
                original = self.provenance[key]
                self.provenance[key] = value
                self.write_provenance(self.local)
                self.assertIsNone(self.reuse())
                with self.assertRaises(ValueError):
                    self.persist()
                self.provenance[key] = original

    def test_code_and_path_traversal_are_rejected_without_execution(self):
        for name in ('evil.py', '../escape', 'nested/tokenizer.json', 'C:\\escape'):
            with self.subTest(name=name):
                self.provenance['files'][name] = '0' * 64
                self.write_provenance(self.local)
                self.assertIsNone(self.reuse())
                del self.provenance['files'][name]
        self.write_provenance(self.local)
        (self.local / 'evil.py').write_text('raise AssertionError("never execute")')
        self.assertIsNone(self.reuse())

    def test_changed_git_identity_does_not_use_old_cache(self):
        self.persist()
        shutil.rmtree(self.local)
        self.identity['revision'] = 'new-revision'
        self.config.write_text(json.dumps(self.identity), encoding='utf-8')
        self.assertIsNone(self.reuse())

    def test_invalid_cache_can_be_replaced_after_explicit_valid_fetch(self):
        cache = Path(self.persist()['cachePath'])
        (cache / 'model.safetensors').write_bytes(b'interrupted')
        self.assertEqual(self.persist()['source'], 'stored')
        shutil.rmtree(self.local)
        self.assertEqual(self.reuse()['source'], 'drive')

    def test_invalid_json_provenance_is_a_cache_miss(self):
        for value in ('[]', 'null', 'not json'):
            (self.local / 'base-provenance.json').write_text(value, encoding='utf-8')
            self.assertIsNone(self.reuse())

    def test_corrupted_copy_never_replaces_existing_local_files(self):
        self.persist()
        (self.local / 'model.safetensors').write_bytes(b'preserve-until-verified')
        copyfile = shutil.copyfile

        def interrupted_copy(source, destination):
            result = copyfile(source, destination)
            if Path(destination).name == 'model.safetensors':
                Path(destination).write_bytes(b'corrupted-in-transit')
            return result

        with patch('colab_model_cache.shutil.copyfile', side_effect=interrupted_copy):
            with self.assertRaisesRegex(ValueError, 'verification'):
                self.reuse()
        self.assertEqual((self.local / 'model.safetensors').read_bytes(), b'preserve-until-verified')

    def test_symlinked_model_file_is_rejected(self):
        original = self.root / 'weights'
        (self.local / 'model.safetensors').rename(original)
        try:
            (self.local / 'model.safetensors').symlink_to(original)
        except OSError:
            self.skipTest('Creating symlinks requires unavailable OS privileges')
        self.assertIsNone(self.reuse())


if __name__ == '__main__':
    unittest.main()
