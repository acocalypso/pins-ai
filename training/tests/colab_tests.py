"""Offline checks for Drive data provenance and Git/Drive separation."""
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
import zipfile
import json
import hashlib
import ast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training'))
from colab_workspace import use_drive_data


class DriveDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.project = Path(self.temp.name) / 'project'
        self.drive = Path(self.temp.name) / 'drive'
        self.project.mkdir()
        self.drive.mkdir()
        self.data = self.project / 'training/datasets/mcp-v2/data'
        self.data.mkdir(parents=True)
        for name in ('manifest.json', 'contract-audit.json', 'train.jsonl'):
            (self.data / name).write_text('{}\n', encoding='utf-8')
        (self.data / 'manifest.json').write_text(json.dumps({'sha256': {
            'train.jsonl': hashlib.sha256((self.data / 'train.jsonl').read_bytes()).hexdigest()}}), encoding='utf-8')
        subprocess.run(['git', 'init', '--quiet', str(self.project)], check=True)
        subprocess.run(['git', 'add', '.'], cwd=self.project, check=True, capture_output=True)

    def make_bundle(self, mismatch=False, legacy=False):
        with zipfile.ZipFile(self.drive / 'pins-ai-training-0.1.0.zip', 'w') as archive:
            for path in self.data.iterdir():
                content = b'changed' if mismatch and path.name == 'train.jsonl' else path.read_bytes()
                name=path.relative_to(self.project).as_posix()
                if legacy: name=name.replace('training/datasets/mcp-v2','PINS_MCP_AI_Training_v2')
                archive.writestr(name, content)
            archive.writestr('training/train.py', 'untrusted code must not be imported')

    def test_zip_uses_only_matching_data(self):
        self.make_bundle()
        result = use_drive_data(self.project, self.drive)
        self.assertEqual(result['verifiedFiles'], 2)
        self.assertEqual(result['kind'], 'zip')
        self.assertFalse((self.project / 'training/train.py').exists())

    def test_existing_drive_zip_with_legacy_layout_is_supported(self):
        self.make_bundle(legacy=True)
        self.assertEqual(use_drive_data(self.project,self.drive)['verifiedFiles'],2)

    def test_mismatch_preserves_checkout(self):
        self.make_bundle(mismatch=True)
        before = {p.name: p.read_bytes() for p in self.data.iterdir()}
        with self.assertRaisesRegex(ValueError, 'Drive data differs'):
            use_drive_data(self.project, self.drive)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.data.iterdir()})

    def test_directory_and_missing_data(self):
        with self.assertRaises(FileNotFoundError):
            use_drive_data(self.project, self.drive)
        folder = self.drive / 'data'
        folder.mkdir()
        for path in self.data.iterdir():
            (folder / path.name).write_bytes(path.read_bytes())
        self.assertEqual(use_drive_data(self.project, self.drive)['kind'], 'directory')


class NotebookTests(unittest.TestCase):
    def test_checked_in_notebook_cells_compile_and_outputs_are_cleared(self):
        root = Path(__file__).resolve().parents[2]
        notebook = json.loads((root/'training/notebooks/pins-ai-colab.ipynb').read_text(encoding='utf-8'))
        settings = {}
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code': continue
            source = ''.join(cell['source'])
            tree = ast.parse(source, filename=f'colab-cell-{index}')
            compile(tree, f'colab-cell-{index}', 'exec')
            self.assertIsNone(cell['execution_count'])
            self.assertEqual(cell['outputs'], [])
            for node in tree.body:
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                    for target in node.targets:
                        if isinstance(target, ast.Name): settings[target.id] = node.value.value
        config = json.loads((root/settings['TRAIN_CONFIG']).read_text())
        self.assertIn('generationValidation', config)
        self.assertFalse(settings['RUN_TRAINING'])


if __name__ == '__main__':
    unittest.main()
