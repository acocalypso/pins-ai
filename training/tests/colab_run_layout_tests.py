"""CPU-only original-run/append-only Colab continuation layout contracts."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from colab_run_layout import (create_layout, latest_source, publish_latest,
                              preserve_resume_evidence, archive_attempt, POINTER)


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.drive = Path(self.temporary.name) / 'drive'
        self.root = self.drive / 'runs/original'
        self.full(self.root)

    def full(self, attempt, status='started'):
        (attempt / 'full').mkdir(parents=True)
        (attempt / 'full/training-run.json').write_text(json.dumps({'status': status, 'smokeOnly': False}))

    def test_multiple_attempts_preserve_original_and_select_latest_training(self):
        original = self.root / 'full/training-run.json'
        before = original.read_bytes()
        first = create_layout('resume', self.drive, 'ignored-new-run', self.root)
        self.assertEqual(first['runRoot'], self.root.resolve())
        self.assertEqual(first['source'], self.root.resolve())
        self.assertFalse(publish_latest(self.root, first['attemptRoot']))
        self.assertEqual(latest_source(self.root), self.root.resolve())
        self.full(first['attemptRoot'])
        self.assertTrue(publish_latest(self.root, first['attemptRoot']))
        second = create_layout('resume', self.drive, 'ignored-again', self.root)
        self.assertEqual(second['source'], first['attemptRoot'])
        self.full(second['attemptRoot'])
        # Simulate VM death before the pointer could update.
        self.assertEqual(latest_source(self.root), second['attemptRoot'])
        self.assertEqual(original.read_bytes(), before)
        self.assertEqual([p.name for p in self.root.parent.iterdir()], ['original'])

    def test_completed_latest_never_restarts_older_run(self):
        attempt = self.root / 'attempts/resume-999'
        self.full(attempt, 'completed')
        with self.assertRaisesRegex(ValueError, 'completed'):
            latest_source(self.root)

    def test_pointer_escape_and_unstarted_target_are_rejected(self):
        for value in ('../other', '/tmp/other', 'C:\\escape', 'attempts/resume-missing'):
            with self.subTest(value=value):
                (self.root / POINTER).write_text(json.dumps({'attempt': value}))
                with self.assertRaises(ValueError): latest_source(self.root)

    def test_zip_infers_run_name_and_rejects_identity_conflict(self):
        source = self.drive / 'archived-results.zip'
        with zipfile.ZipFile(source, 'w') as archive:
            archive.writestr('full/training-run.json', '{}')
        layout = create_layout('resume', self.drive, 'irrelevant', source)
        self.assertEqual(layout['runRoot'].name, 'archived')
        self.assertEqual(layout['source'], source.resolve())
        mismatch = self.drive / 'wrong-results.zip'
        with zipfile.ZipFile(mismatch, 'w') as archive:
            archive.writestr('source-git.json', json.dumps({'runName': 'original'}))
        with self.assertRaisesRegex(ValueError, 'disagree'):
            create_layout('resume', self.drive, 'ignored', mismatch)
        with self.assertRaisesRegex(ValueError, 'different'):
            create_layout('resume', self.drive, 'ignored', source, self.drive / 'runs/other')

    def test_ambiguous_zip_needs_explicit_root(self):
        source = self.drive / 'download.zip'
        with zipfile.ZipFile(source, 'w') as archive: archive.writestr('full/training-run.json', '{}')
        with self.assertRaisesRegex(ValueError, 'RESUME_RUN_ROOT'):
            create_layout('resume', self.drive, 'ignored', source)
        layout = create_layout('resume', self.drive, 'ignored', source, self.drive / 'runs/restored')
        self.assertEqual(layout['runRoot'].name, 'restored')

    def test_existing_original_directory_is_preferred_to_older_zip(self):
        source = self.drive / 'original-results.zip'
        with zipfile.ZipFile(source, 'w') as archive: archive.writestr('full/training-run.json', '{}')
        self.assertEqual(create_layout('resume', self.drive, 'ignored', source)['source'], self.root.resolve())

    def test_preserved_evidence_and_archive_are_small_and_self_contained(self):
        attempt = create_layout('resume', self.drive, 'ignored', self.root)['attemptRoot']
        self.full(attempt)
        stage = Path(self.temporary.name) / 'vm-stage'
        best = stage / 'full/checkpoints/checkpoint-1'
        checkpoint = stage / 'full/checkpoints/checkpoint-2'
        best.mkdir(parents=True)
        checkpoint.mkdir()
        for filename in ('adapter_model.safetensors', 'adapter_config.json'):
            (best / filename).write_bytes(b'original best')
        (checkpoint / 'trainer_state.original.json').write_text('{}')
        (checkpoint / 'trainer_state.json').write_text('{}')
        (stage / 'full/training-run.json').write_text('{}')
        (checkpoint / 'optimizer.pt').write_bytes(b'large optimizer excluded')
        report = {'checkpoint': str(checkpoint), 'relocatedBestCheckpoint': str(best)}
        (stage / 'resume-restoration.json').write_text(json.dumps(report))
        preserve_resume_evidence(attempt, stage)
        (attempt / 'old.zip').write_bytes(b'large prior ZIP')
        archive = archive_attempt(self.root, attempt)
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            self.assertIn('resume-source/full/checkpoints/checkpoint-1/adapter_model.safetensors', names)
            self.assertIn('resume-source/resume-restoration.json', names)
            self.assertFalse(any(name.endswith(('.zip', 'optimizer.pt')) for name in names))
        with self.assertRaises(FileExistsError): archive_attempt(self.root, attempt)

    def test_fresh_layout_still_uses_root(self):
        layout = create_layout('fresh', self.drive, 'fresh', None)
        self.assertEqual(layout['runRoot'], layout['attemptRoot'])
        with self.assertRaises(FileExistsError): create_layout('fresh', self.drive, 'fresh', None)


if __name__ == '__main__': unittest.main()
