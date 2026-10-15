"""Offline continuation provenance and resumed-loss accounting tests."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training'))
from training_resume import validate_resume, select_checkpoint, saved_resume_config, REQUIRED
from metrics import TrainingMetrics
from common import write_jsonl, sha256


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.checkpoint = self.root / 'full/checkpoints/checkpoint-700'
        self.checkpoint.mkdir(parents=True)
        self.config = {'batchSize': 1, 'gradientAccumulation': 16}
        self.data, self.base = {'splits': 'unchanged'}, {'revision': 'pinned'}
        self.run = {'config': self.config, 'dataset': self.data, 'base': self.base, 'smokeOnly': False,
                    'datasetMode': 'mcp', 'method': 'qlora', 'precision': 'fp16', 'hardware': {'gpu': 'T4'}}
        self.state = {'global_step': 700, 'max_steps': 846, 'train_batch_size': 1,
                      'best_model_checkpoint': str(self.checkpoint), 'best_metric': 0.11}
        for name in REQUIRED: (self.checkpoint / name).write_bytes(b'fixture hash only')
        self.save()

    def save(self):
        (self.root / 'full/training-run.json').write_text(json.dumps(self.run), encoding='utf-8')
        (self.checkpoint / 'trainer_state.json').write_text(json.dumps(self.state), encoding='utf-8')

    def test_identical_contract_resumes_at_700_with_146_steps_remaining(self):
        _, _, info = validate_resume(self.checkpoint, self.config, self.data, self.base, 'qlora')
        self.assertEqual(info['targetSteps'] - info['startStep'], 146)
        self.assertEqual(len(info['checkpointFilesSha256']), len(REQUIRED))
        self.assertFalse(info['bitIdenticalToColab'])

    def test_notebook_uses_selected_run_config_without_changing_original(self):
        original = (self.root / 'full/training-run.json').read_bytes()
        output = self.root / 'new-session/resume-config.json'
        self.assertEqual(saved_resume_config(self.checkpoint, self.data, output), output)
        self.assertEqual(json.loads(output.read_text()), self.config)
        self.assertEqual((self.root / 'full/training-run.json').read_bytes(), original)
        self.assertEqual(saved_resume_config(self.checkpoint, self.data, output), output)

    def test_wrong_run_data_and_existing_foreign_config_are_rejected(self):
        output = self.root / 'resume-config.json'
        with self.assertRaisesRegex(ValueError, 'RESUME_SOURCE'):
            saved_resume_config(self.checkpoint, {'sourceProfile':'different-run'}, output)
        self.assertFalse(output.exists())
        output.write_text('{"batchSize":99}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'another source'):
            saved_resume_config(self.checkpoint, self.data, output)
        with self.assertRaisesRegex(ValueError, 'outside'):
            saved_resume_config(self.checkpoint, self.data, self.root / 'full/config.json')

    def test_bf16_lora_checkpoint_does_not_require_fp16_scaler(self):
        self.config['precision']='bf16'; self.run['precision']='bf16'; self.run['method']='lora'
        (self.checkpoint / 'scaler.pt').unlink(); self.save()
        _, _, info=validate_resume(self.checkpoint,self.config,self.data,self.base,'lora')
        self.assertNotIn('scaler.pt',info['checkpointFilesSha256'])
        prefix, _, _, _=self.select()
        self.assertIn('checkpoint-700',prefix)

    def test_fp16_checkpoint_still_requires_scaler(self):
        (self.checkpoint / 'scaler.pt').unlink()
        with self.assertRaises(FileNotFoundError): validate_resume(self.checkpoint,self.config,self.data,self.base,'qlora')

    def test_config_dataset_base_changes_are_rejected(self):
        for values in [({'batchSize': 1, 'gradientAccumulation': 8}, self.data, self.base),
                       (self.config, {'different': 'data'}, self.base),
                       (self.config, self.data, {'different': 'base'})]:
            with self.assertRaises(ValueError): validate_resume(self.checkpoint, *values, 'qlora')

    def test_missing_optimizer_or_best_checkpoint_is_rejected(self):
        (self.checkpoint / 'optimizer.pt').unlink()
        with self.assertRaises(FileNotFoundError): validate_resume(self.checkpoint, self.config, self.data, self.base, 'qlora')
        (self.checkpoint / 'optimizer.pt').write_bytes(b'fixture')
        self.state['best_model_checkpoint'] = '/nonexistent-colab-path'; self.save()
        with self.assertRaisesRegex(ValueError, 'relocated'): validate_resume(self.checkpoint, self.config, self.data, self.base, 'qlora')

    def test_smoke_and_exhausted_checkpoints_rejected(self):
        self.run['smokeOnly'] = True; self.save()
        with self.assertRaises(ValueError): validate_resume(self.checkpoint, self.config, self.data, self.base, 'qlora')
        self.run['smokeOnly'] = False; self.state['global_step'] = 846; self.save()
        with self.assertRaises(ValueError): validate_resume(self.checkpoint, self.config, self.data, self.base, 'qlora')

    def test_resumed_loss_not_mislabeled_whole_run(self):
        callback = TrainingMetrics(self.root / 'metrics', resume_start_step=700)
        callback.latest.update(eval_loss=0.12, best_eval_loss=0.12)
        state = SimpleNamespace(global_step=800, max_steps=846, epoch=2.83)
        callback.on_log(None, state, None, {'eval_loss': 0.11})
        self.assertAlmostEqual(callback.latest['eval_loss_delta'], -0.01)
        callback.on_log(None, state, None, {'train_loss': 0.0125})
        self.assertAlmostEqual(callback.latest['train_loss_segment'], 0.1)
        self.assertNotIn('train_loss_overall', callback.latest)

    def test_prepared_jsonl_hashes_use_linux_line_endings_on_windows(self):
        path = self.root / 'data.jsonl'
        write_jsonl(path, [{'x': 1}, {'x': 2}])
        self.assertEqual(path.read_bytes(), b'{"x": 1}\n{"x": 2}\n')

    def select(self):
        names = [p.relative_to(self.root).as_posix() for p in self.root.rglob('*') if p.is_file()]
        def read(name, size_only=False):
            path = self.root / name
            return path.stat().st_size if size_only else path.read_bytes()
        return select_checkpoint(names, read)

    def add_checkpoint(self, step, complete=True):
        path = self.checkpoint.parent / f'checkpoint-{step}'
        path.mkdir()
        state = dict(self.state, global_step=step, max_steps=1200)
        for name in REQUIRED:
            if complete or name != 'optimizer.pt': (path / name).write_bytes(b'fixture')
        (path / 'trainer_state.json').write_text(json.dumps(state))

    def test_latest_numeric_checkpoint_is_not_best_checkpoint(self):
        self.add_checkpoint(900); self.add_checkpoint(1000)
        prefix, best, state, skipped = self.select()
        self.assertIn('checkpoint-1000', prefix)
        self.assertIn('checkpoint-700', best)
        self.assertEqual(state['global_step'], 1000)
        self.assertEqual(skipped, [])

    def test_incomplete_latest_checkpoint_is_reported_and_skipped(self):
        self.add_checkpoint(800, complete=False)
        prefix, _, _, skipped = self.select()
        self.assertIn('checkpoint-700', prefix)
        self.assertEqual(skipped[0]['checkpoint'], 'checkpoint-800')
        self.assertIn('optimizer.pt', skipped[0]['reason'])

    def test_completed_run_is_not_restarted(self):
        self.run['status'] = 'completed'; self.save()
        with self.assertRaisesRegex(ValueError, 'incomplete'): self.select()

    def test_finished_checkpoint_does_not_fall_back_to_older_training(self):
        self.add_checkpoint(1200)
        with self.assertRaisesRegex(RuntimeError, 'finished'): self.select()


if __name__ == '__main__': unittest.main()
