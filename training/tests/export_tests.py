"""Export source guards; no model loading, training or equipment access."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export import export_adapter


class SourceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / 'full'
        self.checkpoint = self.run / 'checkpoints/checkpoint-400'
        self.checkpoint.mkdir(parents=True)
        (self.run / 'training-run.json').write_text(json.dumps({'status': 'started', 'smokeOnly': False}))
        self.state = {'global_step': 400, 'max_steps': 2184, 'best_global_step': 400,
                      'best_model_checkpoint': '/content/drive/full/checkpoints/checkpoint-400'}
        (self.checkpoint / 'trainer_state.json').write_text(json.dumps(self.state))
        (self.checkpoint / 'adapter_config.json').write_text(json.dumps({'peft_type': 'LORA', 'task_type': 'CAUSAL_LM'}))
        (self.checkpoint / 'adapter_model.safetensors').write_bytes(b'fixture-hash-only')

    def test_incomplete_run_rejected_even_with_smoke_opt_in(self):
        with self.assertRaisesRegex(ValueError, 'completed'):
            export_adapter(self.run, allow_smoke=True)

    def test_explicit_saved_best_checkpoint_is_diagnostic(self):
        run, adapter, directory, metadata = export_adapter(checkpoint=self.checkpoint)
        self.assertEqual(adapter, self.checkpoint)
        self.assertEqual(directory, self.run)
        self.assertEqual(run['status'], 'started')
        self.assertTrue(metadata['diagnosticIncompleteCheckpoint'])
        self.assertEqual(metadata['globalStep'], 400)
        self.assertEqual(metadata['targetSteps'], 2184)

    def test_nonbest_checkpoint_rejected(self):
        self.state['best_global_step'] = 300
        (self.checkpoint / 'trainer_state.json').write_text(json.dumps(self.state))
        with self.assertRaisesRegex(ValueError, 'best saved'):
            export_adapter(checkpoint=self.checkpoint)

    def test_completed_run_still_requires_best_checkpoint_selection(self):
        (self.run / 'training-run.json').write_text(json.dumps({'status': 'completed', 'smokeOnly': False}))
        self.state['best_global_step'] = 300
        (self.checkpoint / 'trainer_state.json').write_text(json.dumps(self.state))
        with self.assertRaisesRegex(ValueError, 'best saved'):
            export_adapter(checkpoint=self.checkpoint)

    def test_step_outside_original_schedule_rejected(self):
        self.state['max_steps'] = 300
        (self.checkpoint / 'trainer_state.json').write_text(json.dumps(self.state))
        with self.assertRaisesRegex(ValueError, 'step metadata'):
            export_adapter(checkpoint=self.checkpoint)

    def test_base_control_has_no_adapter(self):
        self.assertEqual(export_adapter(), (None, None, None, None))


if __name__ == '__main__':
    unittest.main()
