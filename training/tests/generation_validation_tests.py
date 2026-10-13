"""Checkpoint selection tests: no downloads, full training or equipment access."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from generation_validation import aggregate, select_validation, CRITICAL_NO_TOOL, make_trainer_class, generate_validation, validate_smoke
from evaluation_state import select_cases
from v2_contracts import definitions

SPEC = {'casesPerTool': 2, 'otherNoToolCasesPerCategory': 2, 'maxNewTokens': 192}


def result(name='get_status', correct=True, text='plain reply', valid=True, call=None, truncated=False):
    return {'expected': {'first_tool': name, 'arguments': {} if name else None},
            'toolChoiceCorrect': correct, 'exactArgumentsCorrect': correct,
            'schemaValid': valid, 'text': text, 'call': call, 'outputTruncated': truncated}


class SelectionTests(unittest.TestCase):
    def test_all_tools_languages_and_critical_no_tool_cases_are_retained(self):
        rows = select_cases('validation_smoke', prompt_mode='chat')[0]
        selected = select_validation(rows, SPEC)
        self.assertEqual(selected, select_validation(list(reversed(rows)), SPEC))
        self.assertEqual({r['expected'].get('first_tool', (r['expected'].get('tool_sequence') or [None])[0])
                          for r in selected} - {None}, set(definitions()))
        self.assertEqual(len({r['locale'] for r in selected}), 10)
        for category in CRITICAL_NO_TOOL:
            self.assertEqual({r['id'] for r in selected if r['category'] == category},
                             {r['id'] for r in rows if r['category'] == category})
        self.assertEqual(len(selected), len({r['id'] for r in selected}))

    def test_macro_score_prevents_frequent_tools_hiding_missing_tools(self):
        metrics = aggregate([result()] * 20 + [result('get_job', False)])
        self.assertAlmostEqual(metrics['tool_exact'], 20/21)
        self.assertEqual(metrics['tool_macro_exact'], 0.5)

    def test_safety_and_truncation_dominate_checkpoint_selection(self):
        safe = aggregate([result(correct=False), result(None)])
        for bad in (result(None, False, '<tool_call>broken', valid=False),
                    result(None, False, call={'name': 'start_sequence', 'arguments': {}}),
                    result(truncated=True)):
            metrics = aggregate([result()] * 100 + [bad])
            self.assertLess(metrics['selection_score'], 0)
            self.assertGreaterEqual(safe['selection_score'], 0)

    def test_empty_or_unbounded_selection_is_rejected(self):
        with self.assertRaises(ValueError): aggregate([])
        with self.assertRaises(ValueError): select_validation([], dict(SPEC, casesPerTool=0))

    def test_full_run_rejects_untested_recipe_or_hardware(self):
        hardware = {'ready':True, 'backend':'rocm', 'python':'3.11', 'packages':{'torch':'fixture'},
                    'gpu':{'name':'MI355X', 'architecture':'gfx950', 'totalMiB':1000, 'freeMiB':900}}
        config = {'generationValidation':SPEC, 'learningRate':5e-5}
        smoke = {'hardware':copy.deepcopy(hardware), 'config':copy.deepcopy(config)}
        validate_smoke(smoke, config, dict(hardware, gpu=dict(hardware['gpu'], freeMiB=800)))
        for field, value in [('name','different GPU'), ('architecture','gfx1100'), ('totalMiB',500)]:
            changed = copy.deepcopy(hardware); changed['gpu'][field] = value
            with self.assertRaises(ValueError): validate_smoke(smoke, config, changed)
        with self.assertRaises(ValueError): validate_smoke(smoke, dict(config, learningRate=3e-5), hardware)
        with self.assertRaises(ValueError): validate_smoke(smoke, config, dict(hardware, packages={'torch':'changed'}))

    def test_accuracy_metrics_persist_without_replacing_losses(self):
        from metrics import TrainingMetrics
        with tempfile.TemporaryDirectory() as directory:
            monitor = TrainingMetrics(directory)
            state = SimpleNamespace(global_step=100, max_steps=300, epoch=0.5)
            logged = {'eval_loss':0.2, **{'eval_' + k:v for k,v in aggregate([result()]).items()}}
            monitor.on_log(None, state, None, logged)
            latest = json.loads((Path(directory)/'metrics-latest.json').read_text())
            self.assertEqual(latest['eval_loss'], .2)
            self.assertEqual(latest['best_eval_loss'], .2)
            self.assertEqual(latest['eval_selection_score'], 1)
            self.assertEqual(latest['eval_generation_cases'], 1)
            with self.assertRaises(ValueError): monitor.on_log(None, state, None, dict(logged, eval_tool_exact=float('nan')))


@unittest.skipUnless(importlib.util.find_spec('torch'), 'CPU torch integration is optional in offline CI')
class TorchIntegrationTests(unittest.TestCase):
    def test_generation_restores_mode_cache_and_training_rng(self):
        import torch
        class Batch(dict):
            def to(self, device): return Batch({k:v.to(device) for k,v in self.items()})
        class Tokenizer:
            eos_token_id = 2
            unk_token_id = 0
            def convert_tokens_to_ids(self, text): return 2
            def __call__(self, *args, **kwargs):
                return Batch(input_ids=torch.tensor([[3, 4]]), attention_mask=torch.tensor([[1, 1]]))
            def decode(self, *args, **kwargs):
                return '<tool_call>{"name":"get_status","arguments":{}}</tool_call><|im_end|>'
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.emb = torch.nn.Embedding(5, 2)
                self.config = SimpleNamespace(use_cache=False)
            def get_input_embeddings(self): return self.emb
            def generate(self, input_ids, **kwargs):
                assert not self.training and self.config.use_cache and not torch.is_grad_enabled()
                torch.rand(1)
                return torch.cat([input_ids, torch.tensor([[1, 2]])], dim=1)
        model, tokenizer = Model(), Tokenizer()
        case = {'id': 'unit', 'locale': 'en', 'category': 'status', 'caseSha256': 'case',
                'promptSha256': 'prompt', 'prompt': 'unit', 'expected': {'first_tool':'get_status','arguments':{}},
                'allowedNames':['get_status'], 'toolMode':'chat'}
        before = torch.random.get_rng_state().clone()
        metrics, predictions = generate_validation(model, tokenizer, [case], 192)
        self.assertEqual(metrics['tool_exact'], 1)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
        self.assertTrue(model.training); self.assertFalse(model.config.use_cache)
        with patch.object(model, 'generate', side_effect=RuntimeError('fixture failure')):
            with self.assertRaises(RuntimeError): generate_validation(model, tokenizer, [case], 192)
        self.assertTrue(model.training); self.assertFalse(model.config.use_cache)

    def test_real_cpu_trainer_receives_selection_metrics_before_callbacks(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments, TrainerCallback
        received = []
        class Observe(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs): received.append(dict(metrics))
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8))
        metrics = aggregate([result()])
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, report_to=[], disable_tqdm=True,
                         prediction_loss_only=True, metric_for_best_model='eval_selection_score', greater_is_better=True)
            with patch('generation_validation.generate_validation', return_value=(metrics, [result()])):
                trainer = make_trainer_class()(model=model, args=args,
                    eval_dataset=[{'input_ids':[3,4,5], 'attention_mask':[1,1,1], 'labels':[-100,4,5]}],
                    generation_cases=['fixture'], generation_contract={'selection':SPEC},
                    generation_output=Path(directory)/'generation', callbacks=[Observe()])
                measured = trainer.evaluate()
            self.assertEqual(len(received), 1)
            self.assertEqual(received[0]['eval_selection_score'], 1)
            self.assertIn('eval_loss', measured)
            self.assertTrue(trainer._determine_best_metric(measured, trial=None))
            self.assertEqual(trainer.state.best_metric, 1)
            self.assertTrue((Path(directory)/'generation/step-000000.json').is_file())

    def test_cpu_training_saves_and_selects_accuracy_over_lower_loss(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM, TrainingArguments
        model = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                 num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, head_dim=8))
        parent = make_trainer_class()
        class ControlledLossTrainer(parent):
            def evaluation_loop(self, *args, **kwargs):
                output = super().evaluation_loop(*args, **kwargs)
                output.metrics['eval_loss'] = .1 if self.state.global_step == 1 else .9
                return output
        scores = [dict(aggregate([result()]), selection_score=value) for value in (.25, .5)]
        row = {'input_ids':[3,4,5], 'attention_mask':[1,1,1], 'labels':[-100,4,5]}
        with tempfile.TemporaryDirectory() as directory:
            args = TrainingArguments(output_dir=directory, use_cpu=True, report_to=[], disable_tqdm=True,
                  max_steps=2, per_device_train_batch_size=1, eval_strategy='steps', save_strategy='steps',
                  eval_steps=1, save_steps=1, load_best_model_at_end=True, save_total_limit=2,
                  prediction_loss_only=True, metric_for_best_model='eval_selection_score', greater_is_better=True)
            with patch('generation_validation.generate_validation', side_effect=[(m,[result()]) for m in scores]):
                trainer = ControlledLossTrainer(model=model, args=args, train_dataset=[row,row], eval_dataset=[row],
                    generation_cases=['fixture'], generation_contract={'selection':SPEC},
                    generation_output=Path(directory)/'generation')
                trainer.train()
            self.assertEqual(trainer.state.best_metric, .5)
            self.assertTrue(trainer.state.best_model_checkpoint.endswith('checkpoint-2'))
            self.assertEqual([r['eval_loss'] for r in trainer.state.log_history if 'eval_loss' in r], [.1,.9])
            state = json.loads((Path(directory)/'checkpoint-2/trainer_state.json').read_text())
            self.assertEqual(state['best_metric'], .5)
            self.assertTrue((Path(directory)/'generation/step-000002.json').is_file())


if __name__ == '__main__': unittest.main()
