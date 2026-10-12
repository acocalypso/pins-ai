"""Offline resume and archive safety tests; no model generation or MCP execution."""
import copy
import io
import json
from pathlib import Path
import stat
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training'))
from common import fingerprint
from evaluation_state import new_report, resume_report, score, summarize, ProgressWriter, select_cases
from restore_results import checked_members
from compare_evaluations import load_reports
from evaluate_v2 import adapter_metadata


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.cases = [{'id': str(i), 'locale': locale, 'category': 'tool_success',
                       'expected': {'first_tool': 'get_status', 'arguments': {}},
                       'allowedNames': ['get_status'], 'caseSha256': fingerprint(['case', i]),
                       'promptSha256': fingerprint(['prompt', i])} for i, locale in enumerate(('en', 'de', 'en'))]
        self.identity = {'model': 'adapter', 'adapter': {'weightsSha256': 'adapter-a'},
                         'base': {'weightsSha256': 'base-a'}, 'suite': 'validation_smoke',
                         'inputSha256': 'dataset-a', 'scorerVersion': 'scorer-a', 'policySha256': 'policy-a',
                         'templateSha256': 'template-a', 'generation': {'maxNewTokens': 128},
                         'backend': 'cuda-nf4', 'selection': {'perLanguage': 0}}
        self.identity['caseInputsSha256'] = fingerprint([{k: c[k] for k in ('id', 'caseSha256', 'promptSha256')} for c in self.cases])
        self.current = new_report(self.identity, self.cases)
        self.source = self.root / 'original.json'

    def result(self, index, text='<tool_call>{"name":"get_status","arguments":{}}</tool_call><|im_end|>'):
        case = self.cases[index]
        return {**{k: case[k] for k in ('id', 'locale', 'category', 'expected', 'caseSha256', 'promptSha256')},
                **score(case, text), 'text': text, 'seconds': 1.0, 'outputTokens': 20,
                'tokensPerSecond': 20.0, 'outputTruncated': False, 'conditionalPoll': False}

    def save(self, results):
        report = copy.deepcopy(self.current)
        report['results'] = results
        summarize(report)
        self.source.write_text(json.dumps(report), encoding='utf-8')
        return report

    def test_successful_resume_and_totals_without_double_counting(self):
        self.save([self.result(0)])
        original = self.source.read_bytes()
        report = resume_report(self.source, self.current, self.cases)
        report['results'] += [self.result(1), self.result(2)]
        summarize(report)
        writer = ProgressWriter(self.root / 'continued.json', self.source)
        writer.write(report)
        self.assertTrue(report['completed'])
        self.assertEqual(report['overall']['cases'], 3)
        self.assertEqual(report['perLanguage']['en']['cases'], 2)
        self.assertEqual(report['perCategory']['tool_success']['exactArgumentsCorrect'], 3)
        self.assertEqual(self.source.read_bytes(), original)

    def test_interrupted_resume_remains_incomplete_and_can_resume_again(self):
        self.save([self.result(0)])
        report = resume_report(self.source, self.current, self.cases)
        report['results'].append(self.result(1))
        summarize(report)
        next_path = self.root / 'interrupted.json'
        ProgressWriter(next_path, self.source).write(report)
        self.assertFalse(report['completed'])
        resumed = resume_report(next_path, self.current, self.cases)
        self.assertEqual([r['id'] for r in resumed['results']], ['0', '1'])
        self.assertEqual(resumed['overall']['cases'], 2)

    def test_duplicate_case_ids_rejected(self):
        self.save([self.result(0), self.result(0)])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            resume_report(self.source, self.current, self.cases)

    def test_identity_mismatches_rejected(self):
        self.save([self.result(0)])
        for key in ('adapter', 'base', 'inputSha256', 'scorerVersion', 'policySha256', 'templateSha256', 'generation', 'backend', 'suite'):
            with self.subTest(key=key):
                current = copy.deepcopy(self.current)
                current['identity'][key] = 'different'
                with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                    resume_report(self.source, current, self.cases)

    def test_recorded_inputs_and_scores_revalidated(self):
        for key, value in [('expected', {}), ('promptSha256', 'wrong'), ('caseSha256', 'wrong'),
                           ('locale', 'fr'), ('toolChoiceCorrect', False), ('text', 'changed'), ('outputTokens', 129)]:
            with self.subTest(key=key):
                result = self.result(0); result[key] = value
                self.save([result])
                with self.assertRaises(ValueError):
                    resume_report(self.source, self.current, self.cases)

    def test_validly_recorded_wrong_predictions_are_skipped_too(self):
        self.save([self.result(0, 'Cannot help.')])
        report = resume_report(self.source, self.current, self.cases)
        self.assertEqual(len(report['results']), 1)
        self.assertEqual(report['overall']['toolChoiceCorrect'], 0)

    def test_out_of_order_cases_rejected(self):
        self.save([self.result(1)])
        with self.assertRaisesRegex(ValueError, 'prefix'):
            resume_report(self.source, self.current, self.cases)

    def test_false_completion_rejected(self):
        report = self.save([self.result(0)])
        report['completed'] = True
        self.source.write_text(json.dumps(report), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'falsely marked'):
            resume_report(self.source, self.current, self.cases)

    def test_legacy_report_not_silently_mixed(self):
        self.source.write_text('{"model":"base","results":[]}', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Legacy report'):
            resume_report(self.source, self.current, self.cases)

    def test_existing_output_and_original_cannot_be_overwritten(self):
        self.save([])
        with self.assertRaises(ValueError): ProgressWriter(self.source, self.source)
        with self.assertRaises(FileExistsError): ProgressWriter(self.source)

    def test_failed_atomic_replace_preserves_previous_progress(self):
        path = self.root / 'progress.json'
        writer = ProgressWriter(path)
        writer.write(self.current)
        before = path.read_bytes()
        with patch('evaluation_state.os.replace', side_effect=OSError('simulated interruption')):
            with self.assertRaises(OSError): writer.write({'changed': True})
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_real_smoke_selection_is_20_cases_and_has_no_polling(self):
        cases, _, _ = select_cases('validation_smoke', 2, True)
        self.assertEqual(len(cases), 20)
        self.assertEqual(len({c['locale'] for c in cases}), 10)
        self.assertFalse(any(c.get('conditionalPoll') for c in cases))

    def test_comparison_rejects_incompatible_and_incomplete_evidence(self):
        report = self.save([self.result(i) for i in range(3)])
        other = copy.deepcopy(report)
        other['identity']['adapter']['weightsSha256'] = 'adapter-b'
        other['identitySha256'] = fingerprint(other['identity'])
        second = self.root / 'second.json'
        second.write_text(json.dumps(other), encoding='utf-8')
        self.assertEqual(len(load_reports({'v2': self.source, 'v3': second})), 2)
        other['identity']['generation']['maxNewTokens'] = 64
        other['identitySha256'] = fingerprint(other['identity'])
        second.write_text(json.dumps(other), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Incompatible'):
            load_reports({'v2': self.source, 'v3': second})
        other['completed'] = False
        second.write_text(json.dumps(other), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'completed'):
            load_reports({'v3': second})

    def test_incomplete_checkpoint_requires_explicit_diagnostic_and_best_metadata(self):
        run = self.root / 'full'
        adapter = run / 'checkpoints/checkpoint-700'
        adapter.mkdir(parents=True)
        (run / 'training-run.json').write_text('{"status":"started"}', encoding='utf-8')
        (adapter / 'adapter_config.json').write_text('{"peft_type":"LORA","task_type":"CAUSAL_LM"}', encoding='utf-8')
        (adapter / 'adapter_model.safetensors').write_bytes(b'fixture hash only')
        state = {'global_step': 700, 'best_global_step': 700, 'best_model_checkpoint': '/colab/checkpoints/checkpoint-700'}
        (adapter / 'trainer_state.json').write_text(json.dumps(state), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'did not complete'):
            adapter_metadata(adapter)
        _, identity = adapter_metadata(adapter, True)
        self.assertTrue(identity['diagnosticIncompleteCheckpoint'])
        state['best_global_step'] = 600
        (adapter / 'trainer_state.json').write_text(json.dumps(state), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'best saved'):
            adapter_metadata(adapter, True)


class ArchiveTests(unittest.TestCase):
    def check(self, names):
        data = io.BytesIO()
        with zipfile.ZipFile(data, 'w') as archive:
            for name in names:
                # Windows ZipInfo construction normalizes backslashes; simulate
                # externally authored member names rather than sanitizing the fixture.
                if isinstance(name, str):
                    entry = zipfile.ZipInfo('placeholder')
                    entry.filename = name
                else:
                    entry = name
                archive.writestr(entry, 'data')
        data.seek(0)
        with zipfile.ZipFile(data) as archive: return checked_members(archive)

    def test_unsafe_paths_rejected(self):
        for name in ('../escape', '/absolute', 'C:/absolute', 'folder\\escape', 'a/../escape', 'NUL.txt', 'name:stream', 'file. '):
            with self.subTest(name=name), self.assertRaises(ValueError): self.check([name])

    def test_casefold_duplicates_and_parent_collisions_rejected(self):
        for names in (['File.json', 'file.json'], ['a', 'a/b.json']):
            with self.assertRaises(ValueError): self.check(names)

    def test_symlink_rejected(self):
        entry = zipfile.ZipInfo('link'); entry.create_system = 3
        entry.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaisesRegex(ValueError, 'Non-regular'): self.check([entry])

    def test_normal_paths_accepted(self):
        self.assertEqual(len(self.check(['full/training-run.json', 'full/checkpoints/checkpoint-700/adapter_model.safetensors'])), 2)


if __name__ == '__main__':
    unittest.main()
