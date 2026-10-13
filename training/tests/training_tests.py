"""Offline pipeline invariants; optional integration with the real pinned tokenizer."""
import copy
import json
import os
import re
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from common import completion_example, pad_examples, policy, read_jsonl, split_records, validate_record
from evaluate import compare, score
from export import commands
from import_tools import convert, reject_external_refs
from prepare import generate, minutes


class TokenizerFixture:
    eos_token = '!'
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        return ''.join(m['content'] for m in messages) + 'ASSISTANT:'
    def encode(self, text, **unused): return [ord(c) for c in text]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.gold = read_jsonl(ROOT / 'training/dataset/baseline/gold.jsonl')
        self.rows = generate() + read_jsonl(ROOT / 'training/dataset/baseline/authored.jsonl')

    def test_arithmetic_ground_truth(self):
        self.assertEqual(minutes(30, 120), '60')
        self.assertEqual(minutes(20, 180), '60')
        self.assertEqual(minutes(9, 75), '11.25')

    def test_grouped_splits_reproducible_and_disjoint(self):
        a = split_records(self.rows, {r['group'] for r in self.gold})
        b = split_records(list(reversed(self.rows)), {r['group'] for r in self.gold})
        self.assertEqual(a, b)
        groups = {k: {r['group'] for r in v} for k, v in a.items()}
        self.assertFalse(groups['train'] & groups['validation'] or groups['train'] & groups['test'] or groups['validation'] & groups['test'])
        self.assertEqual(sum(map(len, a.values())), 432)
        for split in a.values():
            for group in {r['group'] for r in split}:
                self.assertEqual({r['locale'] for r in split if r['group'] == group}, {'en', 'de'})
        arithmetic_groups = {k: {r['group'] for r in v if r.get('stratum') == 'arithmetic'} for k, v in a.items()}
        self.assertTrue(all(arithmetic_groups.values()))
        self.assertTrue(all(r['group'] == 'exposure-plan:' + r['id'].split(':')[1] for r in generate()))
        tiny = []
        for i in range(3):
            row = copy.deepcopy(self.rows[0]); row['id'] = row['group'] = f'tiny:{i}'
            row['messages'][0]['content'] = f'Tiny split fixture {i}'
            tiny.append(row)
        self.assertTrue(all(split_records(tiny, set()).values()))

    def test_gold_cases_unchanged(self):
        cases = json.loads((ROOT / 'plugin/benchmarks/cases.json').read_text(encoding='utf-8'))
        self.assertEqual(cases, [{k: r[k] for k in ('id', 'locale', 'messages')} for r in self.gold])

    def test_german_unicode_is_preserved(self):
        text = json.dumps(self.rows + self.gold, ensure_ascii=False)
        self.assertIn('Nachführung', text)
        self.assertFalse(re.search(r'\w\?\w', text))

    def test_gold_group_leakage_rejected(self):
        row = copy.deepcopy(self.rows[0]); row['group'] = self.gold[0]['group']
        with self.assertRaises(ValueError): split_records([row], {r['group'] for r in self.gold})

    def test_duplicate_prompts_rejected(self):
        row = copy.deepcopy(self.rows[0]); row['id'] = 'other'; row['group'] = 'different'
        with self.assertRaises(ValueError): split_records([self.rows[0], row], set())

    def test_private_or_tool_data_rejected_from_chat(self):
        row = copy.deepcopy(self.rows[0]); row['provenance']['containsPrivateData'] = True
        with self.assertRaises(ValueError): validate_record(row)
        row = copy.deepcopy(self.rows[0]); row['tools'] = ['invented']
        with self.assertRaises(ValueError): validate_record(row)

    def test_final_completion_only_masks_history(self):
        row = next(r for r in self.rows if len(r['messages']) == 4)
        encoded = completion_example(TokenizerFixture(), row, 10000)
        active = [x for x in encoded['labels'] if x != -100]
        self.assertEqual(''.join(map(chr, active)), row['messages'][-1]['content'] + '!')
        self.assertTrue(all(x == -100 for x in encoded['labels'][:-len(active)]))

    def test_padding_does_not_mask_real_eos(self):
        row = {'input_ids': [1, 2], 'attention_mask': [1, 1], 'labels': [-100, 2]}
        short = {'input_ids': [2], 'attention_mask': [1], 'labels': [2]}
        batch = pad_examples([row, short], 2)
        self.assertEqual(batch['labels'], [[-100, 2], [2, -100]])
        self.assertEqual(batch['attention_mask'], [[1, 1], [1, 0]])

    def test_overlength_target_never_truncated(self):
        with self.assertRaises(ValueError): completion_example(TokenizerFixture(), self.rows[0], 2)

    def test_absent_schema_and_external_refs_rejected(self):
        with self.assertRaises(ValueError): convert({'tools': []}, [])
        with self.assertRaises(ValueError): reject_external_refs({'$ref': 'https://example.invalid/schema'})

    def test_schema_import_validates_supplied_contract(self):
        from jsonschema import ValidationError
        # A schema-validation unit fixture, not a proposed PINS API or a mock MCP server.
        snapshot = {'tools': [{'name': 'schema_validation_fixture', 'inputSchema': {'type': 'object',
            'properties': {'value': {'type': 'integer', 'minimum': 1}}, 'required': ['value'], 'additionalProperties': False}}]}
        row = {'id': 'fixture', 'group': 'fixture:scenario', 'locale': 'en', 'reviewed': True,
               'provenance': {'source': 'unit fixture', 'license': 'MIT', 'containsPrivateData': False},
               'toolNames': ['schema_validation_fixture'], 'messages': [{'role': 'user', 'content': 'unit fixture'},
                {'role': 'assistant', 'content': None, 'tool_calls': [{'id': '1', 'type': 'function',
                 'function': {'name': 'schema_validation_fixture', 'arguments': {'value': 2}}}]}]}
        self.assertEqual(convert(snapshot, [row])[0]['tools'], snapshot['tools'])
        bad = copy.deepcopy(row); bad['messages'][-1]['tool_calls'][0]['function']['arguments'] = {'value': 'wrong'}
        with self.assertRaises(ValidationError): convert(snapshot, [bad])
        bad = copy.deepcopy(row); bad['toolNames'] = ['invented']
        with self.assertRaises(ValueError): convert(snapshot, [bad])

    def test_incomplete_or_changed_evaluation_rejected(self):
        with self.assertRaises(ValueError): score({'samples': [], 'sessions': []}, self.gold)
        report = self.report()
        report['samples'][0]['input']['messages'][0]['content'] = 'different prompt'
        with self.assertRaises(ValueError): score(report, self.gold)

    def report(self):
        config = {'contextTokens': 2048, 'threads': 1, 'maxOutputTokens': 128, 'temperature': .2,
                  'maxTemperatureC': 80, 'minAvailableMemoryMiB': 768, 'maxResidentMemoryMiB': 1200,
                  'maxProcessSwapMiB': 64, 'logLevel': 'Warning'}
        return {'runtime': {'commit': 'same-runtime'}, 'pluginRuntimeSha256': 'same-plugin',
                'nativeRuntimeSha256': 'same-native', 'promptSha256': 'same-prompt',
                'sessions': [{'model': {'id': 'test-model', 'file': 'test-Q4_K_M.gguf'}, 'config': config}],
                'samples': [{'model': 'test-model', 'threads': 1, 'case': row['id'],
                    'input': copy.deepcopy({k: row[k] for k in ('id', 'locale', 'messages')}),
                    'output': row['reference'], 'completed': True, 'error': None,
                    'firstContentSeconds': 2, 'outputTokensPerSecond': 16, 'peakRssMiB': 1000,
                    'peakSwapMiB': 0} for row in self.gold]}

    def reviews(self):
        return {r['id']: {'reviewer': 'unit fixture', 'languageOk': True, 'meaningOk': True, 'noFalseClaims': True} for r in self.gold}

    def test_review_and_regression_gates(self):
        baseline = score(self.report(), self.gold, self.reviews())
        self.assertEqual(baseline['reviewedAccuracy'], 1)
        self.assertTrue(compare(baseline, baseline)['eligibleForManualSelection'])
        for change in ({'medianTokensPerSecond': 1}, {'peakRssMiB': 1300}, {'smokeOnly': True},
                       {'promptSha256': None}, {'allReviewed': False}, {'nativeRuntimeSha256': 'changed'}):
            candidate = {**baseline, **change}
            self.assertFalse(compare(baseline, candidate)['eligibleForManualSelection'])
        no_review = score(self.report(), self.gold)
        self.assertIsNone(no_review['reviewedAccuracy'])
        self.assertFalse(compare(baseline, no_review)['eligibleForManualSelection'])
        self.assertFalse(compare(no_review, baseline)['eligibleForManualSelection'])
        candidate = copy.deepcopy(baseline); candidate['cases'][0]['pass'] = False
        self.assertFalse(compare(baseline, candidate)['eligibleForManualSelection'])
        self.assertFalse(compare(baseline, baseline)['tuningAttributionSupported'])
        control = {**baseline, 'baseRevision': 'same', 'converterSha256': 'same', 'quantizerSha256': 'same'}
        self.assertTrue(compare(control, control)['tuningAttributionSupported'])
        self.assertFalse(compare(control, {**control, 'baseRevision': 'different'})['eligibleForManualSelection'])

    def test_export_commands_use_f16_then_q4(self):
        calls = commands(Path('python'), Path('source'), Path('merged'), Path('output'))
        self.assertIn('f16', calls[0])
        self.assertEqual(calls[1][-1], 'Q4_K_M')
        control = commands(Path('python'), Path('source'), Path('merged'), Path('output'), 'qwen3-0.6b-training-control')
        self.assertNotEqual(control[1][-2], calls[1][-2])

    @unittest.skipUnless(os.environ.get('PINS_TRAINING_TOKENIZER'), 'Optional pinned-tokenizer integration')
    def test_real_qwen_template_and_all_example_lengths(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(os.environ['PINS_TRAINING_TOKENIZER'], local_files_only=True)
        self.assertEqual(tokenizer.eos_token, '<|im_end|>')
        for row in self.rows:
            result = completion_example(tokenizer, row, 512)
            answer = tokenizer.decode([x for x in result['labels'] if x != -100], skip_special_tokens=False)
            self.assertEqual(answer, row['messages'][-1]['content'] + tokenizer.eos_token)
            prefix = tokenizer.decode(result['input_ids'][:next(i for i, v in enumerate(result['labels']) if v != -100)])
            self.assertIn('<think>\n\n</think>', prefix)
        self.assertIn('astronomy information assistant', policy())


if __name__ == '__main__':
    unittest.main()
