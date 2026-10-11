"""Contract, assistant loss masking and holdout protections with the real pinned tokenizer."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from common import read_jsonl, sha256
from v2_contracts import PACKAGE, VERIFIED, assistant_example, definitions, parse_prediction, validate_call


class V2Tests(unittest.TestCase):
    def test_original_package_hashes_and_holdouts(self):
        manifest = json.loads((PACKAGE / 'data/manifest.json').read_text())
        for name, expected in manifest['sha256'].items():
            self.assertEqual(sha256(PACKAGE / 'data' / name), expected)
        groups = [{r['group'] for r in read_jsonl(PACKAGE / 'data/audited' / f'{s}.jsonl')}
                  for s in ('train', 'validation', 'test')]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(len(definitions()), 32)
        self.assertIn('wait_seconds', definitions()['stop_guiding']['inputSchema']['properties'])

    def test_all_audited_arguments_match_current_contract(self):
        for split in ('train', 'validation', 'test'):
            for row in read_jsonl(PACKAGE / 'data/audited' / f'{split}.jsonl'):
                for i, message in enumerate(row['messages']):
                    for call in message.get('tool_calls', []):
                        fn = call['function']
                        validate_call(fn['name'], fn['arguments'], [t['name'] for t in row['tools']])
                        self.assertEqual(row['messages'][i+1]['name'], fn['name'])
                        if not definitions()[fn['name']]['annotations']['readOnlyHint']:
                            self.assertIn('server must re-check): ' + fn['name'], row['messages'][0]['content'])

    def test_real_tokenizer_masks_and_native_tool_markers(self):
        if not (ROOT / '.local/training/base/tokenizer_config.json').is_file():
            self.skipTest('Pinned tokenizer is not installed; no automatic download')
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(ROOT / '.local/training/base', local_files_only=True)
        rows = read_jsonl(PACKAGE / 'data/audited/train.jsonl')
        examples = [next(r for r in rows if r['category'] == c) for c in
                    ('job_followup', 'tool_success', 'missing_exposure', 'chat_math')]
        for row in examples:
            for turn, message in enumerate(row['messages']):
                if message['role'] != 'assistant': continue
                example = assistant_example(tokenizer, row, turn, 1280)
                start = next(i for i, value in enumerate(example['labels']) if value != -100)
                self.assertTrue(all(v == -100 for v in example['labels'][:start]))
                self.assertEqual(example['labels'][start:], example['input_ids'][start:])
                self.assertIn('<|im_end|>', example['completion'])
                if message.get('tool_calls'):
                    self.assertIn('<tool_call>', example['completion'])
                with self.assertRaises(ValueError): assistant_example(tokenizer, row, turn, 1)

    def test_prediction_rejects_invented_and_partial_calls(self):
        for text in ('<tool_call>{"name":"invented","arguments":{}}</tool_call>',
                     '<tool_call>{"name":"get_status"',
                     '<tool_call>{"name":"get_status","arguments":{"extra":1}}</tool_call>',
                     '{"name":"start_sequence","arguments":{"skip_validation":true}}<|im_end|>',
                     '<tool_call>{"name":"get_status","arguments":{}}</tool_call><tool_call>',
                     '<tool_call>[]</tool_call>'):
            with self.assertRaises(Exception): parse_prediction(text, ['get_status'])
        self.assertIsNone(parse_prediction('No action was performed.', []))
        self.assertEqual(parse_prediction('<tool_call>{"name":"get_status","arguments":{}}</tool_call>', ['get_status'])['name'], 'get_status')

    def test_adapter_constraints_are_stricter_than_published_schema(self):
        plan = {'name': 'Field X', 'ra': 5.5, 'dec': -4.2,
                'exposures': [{'exposure_seconds': 60, 'count': 10}], 'dither_every': 32}
        validate_call('add_target', plan, ['add_target'])
        plan['dither_every'] = 33
        with self.assertRaises(ValueError): validate_call('add_target', plan, ['add_target'])
        for filename in ('session.template', 'session.TEMPLATE.json'):
            with self.assertRaises(ValueError): validate_call('save_sequence', {'name': filename}, ['save_sequence'])

    def test_feedback_distinguishes_sequence_start_stop_and_status(self):
        rows = read_jsonl(PACKAGE / 'data/audited/train.jsonl')
        status = next(json.loads(m['content']) for row in rows for m in row['messages']
                      if m['role'] == 'tool' and m['name'] == 'get_status')
        self.assertEqual(len({d['device'] for d in status['devices']}), 11)
        self.assertTrue(all(d['name'] is None for d in status['devices'] if not d['connected']))
        for name in ('start_sequence', 'stop_sequence'):
            row = next(r for r in rows if r['locale'] == 'en' and any(
                m['role'] == 'tool' and m['name'] == name for m in r['messages']))
            payload = next(json.loads(m['content']) for m in row['messages'] if m['role'] == 'tool' and m['name'] == name)
            self.assertTrue(payload['success'])
            self.assertNotIn('completed successfully', row['messages'][-1]['content'])
            if name == 'stop_sequence':
                self.assertIn('status', row['messages'][-1]['content'])
                self.assertNotIn('job', row['messages'][-1]['content'])


if __name__ == '__main__':
    unittest.main()
