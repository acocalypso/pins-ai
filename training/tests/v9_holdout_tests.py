"""Integrity/contract checks only; no models, hardware, downloads or mutation."""
import hashlib
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
from build_v9_holdout import DATA, MANIFEST, build_rows, freeze
from chat_dataset_common import LANGUAGES, validate_chat_call


class FutureHoldoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload = DATA.read_bytes()
        cls.rows = [json.loads(line) for line in cls.payload.splitlines()]
        cls.manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))

    def test_frozen_integrity_and_reproduction(self):
        self.assertEqual(hashlib.sha256(self.payload).hexdigest(), self.manifest['sha256'])
        expected = ''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n' for row in build_rows()).encode('utf-8')
        self.assertEqual(self.payload, expected)
        with self.assertRaisesRegex(SystemExit, 'Refusing to overwrite'):
            freeze()
        self.assertEqual(DATA.read_bytes(), self.payload)

    def test_balance_and_translation_groups(self):
        self.assertEqual(len(self.rows), 100)
        self.assertEqual(Counter(row['label'] for row in self.rows), {'positive': 50, 'negative': 50})
        self.assertEqual(len({row['id'] for row in self.rows}), 100)
        groups = {row['group'] for row in self.rows}
        self.assertEqual(len(groups), 10)
        for group in groups:
            rows = [row for row in self.rows if row['group'] == group]
            self.assertEqual({row['locale'] for row in rows}, set(LANGUAGES))
            self.assertEqual(len({json.dumps(row['expected'], sort_keys=True) for row in rows}), 1)

    def test_targets_contracts_and_prompt_history(self):
        histories = 0
        for row in self.rows:
            self.assertEqual(row['evaluation_turn'], len(row['messages']) - 1)
            self.assertLessEqual(len(row['tools']), 2)
            self.assertEqual(row['profile'], {'context_tokens': 1536, 'max_new_tokens': 192, 'max_tools': 2})
            names = [tool['name'] for tool in row['tools']]
            for message in row['messages']:
                for call in message.get('tool_calls', []):
                    fn = call['function']
                    validate_chat_call(fn['name'], fn['arguments'], names)
                    self.assertFalse(any(fn['arguments'].get(key) for key in ('force', 'overwrite', 'skip_validation', 'abandon')))
                    self.assertLessEqual(fn['arguments'].get('wait_seconds', 0), 5)
            calls = row['messages'][-1].get('tool_calls', [])
            if row['label'] == 'positive':
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]['function'], {'name': row['expected']['first_tool'], 'arguments': row['expected']['arguments']})
            else:
                self.assertFalse(calls)
                self.assertIsNone(row['expected']['first_tool'])
                self.assertIsNone(row['expected']['arguments'])
                self.assertEqual(row['expected']['tool_sequence'], [])
            histories += any(message['role'] == 'tool' for message in row['messages'])
        self.assertEqual(histories, 10)

    def test_future_isolation_provenance(self):
        for row in self.rows:
            provenance = row['provenance']
            self.assertEqual(provenance['split'], 'future_holdout_v9')
            self.assertEqual(provenance['domainReview'], 'pending')
            self.assertEqual(provenance['linguisticReview'], 'pending-native-review')
            for key in ('hardwareExecution', 'usedForTraining', 'usedForCheckpointSelection', 'usedForTuning'):
                self.assertIs(provenance[key], False)
        self.assertTrue(self.manifest['frozen'])
        self.assertIn('incomplete', self.manifest['coverage']['status'])
        self.assertEqual(self.manifest['nativeReview'], 'pending')

    def test_evaluation_keeps_history_but_excludes_expected_final_answer(self):
        from evaluation_state import select_cases
        with self.assertRaisesRegex(ValueError,'chat'):
            select_cases('future_holdout_v9')
        selected, _, _ = select_cases('future_holdout_v9',prompt_mode='chat')
        self.assertEqual(len(selected),100)
        for raw,case in zip(self.rows,selected):
            self.assertEqual(case['messages'],raw['messages'][:raw['evaluation_turn']])
            self.assertEqual(case['expected'],raw['expected'])
            self.assertNotEqual(case['messages'][-1]['role'],'assistant')
        self.assertEqual(sum(any(m['role']=='tool' for m in r['messages']) for r in selected),10)
        self.assertEqual(DATA.read_bytes(),self.payload)


if __name__ == '__main__':
    unittest.main()
