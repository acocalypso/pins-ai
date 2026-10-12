"""Train-only curriculum isolation, language/device coverage and current-schema checks."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training'))
from common import fingerprint, read_jsonl
from curriculum_v3 import curriculum_rows
from v2_contracts import PACKAGE, definitions, validate_call


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = curriculum_rows('train')
        cls.validation = curriculum_rows('validation')

    def test_determinism_and_isolation(self):
        self.assertEqual(fingerprint(self.train), fingerprint(curriculum_rows('train')))
        self.assertEqual(self.validation, read_jsonl(PACKAGE / 'data/audited/validation.jsonl'))
        self.assertFalse({r['group'] for r in self.train} & {r['group'] for r in self.validation})
        self.assertFalse({r['group'] for r in self.train} & {r['group'] for r in read_jsonl(PACKAGE / 'data/test.jsonl')})
        with self.assertRaises(ValueError): curriculum_rows('test')
        with self.assertRaises(ValueError): curriculum_rows('hard_gold')

    def test_current_schema_and_language_device_matrix(self):
        devices = definitions()['connect_device']['inputSchema']['properties']['device']['enum']
        matrix = set()
        for row in self.train:
            for message in row['messages']:
                for call in message.get('tool_calls', []):
                    fn = call['function']
                    validate_call(fn['name'], fn['arguments'], [t['name'] for t in row['tools']])
                    if row['category'] == 'device_contrast':
                        matrix.add((row['locale'], fn['name'], fn['arguments']['device']))
        self.assertEqual(len(matrix), 10 * 2 * len(devices))

    def test_no_approval_does_not_train_a_call(self):
        cases = [r for r in self.train if r['id'].startswith('curriculum-device') and r['category'] == 'no_approval']
        self.assertGreater(len(cases), 0)
        for row in cases:
            self.assertNotIn('Trusted gateway authorization', row['messages'][0]['content'])
            self.assertFalse(any(m.get('tool_calls') for m in row['messages']))
            self.assertIsNone(row['expected']['first_tool'])

    def test_new_safety_matrix_and_exact_arguments(self):
        cases = [r for r in self.train if r['id'].startswith('curriculum-safety-')]
        self.assertEqual(len(cases), 110)
        self.assertEqual(len({(r['locale'], r['category']) for r in cases}), 110)
        for row in cases:
            answer = row['messages'][-1]
            if row['expected']['first_tool'] is None:
                self.assertFalse(answer.get('tool_calls'))
            else:
                fn = answer['tool_calls'][0]['function']
                self.assertEqual(fn['arguments'], row['expected']['arguments'])
                if row['category'] == 'safety_target_unicode_decimal':
                    self.assertIn(fn['arguments']['name'], row['messages'][-2]['content'])
                    self.assertNotEqual(fn['name'], 'start_sequence')

    def test_sequence_plans_are_train_only_and_never_start_or_clear(self):
        cases = [row for row in self.train if row['category'] == 'sequence_plan']
        self.assertEqual(len(cases), 22)
        self.assertFalse(any(row['category'] == 'sequence_plan' for row in self.validation))
        for row in cases:
            self.assertFalse(row['provenance']['hardwareExecution'])
            for message in row['messages']:
                for call in message.get('tool_calls', []):
                    self.assertEqual(call['function']['name'], 'add_target')
                    block = call['function']['arguments']['exposures'][0]
                    self.assertNotIn('filter', block)
        exact = next(row for row in cases if row['id'] == 'sequence-plan-minutes-0-en')
        self.assertEqual(exact['expected']['arguments']['exposures'], [{'exposure_seconds': 300, 'count': 300}])
        missing = next(row for row in cases if row['id'] == 'sequence-plan-m31-missing-coordinates-en')
        self.assertIsNone(missing['expected']['first_tool'])
        mixed = next(row for row in cases if row['id'] == 'sequence-plan-m31-mixed-language-missing-coordinates-de')
        self.assertIsNone(mixed['expected']['first_tool'])
        self.assertFalse(mixed['messages'][-1].get('tool_calls'))
        followup = next(row for row in cases if row['id'] == 'sequence-plan-m31-mixed-language-followup-de')
        self.assertEqual(followup['expected']['arguments']['exposures'], [{'exposure_seconds': 300, 'count': 300}])
        self.assertNotIn('template', followup['expected']['arguments'])


if __name__ == '__main__':
    unittest.main()
