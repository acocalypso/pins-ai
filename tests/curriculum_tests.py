"""Train-only curriculum isolation, language/device coverage and current-schema checks."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
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


if __name__ == '__main__':
    unittest.main()
