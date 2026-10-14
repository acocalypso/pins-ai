"""Offline coverage, contract, result semantics and held-out isolation checks."""
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from common import fingerprint, read_jsonl, sha256
from build_comprehensive_dataset import build_rows, coverage, DESTINATION, LANGUAGES, source_sha
from v2_contracts import PACKAGE, LEGACY_VERIFIED as VERIFIED, POLICY, legacy_definitions as definitions, validate_call


class ComprehensiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = read_jsonl(DESTINATION / 'train.jsonl')
        cls.manifest = json.loads((DESTINATION / 'manifest.json').read_text())

    def test_reproducible_checked_in_artifact_and_hashes(self):
        self.assertEqual(self.rows, build_rows())
        self.assertEqual(self.manifest['trainSha256'], sha256(DESTINATION / 'train.jsonl'))
        self.assertEqual(self.manifest['contractSha256'], sha256(VERIFIED))
        self.assertEqual(self.manifest['policySha256'], fingerprint(POLICY))
        self.assertEqual(self.manifest['coverageSha256'], sha256(DESTINATION / 'coverage.json'))
        self.assertEqual(self.manifest['heldOutInputs'], [])
        # Generator source hashing ignores checkout line endings, as Git normalizes code.
        self.assertEqual(self.manifest['languageSourceSha256'], source_sha(PACKAGE / 'scripts/build_dataset.py'))
        self.assertEqual(self.manifest['generatorSha256'], source_sha(ROOT / 'training/build_comprehensive_dataset.py'))

    def test_tool_language_argument_coverage(self):
        report = coverage(self.rows)
        self.assertEqual(set(report['tools']), set(definitions()))
        for tool in report['tools'].values():
            self.assertEqual(set(tool['languages']), set(LANGUAGES))
            self.assertEqual(tool['uncoveredArguments'], [])
        devices = definitions()['connect_device']['inputSchema']['properties']['device']['enum']
        matrix = {(r['locale'], fn['name'], fn['arguments']['device'])
                  for r in self.rows for m in r['messages'] for c in m.get('tool_calls', [])
                  for fn in [c['function']] if fn['name'] in ('connect_device', 'disconnect_device')}
        self.assertEqual(len(matrix), len(devices) * 2 * len(LANGUAGES))

    def test_every_call_is_exposed_valid_and_mutations_have_exact_approval(self):
        catalog = definitions()
        for row in self.rows:
            self.assertFalse(row['provenance']['hardwareExecution'])
            self.assertEqual(row['provenance']['split'], 'train')
            first_user = next(i for i, m in enumerate(row['messages']) if m['role'] == 'user')
            self.assertEqual(first_user, 1)
            for i, message in enumerate(row['messages']):
                for call in message.get('tool_calls', []):
                    fn = call['function']
                    validate_call(fn['name'], fn['arguments'], [t['name'] for t in row['tools']])
                    self.assertFalse(any(fn['arguments'].get(k) is True for k in ('force', 'overwrite', 'skip_validation', 'abandon')))
                    if not catalog[fn['name']]['annotations']['readOnlyHint']:
                        if row['group'].startswith('comprehensive-get_status-cancellation-'):
                            self.assertIn('Historical cancellation approved and consumed', row['messages'][0]['content'])
                        else:
                            self.assertIn('Synthetic exact application approval for ' + fn['name'], row['messages'][0]['content'])
                    # Feedback, when supplied, is paired with the actual preceding tool.
                    if i + 1 < len(row['messages']) and row['messages'][i + 1]['role'] == 'tool':
                        self.assertEqual(row['messages'][i + 1]['name'], fn['name'])

    def test_no_action_boundaries_and_failures_do_not_retry(self):
        for row in self.rows:
            if row['category'] in ('comprehensive_boundary', 'comprehensive_missing', 'comprehensive_crossfield',
                                   'comprehensive_approval', 'comprehensive_busy', 'comprehensive_unavailable',
                                   'comprehensive_unsupported', 'comprehensive_clarify'):
                self.assertFalse(any(m.get('tool_calls') for m in row['messages']), row['id'])
            if row['category'] == 'comprehensive_poll':
                names = row['expected']['tool_sequence']
                self.assertEqual(names, [row['capability'], 'get_job'])
                running = json.loads(row['messages'][3]['content'])['job']
                poll = row['messages'][4]['tool_calls'][0]['function']['arguments']
                self.assertEqual(poll['job_id'], running['id'])
                final = json.loads(row['messages'][5]['content'])['job']
                self.assertEqual(final['outcome']['success'], final['state'] == 'succeeded')

    def test_heldout_group_and_exact_prompt_isolation(self):
        # Held-out files are read only by this validator, never by the generator.
        groups = {r['group'] for r in self.rows}
        prompts = {(r['locale'], next(m['content'] for m in r['messages'] if m['role'] == 'user').strip().casefold()) for r in self.rows}
        for name in ('validation', 'test', 'eval_gold', 'hard_gold'):
            heldout = read_jsonl(PACKAGE / 'data' / (name + '.jsonl'))
            self.assertFalse(groups & {r.get('group') for r in heldout})
            other = {(r['locale'], next(m['content'] for m in r['messages'] if m['role'] == 'user').strip().casefold())
                     for r in heldout if r.get('messages')}
            self.assertFalse(prompts & other, name + ': exact prompt collision')

    def test_grounded_readings_and_capture_save_uncertainty(self):
        for row in self.rows:
            if '-grounded-camera-' in row['group']:
                camera = json.loads(row['messages'][3]['content'])['devices'][0]
                answer = row['messages'][-1]['content']
                if camera['connected']:
                    self.assertIn(str(camera['properties']['temperature']), answer)
                    self.assertIn(str(camera['properties']['coolerPower']), answer)
                else:
                    self.assertFalse(camera['properties'])
            if row['capability'] == 'capture_image' and row['category'] == 'comprehensive_arguments':
                args = row['expected']['arguments']
                if args.get('save'):
                    payload = json.loads(row['messages'][3]['content'])
                    self.assertIsNone(payload['job']['outcome']['data']['filePath'])
                    from build_comprehensive_dataset import SAVE_QUEUED
                    self.assertEqual(row['messages'][-1]['content'], SAVE_QUEUED[row['locale']])


if __name__ == '__main__': unittest.main()
