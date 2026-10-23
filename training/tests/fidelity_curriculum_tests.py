"""CPU-only independent curriculum checks; never read held-out examples."""
from collections import Counter
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from chat_dataset_common import LANGUAGES, validate_chat_call
from fidelity_curriculum import build_rows, coverage


def last_call(row):
    calls = row['messages'][-1].get('tool_calls', [])
    return calls[0]['function'] if calls else None


class FidelityCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Enforce train generation's read boundary, including helpers. The only
        # external input allowed is the public compiled MCP schema snapshot.
        original = Path.read_text
        reads = []

        def guarded(path, *args, **kwargs):
            reads.append(path.resolve())
            if path.resolve() != (ROOT / 'training/dataset/contracts/tools-list.json').resolve():
                raise AssertionError('Unexpected curriculum input: ' + str(path))
            return original(path, *args, **kwargs)

        with patch.object(Path, 'read_text', guarded):
            cls.rows = build_rows()
        cls.reads = reads
        cls.by_id = {row['id']: row for row in cls.rows}

    def row(self, group, locale='en'):
        return self.by_id['v9-fidelity-' + group + '-' + locale]

    def test_bounded_balanced_unique_and_train_only(self):
        self.assertEqual(len(self.rows), 850)
        self.assertEqual(len(self.by_id), len(self.rows))
        self.assertEqual(Counter(r['locale'] for r in self.rows), dict.fromkeys(LANGUAGES, 85))
        self.assertTrue(self.reads)
        for row in self.rows:
            self.assertLessEqual(len(row['tools']), 2)
            self.assertEqual(row['provenance']['split'], 'train')
            self.assertFalse(row['provenance']['heldoutInputs'])
            self.assertFalse(row['provenance']['hardwareExecution'])
            self.assertEqual(row['provenance']['nativeReview'], 'pending')

    def test_every_assistant_call_is_current_gateway_schema_valid(self):
        for row in self.rows:
            names = [t['name'] for t in row['tools']]
            for message in row['messages']:
                for call in message.get('tool_calls', []):
                    fn = call['function']
                    validate_chat_call(fn['name'], fn['arguments'], names)
                    self.assertFalse(any(k in fn['arguments'] for k in
                                         ('force', 'overwrite', 'abandon', 'skip_validation')))
                    self.assertLessEqual(fn['arguments'].get('wait_seconds', 0), 5)

    def test_no_tool_at_authority_missing_invalid_withdrawal_unknown_turns(self):
        categories = {'authority', 'codeswitch', 'missing_or_partial', 'invalid_bound',
                      'withdrawal', 'job_unknown', 'job_terminal_explain'}
        for row in self.rows:
            if row['category'].removeprefix('fidelity_') in categories:
                self.assertIsNone(last_call(row), row['id'])
                self.assertTrue(row['messages'][-1]['content'])

    def test_authority_does_not_remove_legitimate_positive_capability(self):
        for locale in LANGUAGES:
            for key, name, action in (('park', 'park_mount', None), ('open', 'control_dome', 'open_shutter')):
                positive = last_call(self.row('action-' + key, locale))
                self.assertEqual(positive['name'], name)
                if action:
                    self.assertEqual(positive['arguments']['action'], action)
                for source in ('note', 'quote'):
                    self.assertIsNone(last_call(self.row('authority-' + source + '-' + key, locale)))
            self.assertEqual(last_call(self.row('authority-real-request', locale))['arguments'], {'action': 'open_shutter'})

    def test_opposing_actions_and_booleans_preserve_direction(self):
        for locale in LANGUAGES:
            self.assertEqual(last_call(self.row('action-park', locale))['name'], 'park_mount')
            self.assertEqual(last_call(self.row('action-unpark', locale))['name'], 'unpark_mount')
            self.assertEqual(last_call(self.row('correct-to-unpark', locale))['name'], 'unpark_mount')
            for a, b in (('open', 'close'), ('cover_open', 'cover_close'), ('light_on', 'light_off')):
                self.assertNotEqual(last_call(self.row('action-' + a, locale))['arguments'],
                                    last_call(self.row('action-' + b, locale))['arguments'])
            for group, field, flag in (('mechanical', 'mechanical', True), ('sky', 'mechanical', False),
                                       ('log_previous', 'previous_run', True), ('log_current', 'previous_run', False),
                                       ('samples_yes', 'include_samples', True), ('samples_no', 'include_samples', False)):
                self.assertIs(last_call(self.row(group, locale))['arguments'][field], flag)

    def test_zero_omission_and_boundary_values_stay_distinct(self):
        for locale in LANGUAGES:
            for omitted, explicit, key in (('omission-guider', 'steps-0', 'steps'),
                                           ('omission-warm', 'warm-0', 'duration_minutes')):
                self.assertEqual(last_call(self.row(omitted, locale))['arguments'], {})
                self.assertEqual(last_call(self.row(explicit, locale))['arguments'][key], 0)
            self.assertEqual(last_call(self.row('history-0', locale))['arguments'], {'history': 0})
            self.assertEqual(last_call(self.row('history-10', locale))['arguments'], {'history': 10})
            self.assertEqual(last_call(self.row('center-0', locale))['arguments'], {'ra': 0, 'dec': -90, 'wait_seconds': 0})
            self.assertEqual(last_call(self.row('find-50', locale))['arguments']['limit'], 50)
            self.assertEqual(last_call(self.row('warm-60', locale))['arguments']['duration_minutes'], 60)

    def test_enum_and_argument_coverage_are_explicit(self):
        report = coverage(self.rows)
        for key, expected in {
            'get_events.kinds': {'safety', 'device', 'sequence', 'image', 'camera', 'guider', 'mount', 'camera,mount'},
            'get_events.min_severity': {'info', 'warning', 'error', 'critical'},
            'get_session_summary.group_by': {'target_filter', 'target', 'filter'},
            'get_session_summary.image_type': {'light', 'all'},
            'plan_targets.darkness': {'nautical', 'astronomical'},
        }.items():
            self.assertEqual({json.loads(v) for v in report['argumentValues'][key]}, expected)
        self.assertIn('get_events.kinds', report['optionalOmissions'])
        for row in self.rows:
            call = last_call(row)
            if call and call['name'] == 'plan_targets':
                self.assertEqual([t['name'] for t in call['arguments']['targets']], ['Cedar', 'Flint'])
                self.assertEqual([t['ra'] for t in call['arguments']['targets']], [7.4, 16.8])
            if call and call['name'] == 'get_guide_incident':
                self.assertEqual(call['arguments']['id'], 'guide-f2a7-004C')

    def test_pending_terminal_and_unknown_jobs_keep_observed_identity(self):
        for row in self.rows:
            if row['category'] not in ('fidelity_job_pending', 'fidelity_job_terminal', 'fidelity_job_terminal_explain'):
                continue
            results = [json.loads(m['content']) for m in row['messages'] if m['role'] == 'tool']
            observed = results[-1]['job']
            call = last_call(row)
            if call:
                self.assertEqual(call['name'], 'get_job')
                self.assertEqual(call['arguments'], {'job_id': observed['id'], 'wait_seconds': 0})
                self.assertNotIn('device', call['arguments'])
            else:
                self.assertIn(observed['id'], row['messages'][-1]['content'])
                self.assertIn(observed['state'], row['messages'][-1]['content'])
            start = next(c['function'] for m in row['messages'] for c in m.get('tool_calls', [])
                         if c['function']['name'] == 'connect_device')
            self.assertEqual(start['arguments']['device'], observed['device'])

    def test_rebuild_is_deterministic(self):
        self.assertEqual(self.rows, build_rows())


if __name__ == '__main__':
    unittest.main()
