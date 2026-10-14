"""Natural chat coverage, runtime alignment, bounded scope and source isolation."""
import json
import re
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training'))
from common import fingerprint, read_jsonl, sha256
from build_chat_dataset import build_rows, coverage, source_hashes
from chat_curriculum import DATA, DENIAL, PROFILE, curriculum_rows, has_json_request, project
from chat_dataset_common import CHAT_POLICY, LANGUAGES, chat_definitions, chat_tools, validate_chat_call, conversation, tool_call, tool_result, user, assistant
from v2_contracts import PACKAGE, definitions, validate_call, compact_tools


class ChatDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = curriculum_rows('train')
        cls.validation = curriculum_rows('validation')
        cls.manifest = json.loads((DATA / 'manifest.json').read_text(encoding='utf-8'))

    def test_single_dataset_entry_point_and_package_wrapper(self):
        import subprocess
        self.assertEqual(DATA, ROOT/'training/dataset')
        self.assertFalse((ROOT/'training/datasets').exists())
        self.assertFalse((ROOT/'training/data').exists())
        result = subprocess.run([sys.executable, str(DATA/'scripts/prepare_qwen3_sft.py'), '--dry-run'],
                                capture_output=True, encoding='utf-8', check=True)
        self.assertEqual(json.loads(result.stdout), self.rows[0])
        import importlib.util
        spec=importlib.util.spec_from_file_location('unified_entrypoint', DATA/'scripts/prepare_qwen3_sft.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        self.assertEqual(module.REPOSITORY, ROOT)

    def test_reproducibility_provenance_and_no_heldout_generator_inputs(self):
        self.assertEqual(self.rows, build_rows()[0])
        self.assertEqual(self.manifest['sourceHashes'], source_hashes())
        self.assertEqual(self.manifest['trainSha256'], sha256(DATA / 'train.jsonl'))
        self.assertEqual(self.manifest['policySha256'], fingerprint(CHAT_POLICY))
        self.assertEqual(self.manifest['heldOutInputs'], [])
        self.assertFalse(self.manifest['hardwareExecution'])
        self.assertEqual(len(self.rows), len({r['id'] for r in self.rows}))
        for path in source_hashes():
            self.assertFalse(any(s in path for s in ('/test.jsonl', 'eval_gold', 'hard_gold')))

    def test_plain_chat_positive_coverage_every_tool_in_ten_languages(self):
        report = coverage(self.rows)
        self.assertEqual(set(report['tools']), set(definitions()))
        self.assertLess(report['jsonInUserConversations'] / len(self.rows), .35)
        self.assertGreater(report['multipleModifyingTools'], 0)
        self.assertGreater(report['multiUserTurnConversations'], 200)
        for name, metrics in report['tools'].items():
            self.assertEqual(set(metrics['naturalIntentLocales']), set(LANGUAGES), name)
            self.assertEqual(metrics['uncoveredArguments'], [], name)
        for row in self.rows:
            if row['category'] == 'natural_chat_intent':
                self.assertFalse(has_json_request(row), row['id'])
                request = row['messages'][1]['content']
                self.assertFalse(any(n in request for n in definitions()), row['id'])

    def test_authored_unicode_text_is_intact(self):
        for row in self.rows:
            if row['id'].startswith(('chat-intent-', 'chat-workflow-', 'chat-generalization-')):
                for message in row['messages']:
                    self.assertIsNone(re.search(r'\?{2,}|\uFFFD|\u00C3[\u0080-\u00BF]|\u00E2\u20AC', message['content']), row['id'])

    def test_no_approval_answer_keys_and_native_proposals_need_no_approval(self):
        for row in self.rows + self.validation:
            system = row['messages'][0]['content']
            self.assertTrue(system.startswith(CHAT_POLICY), row['id'])
            self.assertNotIn('Trusted gateway authorization', system)
            self.assertNotIn('Synthetic exact application approval', system)
            self.assertNotIn('application approval for this exact proposal exists', system)
            self.assertNotIn('"arguments"', system)
        positives = [r for r in self.rows if r['category'] == 'natural_chat_intent']
        self.assertEqual(len(positives), len(definitions()) * len(LANGUAGES))
        self.assertTrue(any(r['expected']['first_tool'] == 'add_target' for r in positives))

    def test_every_native_call_is_valid_bounded_and_never_uses_override(self):
        for row in self.rows + self.validation:
            self.assertLessEqual(len(row['tools']), 4, row['id'])
            names = [t['name'] for t in row['tools']]
            for message in row['messages']:
                self.assertLessEqual(len(message.get('tool_calls', [])), 1)
                for call in message.get('tool_calls', []):
                    fn = call['function']
                    validate_call(fn['name'], fn['arguments'], names)
                    validate_chat_call(fn['name'], fn['arguments'], names)
                    self.assertFalse(any(fn['arguments'].get(k) is True for k in
                                         ('force', 'overwrite', 'skip_validation', 'abandon')), row['id'])

    def test_validation_projection_preserves_source_and_has_real_denials(self):
        original = read_jsonl(PACKAGE / 'data/audited/validation.jsonl')
        self.assertGreater(len(self.validation), len(original))
        self.assertEqual([r['id'] for r in original], [r['id'] for r in self.validation[:len(original)]])
        self.assertEqual(self.validation[:len(original)], [dict(project(r), provenance={**project(r)['provenance'], 'split': 'validation'})
                                          for r in original])
        for row in self.rows + self.validation:
            if row['category'] in ('no_approval', 'comprehensive_approval', 'comprehensive_revoked', 'safety_explicit_denial'):
                self.assertTrue(any(DENIAL[row['locale']] in m['content'] for m in row['messages'] if m['role'] == 'user'))
                self.assertFalse(row['messages'][-1].get('tool_calls'))
        with self.assertRaises(ValueError): curriculum_rows('test')
        with self.assertRaises(ValueError): curriculum_rows('eval_gold')

    def test_heldout_groups_and_exact_requests_remain_separate(self):
        # Validators may read held-out prompts solely for collision detection;
        # no answers are used as augmentation seeds.
        groups = {r['group'] for r in self.rows}
        prompts = {(r['locale'], next(m['content'] for m in r['messages'] if m['role'] == 'user').strip().casefold())
                   for r in self.rows}
        for split in ('validation', 'test', 'eval_gold', 'hard_gold'):
            rows = read_jsonl(PACKAGE / 'data' / (split + '.jsonl'))
            self.assertFalse(groups & {r.get('group') for r in rows}, split)
            other = {(r['locale'], next(m['content'] for m in r['messages'] if m['role'] == 'user').strip().casefold())
                     for r in rows if r.get('messages')}
            self.assertFalse(prompts & other, split + ' exact prompt collision')

    def test_compact_descriptions_match_runtime_contract(self):
        catalog = definitions()
        for tool in compact_tools(list(catalog)):
            fn = tool['function']
            if fn['name'] != 'add_target':
                self.assertEqual(fn['description'], catalog[fn['name']]['description'][:180])

    def test_bounded_ai_schemas_and_job_identifier_provenance(self):
        for tool in chat_tools(list(definitions())):
            props = tool['function']['parameters']['properties']
            self.assertFalse(set(props) & {'force', 'skip_validation', 'overwrite', 'abandon'})
            if 'wait_seconds' in props:
                self.assertEqual(props['wait_seconds']['maximum'], 5)
                self.assertEqual(props['wait_seconds']['default'], 0)
            for key in ('ra', 'dec'):
                if key in props:
                    self.assertEqual(props[key]['type'], 'number')
        with self.assertRaises(Exception): tool_call('get_job', {'job_id': 'job-test', 'wait_seconds': 6})
        with self.assertRaises(ValueError):
            conversation('invented-job', 'en', 'test', ['get_job'],
                         [user('Is it finished?'), tool_call('get_job', {'job_id': 'invented'})])
        with self.assertRaises(ValueError):
            conversation('shortened-id', 'en', 'test', ['get_job'],
                         [user('Check job-focus-17'), tool_call('get_job', {'job_id': 'job-focus'})])
        with self.assertRaises(ValueError):
            conversation('wrong-result-id', 'en', 'test', ['get_job'],
                         [user('Check job-focus-17'), tool_call('get_job', {'job_id': 'job-focus-17'}),
                          tool_result('get_job', {'job': {'id': 'job-focus-99', 'state': 'running'}}),
                          assistant('Still running.')])

    def test_parity_with_actual_gateway_exposed_schemas(self):
        path = ROOT / 'artifacts/router-chat-v7.json'
        if not path.is_file():
            self.skipTest('Run the offline RouterAudit command to produce actual gateway schemas')
        report = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(report['inputs']['trainSha256'], sha256(DATA / 'train.jsonl'),
                         'Re-run RouterAudit after changing the chat corpus')
        self.assertEqual(report['inputs']['catalogSha256'], sha256(__import__('v2_contracts').VERIFIED))
        calls = sum(len(m.get('tool_calls', [])) for r in self.rows for m in r['messages'])
        self.assertEqual(report['callValidation']['callsChecked'], calls)
        self.assertFalse(report['hardwareExecution'])
        self.assertFalse(report['modelInference'])
        self.assertEqual(report['callValidation']['errors'], 0)
        actual = {t['name']: t['inputSchema'] for t in report['exposedTools']}
        expected = {n: t['inputSchema'] for n, t in chat_definitions().items()}
        self.assertEqual(actual, expected)

    def test_multilingual_argument_generalization_and_withdrawals(self):
        variants = [r for r in self.rows if r['category'] == 'multilingual_parameters']
        self.assertEqual(len(variants), 390)
        self.assertEqual({r['locale'] for r in variants}, set(LANGUAGES))
        for row in variants:
            request = next(m['content'] for m in row['messages'] if m['role'] == 'user')
            self.assertFalse(has_json_request(row))
            fn = row['messages'][-1]['tool_calls'][0]['function']
            args = fn['arguments']
            variant = int(row['group'].rsplit('-', 1)[1])
            if 'capture_minutes' in row['group']:
                self.assertEqual(args['exposure_seconds'], (1.25, 2.75, 4.5)[variant] * 60)
            if 'slew_degrees' in row['group']:
                self.assertEqual(args['ra'], (199.125, 72.375, 308.625)[variant] / 15)
            if fn['name'] == 'set_switch':
                self.assertEqual(args['index'], (6, 9, 12)[variant])
                self.assertEqual(args['value'], (37.25, 62.5, 0)[variant])
            if fn['name'] == 'get_job': self.assertIn(args['job_id'], request)
        denials = [r for r in self.rows if r['category'] == 'multilingual_withdrawal']
        modifying = {n for n, d in definitions().items() if not d['annotations']['readOnlyHint']}
        self.assertEqual(len(denials), len(modifying) * len(LANGUAGES))
        for name in modifying:
            self.assertEqual({r['locale'] for r in denials if r['group'].endswith('-' + name)}, set(LANGUAGES))
        self.assertTrue(all(r['expected']['first_tool'] is None and not r['messages'][-1].get('tool_calls') for r in denials))
        histories = [r for r in self.rows if r['category'] == 'multilingual_job_history']
        self.assertEqual(len(histories), len(LANGUAGES))
        for row in histories:
            calls = [c['function'] for m in row['messages'] for c in m.get('tool_calls', [])]
            self.assertEqual([c['name'] for c in calls], ['get_job', 'get_job'])
            self.assertEqual(calls[0]['arguments']['job_id'], calls[1]['arguments']['job_id'])

    def test_new_result_fixtures_match_actual_mcp_shapes(self):
        for row in self.rows:
            if not row['id'].startswith(('chat-intent-', 'chat-workflow-')):
                continue
            for message in row['messages']:
                if message['role'] != 'tool':
                    continue
                payload = json.loads(message['content'])
                if payload.get('isError') and 'job' not in payload:
                    continue
                if message['name'] == 'list_target_templates':
                    self.assertTrue(all(isinstance(t, str) for t in payload['templates']))
                if message['name'] == 'get_sequence' and payload.get('initialized'):
                    self.assertIn('nodes', payload)
                    self.assertNotIn('sequence', payload)
                jobs = payload.get('jobs', []) + ([payload['job']] if 'job' in payload else [])
                for job in jobs:
                    self.assertIn(job['state'], ('running', 'succeeded', 'failed', 'cancelled', 'abandoned'))
                    self.assertIn('kind', job)
                    self.assertNotIn('tool', job)
                    if payload.get('completionConfirmed'):
                        self.assertEqual(job['state'], 'succeeded')
                        self.assertIs(job.get('outcome', {}).get('success'), True)

    def test_builder_rejects_invalid_or_oversized_tool_scopes(self):
        names = list(definitions())[:5]
        with self.assertRaises(ValueError):
            conversation('invalid', 'en', 'test', names, [user('Read status'), tool_call('get_status', {})])
        with self.assertRaises(Exception): tool_call('get_status', {'invented': 1})
        with self.assertRaises(ValueError): tool_call('park_mount', {'force': True})

    def test_chat_validation_and_scorer_use_proposal_contract(self):
        from evaluation_state import select_cases, score
        rows, _, _ = select_cases('validation_smoke', per_language=2, prompt_mode='chat')
        self.assertTrue(all(r['messages'][0]['content'].startswith(CHAT_POLICY) for r in rows))
        self.assertTrue(all(len(r['tools']) <= 4 for r in rows))
        with self.assertRaises(ValueError): select_cases('hard_gold', prompt_mode='chat')
        case = {'expected': {'first_tool': 'get_job', 'arguments': {'job_id': 'job-known', 'wait_seconds': 5}},
                'allowedNames': ['get_job'], 'toolMode': 'chat'}
        good = '<tool_call>{"name":"get_job","arguments":{"job_id":"job-known","wait_seconds":5}}</tool_call>'
        self.assertTrue(score(case, good)['exactArgumentsCorrect'])
        self.assertFalse(score(case, good.replace('":5', '":6'))['schemaValid'])


if __name__ == '__main__':
    unittest.main()
