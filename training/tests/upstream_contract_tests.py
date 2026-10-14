"""MCP48 source transition, new semantics, frozen bytes and split isolation."""
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training'))
from common import sha256
from v2_contracts import AUDIT, definitions, legacy_definitions, validate_call
from chat_dataset_common import chat_definitions, validate_chat_call
from upstream_tool_curriculum import ARGS, build_rows, validation_rows


class UpstreamContracts(unittest.TestCase):
    def test_source_transition_and_frozen_bytes(self):
        audit=json.loads(AUDIT.read_text(encoding='utf-8'))
        self.assertEqual(len(definitions()),48)
        self.assertEqual(set(audit['added']),set(definitions())-set(legacy_definitions()))
        self.assertEqual(audit['schemaChanges'],['add_target'])
        self.assertFalse(audit['hardwareExecution'])
        for path,digest in audit['frozenSourceAndEvaluationHashes'].items():
            self.assertEqual(sha256(ROOT/path),digest,path)

    def test_read_annotations_and_override_exposure(self):
        reads={'get_guider_report','get_guiding_coach_report','list_guide_incidents','get_guide_incident',
               'read_log','find_targets','plan_targets','get_session_summary','get_system_health','get_events'}
        for name in ARGS:
            self.assertEqual(definitions()[name]['annotations']['readOnlyHint'],name in reads)
            validate_chat_call(name,ARGS[name],definitions())
        self.assertNotIn('close_dome_without_mount',chat_definitions()['secure_observatory']['inputSchema']['properties'])

    def test_crossfield_errors_not_encoded_as_positive_calls(self):
        invalid=[('find_targets',{'limit':5}),('find_targets',{'min_size_arcmin':100,'max_size_arcmin':10}),
          ('get_events',{'wait_seconds':1}),('control_dome',{'action':'slew_to_azimuth'}),
          ('control_dome',{'action':'open_shutter','azimuth':15}),('control_flat_device',{'action':'set_brightness'}),
          ('control_flat_device',{'action':'light_on','brightness':5}),
          ('secure_observatory',{'close_dome_without_mount':True}),
          ('plan_targets',{'targets':[{'name':'A','ra':1,'dec':2}],'date':'2026-02-30'}),
          ('plan_targets',{'targets':[{'name':'A','ra':'01:00:00','dec':2}]}),
          ('add_target',{'name':'A','ra':1,'dec':2,'template':'A','meridian_flip':False})]
        for name,args in invalid:
            with self.subTest(name=name,args=args), self.assertRaises(ValueError):
                validate_call(name,args,definitions())

    def test_independent_development_split_and_valid_calls(self):
        train,dev=build_rows(),validation_rows()
        self.assertFalse({r['group'] for r in train}&{r['group'] for r in dev})
        prompts=lambda rows:{(r['locale'],m['content']) for r in rows for m in r['messages'] if m['role']=='user'}
        self.assertFalse(prompts(train)&prompts(dev))
        self.assertEqual({r['expected']['first_tool'] for r in dev if r['expected']['first_tool']},set(ARGS))
        for r in train+dev:
            self.assertFalse(r['provenance']['hardwareExecution'])
            for m in r['messages']:
                for c in m.get('tool_calls',[]):
                    fn=c['function'];validate_chat_call(fn['name'],fn['arguments'],[t['name'] for t in r['tools']])


if __name__=='__main__':unittest.main()
