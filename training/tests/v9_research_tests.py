"""Synthetic outcomes only; no held-out prompts, models or GPU execution."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reliability_selection import metrics, make_progress_callback, production_eligibility, retain_candidates
from research_frontier import frontier, VERSION


def outcomes(unwanted=5, positive=78, invalid_no_tool=1, invalid_required=2):
    rows = []
    for i in range(146):
        needed = i >= 50
        unexpected = not needed and i < unwanted
        valid = i >= invalid_no_tool if not needed else i >= 50 + invalid_required
        exact = i >= unwanted if not needed else i < 50 + positive
        rows.append({'id': str(i), 'locale': 'en', 'category': 'prompt_injection' if not needed else 'tool_success',
                     'caseSha256': str(i), 'promptSha256': str(i),
                     'expected': {'first_tool': 'get_status' if needed else None, 'arguments': {}},
                     'call': {'name': 'get_status', 'arguments': {}} if needed or unexpected else None,
                     'text': 'synthetic outcome', 'schemaValid': valid, 'toolChoiceCorrect': not unexpected,
                     'exactArgumentsCorrect': exact, 'outputTruncated': False})
    return rows


def report(step, rows):
    return {'step': step, 'results': rows, 'completed': True, 'hardwareExecution': False,
            'metrics': {}, 'contract': {'caseIds': [r['id'] for r in rows], 'caseCount': len(rows)}}


class ResearchTests(unittest.TestCase):
    def test_critical_and_positive_metrics_are_persisted_for_monitoring(self):
        from metrics import TrainingMetrics
        with tempfile.TemporaryDirectory() as directory:
            monitor=TrainingMetrics(directory)
            state=SimpleNamespace(global_step=100,max_steps=1740,epoch=.2)
            values={'eval_loss':.2,'eval_critical_action_reversals':1,'eval_invented_coordinates':2,
                    'eval_positive_tool_exact':.9,'eval_prompt_injection_unwanted':3,'eval_no_start_cases':10}
            monitor.on_log(None,state,None,values)
            saved=json.loads((Path(directory)/'metrics-latest.json').read_text())
            for key,value in values.items(): self.assertEqual(saved[key],value)

    def test_v8_600_outranks_500_without_reinterpreting_score(self):
        legacy = {'tool_macro_exact': .8244897959183674, 'unsafe_proposals': 5, 'selection_score': -1.1}
        a = metrics(outcomes(positive=79, invalid_no_tool=3), legacy)
        b = metrics(outcomes(), dict(legacy, tool_macro_exact=.8142857142857143))
        self.assertAlmostEqual(a['selection_score'], -15883061.587755103)
        self.assertAlmostEqual(b['selection_score'], -15882765.592857143)
        self.assertGreater(b['selection_score'], a['selection_score'])
        self.assertEqual((a['invalid_no_tool_outputs'], b['invalid_no_tool_outputs']), (3, 1))

    def test_early_stop_700_peak_stops_at_1100_and_restores_counter(self):
        cb = make_progress_callback(4, .005, VERSION)
        control = SimpleNamespace(should_training_stop=False)
        for step, safety, macro in [(600,-15e6,.814), (700,-34e6,.914), (800,-19e6,.855), (900,-34e6,.832)]:
            cb.on_evaluate(None,None,control,{'eval_selection_score':safety,'eval_tool_macro_exact':macro})
        saved = cb.state()
        restored = cb.__class__(**saved['args'])
        for key,value in saved['attributes'].items(): setattr(restored,key,value)
        restored.on_evaluate(None,None,control,{'eval_selection_score':-38e6,'eval_tool_macro_exact':.862})
        self.assertFalse(control.should_training_stop)
        restored.on_evaluate(None,None,control,{'eval_selection_score':-31e6,'eval_tool_macro_exact':.833})
        self.assertTrue(control.should_training_stop)
        self.assertEqual(restored.counter,4)

    def test_frontier_retains_three_distinct_research_roles(self):
        policy = report(6,outcomes(positive=78))
        capability = report(7,outcomes(unwanted=11,positive=88,invalid_no_tool=2))
        macro = report(8,outcomes(unwanted=6,positive=87,invalid_no_tool=2))
        # Identical case labels with differing success distribution over rare
        # and common tools produce three legitimate research tradeoffs.
        for r in (policy,capability,macro):
            for i,row in enumerate(r['results'][50:]):
                row['expected']['first_tool']='rare' if i < 2 else 'common'
                row['call']['name']=row['expected']['first_tool']
                if r is capability: row['exactArgumentsCorrect']=2 <= i < 90
        ranking=frontier([policy,capability,macro])
        self.assertEqual(ranking['bestPolicyStep'],6)
        self.assertEqual(ranking['bestPositiveStep'],7)
        self.assertEqual(ranking['bestMacroStep'],8)
        self.assertEqual(ranking['productionEligibleSteps'],[])
        self.assertEqual(ranking['retainedWeightSteps'],[6,7,8])
        self.assertFalse(ranking['latencyAxisIncluded'])
        # Rotation tests keep reports identical across steps, only outcome bits differ.
        with tempfile.TemporaryDirectory() as directory:
            full=Path(directory); (full/'generation-validation').mkdir()
            (full/'training-run.json').write_text('{}')
            for r in [policy,capability,macro]:
                cp=full/'checkpoints'/f"checkpoint-{r['step']}"; cp.mkdir(parents=True)
                for name in ('adapter_model.safetensors','adapter_config.json','trainer_state.json'):
                    (cp/name).write_bytes(b'synthetic fixture')
                (full/'generation-validation'/f"step-{r['step']:06d}.json").write_text(json.dumps(r))
                retain_candidates(full,VERSION)
            result=json.loads((full/'research-candidates/selection.json').read_text())
            self.assertEqual(result['version'],VERSION)
            self.assertEqual(result['retainedWeightSteps'],[6,7,8])
            self.assertTrue((full/'checkpoints/checkpoint-6/adapter_model.safetensors').is_file())

    def test_release_gate_rejects_unknown_or_critical_evidence(self):
        clean={'development_gate_passed':1,'critical_action_reversals':0,'invented_coordinates':0}
        evidence=dict(reviewed_independent_holdout=True,unauthorized_executions=0,
                      significant_schema_regression=False,quantized_runtime_passed=True)
        self.assertTrue(production_eligibility(clean,**evidence))
        self.assertFalse(production_eligibility(clean))
        for key in ('critical_action_reversals','invented_coordinates'):
            self.assertFalse(production_eligibility({**clean,key:1},**evidence))
            missing=dict(clean); missing.pop(key)
            self.assertFalse(production_eligibility(missing,**evidence))


if __name__=='__main__': unittest.main()
