"""No GPU, downloads, equipment operations, or historical predictions as training seeds."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'training'))
from analyze_generation_validation import analyze, summarize_results, diagnostics, research_candidates
from reliability_selection import metrics, production_eligibility, make_progress_callback, retain_candidates
from reliability_curriculum import build_rows
from chat_dataset_common import LANGUAGES, validate_chat_call
from audit_split_isolation import audit


def result(i=0, expected='get_status', correct=True, unwanted=False):
    call={'name':'get_status','arguments':{}} if unwanted or expected else None
    return {'id':str(i),'locale':'en','category':'fixture','expected':{'first_tool':expected,'arguments':{}},
      'caseSha256':'case'+str(i),'promptSha256':'prompt'+str(i),'call':call,'text':'fixture',
      'schemaValid':True,'toolChoiceCorrect':correct,'exactArgumentsCorrect':correct,'outputTruncated':False}


def report(step,unwanted,exact):
    rows=[result(i,None,i>=unwanted,i<unwanted) if i<50 else result(i,correct=i<exact+unwanted) for i in range(146)]
    # Explicit exact totals, independent from call/no-call error flags for ranking fixtures.
    for i,row in enumerate(rows): row['exactArgumentsCorrect']=i>=unwanted and i<unwanted+exact
    return {'step':step,'results':rows,'contract':{'caseIds':[r['id'] for r in rows],'caseCount':146,'scorer':'fixture'},
            'metrics':{'historical':'kept'},'completed':True,'hardwareExecution':False}


class ReliabilityTests(unittest.TestCase):
    def test_analyzer_counts_sequence_followups_as_required_tools(self):
        row=result(expected=None)
        row['expected']={'tool_sequence':['get_job'],'arguments':[{'job_id':'synthetic'}]}
        row['call']={'name':'get_job','arguments':{'job_id':'synthetic'}}
        summary=summarize_results([row])
        self.assertEqual(summary['expectedNoTool'],0)
        self.assertEqual(summary['unwantedProposals'],0)

    def test_analyzer_contract_mismatch_and_paired_changes(self):
        with tempfile.TemporaryDirectory() as temp:
            paths=[Path(temp)/f'{s}.json' for s in (300,600)]
            a,b=report(300,9,110),report(600,10,117)
            for p,r in zip(paths,[a,b]):p.write_text(json.dumps(r),encoding='utf-8')
            output=analyze(paths)
            self.assertEqual(output['checkpoints'][0]['summary']['exactCorrect'],110)
            self.assertTrue(output['pairedTransitions'][0]['improved'])
            self.assertEqual(output['researchCandidates']['bestSafetyStep'],300)
            self.assertEqual(output['researchCandidates']['bestAccuracyStep'],600)
            self.assertEqual(output['researchCandidates']['productionEligibleSteps'],[])
            b['contract']['scorer']='changed';paths[1].write_text(json.dumps(b),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'contract'):analyze(paths)

    def test_invalid_no_tool_is_separate_from_unwanted_proposal(self):
        row=result(expected=None);row.update(schemaValid=False,text='invalid bare output',call=None)
        summary=summarize_results([row])
        self.assertEqual(summary['invalidNoToolOutputs'],1)
        self.assertEqual(summary['unwantedProposals'],0)
        row['text']='<tool_call>broken'
        self.assertIn('unwanted_proposal',diagnostics(row))

    def test_v8_rank_is_safety_first_but_research_finds_accuracy(self):
        legacy={'selection_score':-1,'unsafe_proposals':0,'tool_macro_exact':.72}
        with self.assertRaisesRegex(ValueError,'Empty'):
            metrics([],legacy)
        a=metrics(report(300,9,110)['results'],legacy)
        b=metrics(report(600,10,117)['results'],dict(legacy,tool_macro_exact=.802))
        self.assertGreater(a['selection_score'],b['selection_score'])
        self.assertFalse(production_eligibility(a))
        self.assertFalse(production_eligibility(b,reviewed_independent_holdout=True,unauthorized_executions=0,
                                              significant_schema_regression=False,quantized_runtime_passed=True))
        clean=metrics([result()],dict(legacy,tool_macro_exact=1))
        self.assertFalse(production_eligibility(clean))
        self.assertTrue(production_eligibility(clean,reviewed_independent_holdout=True,unauthorized_executions=0,
                                             significant_schema_regression=False,quantized_runtime_passed=True))

    def test_early_stopping_exports_and_restores_research_progress(self):
        from types import SimpleNamespace
        cb=make_progress_callback(4,.005); control=SimpleNamespace(should_training_stop=False)
        for safety,accuracy in [(-10,.72),(-11,.788),(-11,.778),(-12,.802)]:
            cb.on_evaluate(None,None,control,{'eval_selection_score':safety,'eval_tool_macro_exact':accuracy})
        self.assertEqual(cb.counter,0)
        exported=cb.state();restored=cb.__class__(**exported['args'])
        for key,value in exported['attributes'].items():setattr(restored,key,value)
        for _ in range(4):restored.on_evaluate(None,None,control,{'eval_selection_score':-12,'eval_tool_macro_exact':.802})
        self.assertTrue(control.should_training_stop)

    def test_new_data_valid_multilingual_no_eval_inputs(self):
        rows=build_rows()
        self.assertEqual({r['locale'] for r in rows},set(LANGUAGES))
        for r in rows:
            self.assertEqual(r['provenance']['split'],'train')
            for m in r['messages']:
                for c in m.get('tool_calls',[]):
                    f=c['function'];validate_chat_call(f['name'],f['arguments'],[t['name'] for t in r['tools']])
                    for key in ('history','since_hours','lines','steps','alerts','limit','exposure_seconds','duration_minutes'):
                        if key in f['arguments'] and len(r['messages']) == 3:
                            self.assertIn(str(f['arguments'][key]),r['messages'][1]['content'],r['id'])
        source=(ROOT/'training/reliability_curriculum.py').read_text(encoding='utf-8')
        self.assertNotIn('read_jsonl',source)
        self.assertNotIn('generation-validation',source)
        for locale in LANGUAGES:
            actions={c['function']['arguments']['action'] for r in rows if r['locale']==locale for m in r['messages']
                     for c in m.get('tool_calls',[]) if c['function']['name']=='control_flat_device'}
            self.assertEqual(actions,{'light_on','light_off'})

    def test_research_weights_are_bounded_and_original_saves_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            full=Path(temp)
            (full/'training-run.json').write_text('{"status":"started"}',encoding='utf-8')
            (full/'generation-validation').mkdir()
            for step,bad,exact in [(300,9,110),(600,10,117),(700,11,120)]:
                source=full/'checkpoints'/f'checkpoint-{step}';source.mkdir(parents=True)
                for name in ('adapter_config.json','adapter_model.safetensors','trainer_state.json'):
                    (source/name).write_bytes(b'fixture-only')
                (full/'generation-validation'/f'step-{step:06d}.json').write_text(json.dumps(report(step,bad,exact)),encoding='utf-8')
                retain_candidates(full)
            kept=full/'research-candidates/checkpoints'
            self.assertEqual({p.name for p in kept.iterdir()},{'checkpoint-300','checkpoint-700'})
            self.assertTrue((full/'checkpoints/checkpoint-600/adapter_model.safetensors').is_file())

    def test_leak_detector_links_translation_groups_and_numeric_near_duplicates(self):
        r={'id':'reliability-x-en','group':'linked','locale':'en','messages':[{'role':'user','content':'Read twenty recent camera warning events for me'}]}
        other=copy.deepcopy(r);other['id']='held-de';other['locale']='de'
        self.assertTrue(audit([r],{'held':[other]})['linkedGroupLeaks'])
        r['messages'][0]['content']='Read 16 recent camera warning events for me'
        other=copy.deepcopy(r);other['id']='held-en';other['group']='different'
        other['messages'][0]['content']='Read 37 recent camera warning events for me'
        self.assertTrue(audit([r],{'held':[other]})['newNearDuplicates'])

    def test_future_holdout_is_separate_frozen_and_never_exposes_expected_answers(self):
        from evaluation_state import select_cases
        from common import read_jsonl
        with self.assertRaisesRegex(ValueError,'chat'):
            select_cases('future_holdout')
        held,path,_=select_cases('future_holdout',prompt_mode='chat')
        train=read_jsonl(ROOT/'training/dataset/train.jsonl')
        self.assertEqual(len(held),40)
        self.assertEqual({r['locale'] for r in held},set(LANGUAGES))
        self.assertFalse({r['group'] for r in held}&{r['group'] for r in train})
        self.assertTrue(all([m['role'] for m in r['messages']]==['system','user'] for r in held))


if __name__=='__main__':unittest.main()
